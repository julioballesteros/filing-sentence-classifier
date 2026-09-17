"""Public TF-IDF inference from portable, verified synthetic bundles."""

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import warnings
from importlib.metadata import version
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.exceptions import InconsistentVersionWarning
from threadpoolctl import threadpool_info

import filing_sentence_classifier
from filing_sentence_classifier.artifacts import BundleError, BundleFile, BundleManifest
from filing_sentence_classifier.baselines.tfidf import (
    TfidfConfig,
    create_tfidf_pipeline,
    fit_tfidf,
    save_tfidf_model,
)
from filing_sentence_classifier.inference import predictor as public
from filing_sentence_classifier.inference import tfidf as backend
from filing_sentence_classifier.inference.contracts import (
    InputLimits,
    LabelSet,
    PredictionContractError,
    prepare_texts,
)
from filing_sentence_classifier.inference.predictor import Predictor

TRAIN_TEXTS = (
    "We expect growth",
    "We expect investment",
    "Profit increased yesterday",
    "Revenue increased yesterday",
    "Results may differ",
    "Outcomes may differ",
)
TARGETS = (0, 0, 1, 1, 2, 2)
TEXTS = (
    " We expect growth\n",
    "Revenue increased yesterday",
    "zzunknown",
    "!!!",
    "We expect growth",
)
CONFIG = b"""[tfidf]
ngram_range = [1, 2]
min_df = 1
[logistic_regression]
c = 1.0
max_iter = 1000
tol = 0.0001
"""
DEPENDENCIES = (
    "filing-sentence-classifier",
    "scikit-learn",
    "numpy",
    "scipy",
    "joblib",
    "threadpoolctl",
)


def write_manifest(directory, state):
    (directory / "manifest.json").write_text(json.dumps(state), encoding="utf-8")


def make_bundle(directory, num_classes=3):
    directory.mkdir(parents=True)
    config = TfidfConfig.from_toml(CONFIG)
    texts, targets = zip(
        *[
            (text, label)
            for text, label in zip(TRAIN_TEXTS, TARGETS, strict=True)
            if label < num_classes
        ],
        strict=True,
    )
    model = fit_tfidf(
        texts, targets, label_ids=tuple(range(num_classes)), config=config
    )
    save_tfidf_model(model, directory / "model.joblib")
    (directory / "config.toml").write_bytes(CONFIG)
    manifest = BundleManifest(
        model_id="synthetic-tfidf-v1",
        family="tfidf_logreg",
        labels=LabelSet(("specific", "historical", "generic")[:num_classes]),
        limits=InputLimits(max_characters=2000, max_batch_size=8),
        environment={
            "python": platform.python_version(),
            **{name: version(name) for name in DEPENDENCIES},
        },
        source_run_id="synthetic-run",
        source_manifest_sha256="a" * 64,
        files={
            path.name: BundleFile(
                hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size
            )
            for path in directory.iterdir()
        },
    )
    write_manifest(directory, manifest.to_dict())
    return directory, model


@pytest.fixture
def tfidf_bundle(tmp_path):
    return make_bundle(tmp_path / "bundle")


def update_hash(directory, name):
    state = json.loads((directory / "manifest.json").read_bytes())
    content = (directory / name).read_bytes()
    state["files"][name] = {
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    write_manifest(directory, state)


def rewrite_model(directory, model):
    joblib.dump(model, directory / "model.joblib", compress=3, protocol=5)
    update_hash(directory, "model.joblib")


def test_matches_pipeline_in_batches_without_truncating_or_changing_learned_state(
    tfidf_bundle, monkeypatch
):
    directory, original = tfidf_bundle
    predictor = Predictor.from_bundle(directory)
    model = predictor._backend._model
    vocabulary = dict(model.named_steps["tfidf"].vocabulary_)
    idf = model.named_steps["tfidf"].idf_.copy()
    coefficients = model.named_steps["logreg"].coef_.copy()
    long_sentence = "We expect " * 80 + "growth"
    texts = (*TEXTS, long_sentence)
    expected = original.predict_proba(prepare_texts(texts))
    predict_proba = model.predict_proba
    batches = []

    def observe(batch):
        batches.append(batch)
        return predict_proba(batch)

    monkeypatch.setattr(model, "predict_proba", observe)
    results = predictor.predict(texts, batch_size=2)
    assert [len(batch) for batch in batches] == [2, 2, 2]
    assert tuple(text for batch in batches for text in batch) == prepare_texts(texts)
    for size in (1, 2, 8):
        for result, scores in zip(
            predictor.predict(texts, batch_size=size), expected, strict=True
        ):
            assert result.probabilities == pytest.approx(scores, rel=1e-12, abs=1e-12)
            assert result.label_id == int(np.argmax(scores))
            assert result.label == predictor.manifest.labels.names[result.label_id]
            assert result.truncated is False
    assert results[0] == results[4]
    assert predictor.predict(["We\u0092ll expect growth"]) == predictor.predict(
        ["We’ll expect growth"]
    )
    assert model.named_steps["tfidf"].vocabulary_ == vocabulary
    np.testing.assert_array_equal(model.named_steps["tfidf"].idf_, idf)
    np.testing.assert_array_equal(model.named_steps["logreg"].coef_, coefficients)
    # Empty feature vectors are valid, and use only the fitted intercepts.
    assert results[2].probabilities == results[3].probabilities


@pytest.mark.parametrize("num_classes", [2, 3])
def test_probability_columns_follow_class_ids_including_binary_models(
    tmp_path, num_classes
):
    directory, model = make_bundle(tmp_path / "bundle", num_classes)
    expected = model.predict_proba(prepare_texts(TEXTS))
    classifier = model.named_steps["logreg"]
    order = np.arange(num_classes)[::-1]
    classifier.classes_ = classifier.classes_[order]
    if num_classes == 2:
        classifier.coef_ *= -1
        classifier.intercept_ *= -1
    else:
        classifier.coef_ = classifier.coef_[order]
        classifier.intercept_ = classifier.intercept_[order]
    rewrite_model(directory, model)
    predictor = Predictor.from_bundle(directory)
    assert predictor._backend._class_columns == tuple(order)
    for result, scores in zip(predictor.predict(TEXTS), expected, strict=True):
        assert result.probabilities == pytest.approx(scores, rel=1e-12, abs=1e-12)
        assert result.label_id == int(np.argmax(scores))
    classifier.coef_.fill(0)
    classifier.intercept_.fill(0)
    rewrite_model(directory, model)
    # A tie is resolved by the lowest bundle class ID, even with reordered columns.
    assert Predictor.from_bundle(directory).predict(["!!!"])[0].label_id == 0


@pytest.mark.parametrize(
    "texts",
    [
        "one sentence",
        [""],
        ["valid", " "],
        [None],
        ["bad\ud800"],
        ["x" * 2001],
        ["growth"] * 9,
    ],
)
def test_invalid_requests_do_not_reach_the_pipeline(tfidf_bundle, monkeypatch, texts):
    predictor = Predictor.from_bundle(tfidf_bundle[0])

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid inputs must be rejected before pipeline execution.")

    monkeypatch.setattr(predictor._backend._model, "predict_proba", unexpected)
    with pytest.raises(PredictionContractError):
        predictor.predict(texts, batch_size=1)
    assert predictor.predict([]) == ()


def test_reuses_in_memory_pipeline_without_files_refitting_or_runtime_changes(
    tfidf_bundle, monkeypatch
):
    directory = tfidf_bundle[0]
    predictor = Predictor.from_bundle(directory)
    expected = predictor.predict(TEXTS)
    model = predictor._backend._model
    rng = np.random.get_state()
    threads = [(row["filepath"], row["num_threads"]) for row in threadpool_info()]
    shutil.rmtree(directory)

    def unexpected(*args, **kwargs):
        pytest.fail("Inference must reuse the loaded pipeline.")

    monkeypatch.setattr(Path, "open", unexpected)
    monkeypatch.setattr(joblib, "load", unexpected)
    monkeypatch.setattr(model, "fit", unexpected)
    for estimator in model.named_steps.values():
        monkeypatch.setattr(estimator, "fit", unexpected)
        if hasattr(estimator, "fit_transform"):
            monkeypatch.setattr(estimator, "fit_transform", unexpected)
    assert predictor.predict(TEXTS) == expected
    assert [
        (row["filepath"], row["num_threads"]) for row in threadpool_info()
    ] == threads
    after = np.random.get_state()
    np.testing.assert_array_equal(rng[1], after[1])
    assert (rng[0], *rng[2:]) == (after[0], *after[2:])


@pytest.mark.parametrize("dependency", ["python", *DEPENDENCIES])
def test_requires_recorded_environment_before_deserialization(
    tfidf_bundle, monkeypatch, dependency
):
    directory = tfidf_bundle[0]
    state = json.loads((directory / "manifest.json").read_bytes())
    state["environment"][dependency] = "99.0.0"
    write_manifest(directory, state)

    def unexpected(*args, **kwargs):
        pytest.fail("Incompatible versions must not deserialize joblib bytes.")

    monkeypatch.setattr(joblib, "load", unexpected)
    with pytest.raises(BundleError, match="requires"):
        Predictor.from_bundle(directory)


@pytest.mark.parametrize(
    "change",
    [
        "config",
        "params",
        "input_mode",
        "transformer",
        "vocabulary",
        "feature_count",
        "classes",
        "float_classes",
        "coef_shape",
        "intercept_shape",
        "idf_shape",
        "nonfinite",
        "dtype",
        "unfitted",
        "wrong_model",
        "corrupt",
    ],
)
def test_payload_validation_rejects_incompatible_state_even_with_valid_hashes(
    tfidf_bundle, change
):
    directory, model = tfidf_bundle
    vectorizer, classifier = model.named_steps["tfidf"], model.named_steps["logreg"]
    if change == "config":
        (directory / "config.toml").write_bytes(CONFIG.replace(b"c = 1.0", b"c = 2.0"))
        update_hash(directory, "config.toml")
    elif change == "corrupt":
        (directory / "model.joblib").write_bytes(b"not joblib")
        update_hash(directory, "model.joblib")
    else:
        if change == "params":
            vectorizer.lowercase = False
        elif change == "input_mode":
            vectorizer.input = "filename"
        elif change == "transformer":
            vectorizer._tfidf.sublinear_tf = False
        elif change == "vocabulary":
            vectorizer.vocabulary_[next(iter(vectorizer.vocabulary_))] = (
                len(vectorizer.vocabulary_) + 1
            )
        elif change == "feature_count":
            classifier.n_features_in_ += 1
        elif change == "classes":
            classifier.classes_ = np.array([0, 0, 2])
        elif change == "float_classes":
            classifier.classes_ = classifier.classes_.astype(float)
        elif change == "coef_shape":
            classifier.coef_ = classifier.coef_[:, :-1]
        elif change == "intercept_shape":
            classifier.intercept_ = classifier.intercept_[:-1]
        elif change == "idf_shape":
            vectorizer._tfidf.idf_ = vectorizer.idf_[:-1]
        elif change == "nonfinite":
            classifier.coef_[0, 0] = np.nan
        elif change == "dtype":
            classifier.coef_ = classifier.coef_.astype(np.float32)
        elif change == "unfitted":
            model = create_tfidf_pipeline(TfidfConfig.from_toml(CONFIG))
        else:
            model = {"not": "a pipeline"}
        rewrite_model(directory, model)
    with pytest.raises(BundleError):
        Predictor.from_bundle(directory)


def test_sklearn_embedded_version_warning_is_an_error(tfidf_bundle, monkeypatch):
    def wrong_version(*args, **kwargs):
        warnings.warn(
            InconsistentVersionWarning(
                estimator_name="LogisticRegression",
                current_sklearn_version=version("scikit-learn"),
                original_sklearn_version="0.0.0",
            ),
            stacklevel=2,
        )

    monkeypatch.setattr(joblib, "load", wrong_version)
    with pytest.raises(BundleError, match="training scikit-learn version"):
        Predictor.from_bundle(tfidf_bundle[0])


def test_modification_after_verification_is_rejected_before_joblib(
    tfidf_bundle, monkeypatch
):
    directory = tfidf_bundle[0]
    verify = public.verify_bundle

    def mutate(*args, **kwargs):
        manifest = verify(*args, **kwargs)
        path = directory / "model.joblib"
        path.write_bytes(b"x" * path.stat().st_size)
        return manifest

    def unexpected(*args, **kwargs):
        pytest.fail("Modified bytes must not be deserialized.")

    monkeypatch.setattr(public, "verify_bundle", mutate)
    monkeypatch.setattr(joblib, "load", unexpected)
    with pytest.raises(BundleError, match="checksum"):
        Predictor.from_bundle(directory)


def test_joblib_loads_only_once_from_the_verified_snapshot(tfidf_bundle, monkeypatch):
    directory, model = tfidf_bundle
    expected = model.predict_proba(prepare_texts(TEXTS))
    read, load = backend.read_bundle_file, joblib.load
    calls = []

    def mutate_after_read(directory, name, **kwargs):
        content = read(directory, name, **kwargs)
        if name == "model.joblib":
            (directory / name).write_bytes(b"changed after read")
        return content

    def inspect(stream):
        assert isinstance(stream, BytesIO)
        calls.append(1)
        return load(stream)

    monkeypatch.setattr(backend, "read_bundle_file", mutate_after_read)
    monkeypatch.setattr(joblib, "load", inspect)
    predictor = Predictor.from_bundle(directory)
    for _ in range(2):
        for result, scores in zip(predictor.predict(TEXTS), expected, strict=True):
            assert result.probabilities == pytest.approx(scores, rel=1e-12, abs=1e-12)
    assert calls == [1]


@pytest.mark.parametrize(
    "scores", [np.zeros((1, 3)), np.ones((1, 2)), np.array([[np.nan, 0, 1]])]
)
def test_invalid_pipeline_outputs_fail_the_shared_contract(
    tfidf_bundle, monkeypatch, scores
):
    predictor = Predictor.from_bundle(tfidf_bundle[0])
    monkeypatch.setattr(predictor._backend._model, "predict_proba", lambda _: scores)
    with pytest.raises(PredictionContractError):
        predictor.predict(["growth"])


def test_fresh_process_uses_moved_bundle_offline_without_neural_or_data_tools(
    tfidf_bundle, tmp_path
):
    directory = tfidf_bundle[0]
    expected = [
        result.to_dict() for result in Predictor.from_bundle(directory).predict(TEXTS)
    ]
    moved = tmp_path / "delivery"
    directory.rename(moved)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    script = """import builtins, json, sys
from pathlib import Path
original_import = builtins.__import__
forbidden = {'torch', 'mlflow', 'matplotlib', 'pandas', 'pyarrow', 'huggingface_hub'}
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in forbidden:
        raise ModuleNotFoundError('Optional package unavailable', name=name)
    assert not name.startswith('filing_sentence_classifier.text'), name
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
def offline(event, args):
    assert event != 'socket.connect', 'Inference must be offline.'
sys.addaudithook(offline)
from filing_sentence_classifier.inference.predictor import Predictor
predictor = Predictor.from_bundle(Path(sys.argv[1]))
print(json.dumps([r.to_dict() for r in predictor.predict(json.loads(sys.argv[2]))]))
assert not forbidden & sys.modules.keys()
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(moved), json.dumps(TEXTS)],
        cwd=elsewhere,
        check=False,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stderr
    for actual, reference in zip(json.loads(child.stdout), expected, strict=True):
        assert actual.pop("probabilities") == pytest.approx(
            reference.pop("probabilities"), rel=1e-12, abs=1e-12
        )
        assert actual == reference


def test_missing_sklearn_is_actionable_and_public_import_stays_lightweight(
    tfidf_bundle, tmp_path
):
    package_parent = Path(filing_sentence_classifier.__file__).resolve().parent.parent
    script = """import builtins, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from filing_sentence_classifier.inference.predictor import Predictor
from filing_sentence_classifier.artifacts import BundleError
assert not {'torch', 'numpy', 'sklearn', 'joblib', 'mlflow'} & sys.modules.keys()
original_import = builtins.__import__
def missing(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'scipy', 'sklearn', 'joblib', 'threadpoolctl'}:
        raise ModuleNotFoundError("Missing optional runtime", name=name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = missing
try:
    Predictor.from_bundle(Path(sys.argv[2]))
except BundleError as error:
    assert '[data] extra' in str(error), error
else:
    raise AssertionError('Expected a dependency error.')
"""
    subprocess.run(
        [sys.executable, "-S", "-c", script, str(package_parent), str(tfidf_bundle[0])],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
