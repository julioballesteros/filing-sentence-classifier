"""Restore portable neural bundles and exercise the public prediction boundary."""

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from importlib.metadata import version
from io import BytesIO
from pathlib import Path

import pytest
import torch

import filing_sentence_classifier
from filing_sentence_classifier.artifacts import BundleError, BundleFile, BundleManifest
from filing_sentence_classifier.inference import predictor as public
from filing_sentence_classifier.inference import pytorch as backend
from filing_sentence_classifier.inference.contracts import (
    InputLimits,
    LabelSet,
    PredictionContractError,
    prepare_texts,
)
from filing_sentence_classifier.inference.predictor import Predictor
from filing_sentence_classifier.models.mean_pool_mlp import MeanPoolMLP
from filing_sentence_classifier.text.encoding import TextEncoder
from filing_sentence_classifier.text.tokenization import tokenize
from filing_sentence_classifier.text.vocabulary import Vocabulary
from filing_sentence_classifier.training.checkpoints import save_checkpoint
from filing_sentence_classifier.training.config import ModelConfig, TrainingConfig

TEXTS = (
    "growth.",
    "  We expect growth.\n",
    "unseenword",
    "We expect growth next year.",
    "We expect growth.",
)


def write_config(path, config):
    lines = ["schema_version = 2"]
    for section, values in config.to_dict().items():
        if isinstance(values, dict):
            lines += [
                f"[{section}]",
                *(f"{key} = {json.dumps(value)}" for key, value in values.items()),
            ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(directory, state):
    (directory / "manifest.json").write_text(json.dumps(state), encoding="utf-8")


def update_payload_hash(directory, name):
    state = json.loads((directory / "manifest.json").read_bytes())
    content = (directory / name).read_bytes()
    state["files"][name] = {
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }
    write_manifest(directory, state)


@pytest.fixture
def neural_bundle(tmp_path):
    """A small serialized model with dropout and an intentionally short encoder."""
    directory = tmp_path / "bundle"
    directory.mkdir()
    config = TrainingConfig(model=ModelConfig(8, 5, 0.7))
    encoder = TextEncoder(
        Vocabulary.fit(
            map(tokenize, ("We expect growth.", "Historical revenue grew.")),
            min_frequency=1,
        ),
        max_length=4,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = MeanPoolMLP(
            len(encoder.vocabulary), embedding_dim=8, hidden_dim=5, dropout=0.7
        )
    model.eval()
    write_config(directory / "config.toml", config)
    (directory / "encoder.json").write_text(
        json.dumps(encoder.to_dict()), encoding="utf-8"
    )
    save_checkpoint(directory / "model.pt", model, config=config, epoch=1, macro_f1=0.5)
    manifest = BundleManifest(
        model_id="synthetic-neural-v1",
        family="mean_pool_mlp",
        labels=LabelSet(("specific", "historical", "generic")),
        limits=InputLimits(max_characters=2000, max_batch_size=8),
        environment={
            "python": platform.python_version(),
            **{name: version(name) for name in ("torch", "filing-sentence-classifier")},
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
    return directory, model, encoder, config


def reference_probabilities(model, encoder, texts):
    with torch.inference_mode():
        return [
            model(
                ids := torch.tensor([encoder.encode(text).input_ids], dtype=torch.long),
                ids != 0,
            )
            .softmax(dim=1)
            .tolist()[0]
            for text in prepare_texts(texts)
        ]


def test_batching_matches_reference_with_padding_unknowns_and_truncation(neural_bundle):
    directory, model, encoder, _ = neural_bundle
    predictor = Predictor.from_bundle(directory)
    expected = reference_probabilities(model, encoder, TEXTS)
    observed_batches = []

    def inspect(module, inputs):
        ids, mask = inputs
        assert not module.training
        assert not torch.is_grad_enabled() and torch.is_inference_mode_enabled()
        assert ids.dtype == torch.long and ids.device.type == "cpu"
        assert mask.dtype == torch.bool and torch.equal(mask, ids != 0)
        observed_batches.append(ids.clone())

    hook = predictor._backend._model.register_forward_pre_hook(inspect)
    results = predictor.predict(TEXTS, batch_size=2)
    hook.remove()
    assert [list(ids.shape) for ids in observed_batches] == [[2, 4], [2, 4], [1, 4]]
    assert observed_batches[1][0].tolist() == [1, 0, 0, 0]
    assert [result.truncated for result in results] == [
        False,
        False,
        False,
        True,
        False,
    ]
    for size in (1, 2, 8):
        repeated = predictor.predict(TEXTS, batch_size=size)
        for result, scores in zip(repeated, expected, strict=True):
            assert result.probabilities == pytest.approx(scores, rel=1e-6, abs=1e-7)
            assert result.label_id == scores.index(max(scores))
            assert result.label == predictor.manifest.labels.names[result.label_id]
    assert predictor.predict(TEXTS, batch_size=2) == results
    assert results[1].label_id == results[4].label_id
    assert results[1].probabilities == pytest.approx(
        results[4].probabilities, rel=1e-6, abs=1e-7
    )
    # The source normalizer and saved tokenizer share the training path.
    assert predictor.predict(["We\u0092ll expect growth."]) == predictor.predict(
        ["We’ll expect growth."]
    )
    assert predictor._backend._encoder.to_dict() == encoder.to_dict()


@pytest.mark.parametrize(
    "texts",
    [
        "one sentence",
        [""],
        ["growth.", " "],
        [None],
        ["bad\ud800"],
        ["x" * 2001],
        ["growth."] * 9,
    ],
)
def test_invalid_request_is_rejected_before_any_model_execution(
    neural_bundle, monkeypatch, texts
):
    predictor = Predictor.from_bundle(neural_bundle[0])

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid requests must not execute the model.")

    monkeypatch.setattr(predictor._backend._model, "forward", unexpected)
    with pytest.raises(PredictionContractError):
        predictor.predict(texts, batch_size=1)
    assert predictor.predict([]) == ()


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_computation_batch_size_is_rejected(neural_bundle, batch_size):
    with pytest.raises(PredictionContractError, match="batch_size"):
        Predictor.from_bundle(neural_bundle[0]).predict(TEXTS, batch_size=batch_size)


def test_loaded_predictor_needs_no_files_and_never_refits_or_reloads(
    neural_bundle, monkeypatch
):
    directory = neural_bundle[0]
    predictor = Predictor.from_bundle(directory)
    expected = predictor.predict(TEXTS)
    shutil.rmtree(directory)

    def unexpected(*args, **kwargs):
        pytest.fail("Prediction must reuse in-memory state.")

    monkeypatch.setattr(Path, "open", unexpected)
    monkeypatch.setattr(torch, "load", unexpected)
    monkeypatch.setattr(Vocabulary, "fit", unexpected)
    monkeypatch.setattr(TextEncoder, "from_dict", unexpected)
    assert predictor.predict(TEXTS) == expected


def test_loading_and_prediction_preserve_rng_and_runtime_settings(neural_bundle):
    directory = neural_bundle[0]
    expected = Predictor.from_bundle(directory).predict(TEXTS)
    before_rng, threads, dtype = (
        torch.get_rng_state().clone(),
        torch.get_num_threads(),
        torch.get_default_dtype(),
    )
    try:
        torch.set_default_dtype(torch.float64)
        predictor = Predictor.from_bundle(directory)
        model = predictor._backend._model
        assert all(
            p.dtype == torch.float32 and p.device.type == "cpu"
            for p in model.parameters()
        )
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            assert predictor.predict(TEXTS) == expected
        assert torch.get_default_dtype() == torch.float64
        assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
        assert torch.equal(before_rng, torch.get_rng_state())
        assert torch.get_num_threads() == threads
    finally:
        torch.set_default_dtype(dtype)


@pytest.mark.parametrize(
    "dependency", ["python", "torch", "filing-sentence-classifier"]
)
def test_incompatible_environment_fails_before_deserialization(
    neural_bundle, monkeypatch, dependency
):
    directory = neural_bundle[0]
    state = json.loads((directory / "manifest.json").read_bytes())
    state["environment"][dependency] = "99.0.0"
    write_manifest(directory, state)

    def unexpected(*args, **kwargs):
        pytest.fail("Incompatible runtimes must fail before deserializing weights.")

    monkeypatch.setattr(torch, "load", unexpected)
    with pytest.raises(BundleError, match="requires"):
        Predictor.from_bundle(directory)


def test_python_patch_and_cpu_wheel_suffix_are_compatible(neural_bundle):
    directory = neural_bundle[0]
    state = json.loads((directory / "manifest.json").read_bytes())
    state["environment"]["python"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}.0"
    )
    state["environment"]["torch"] = version("torch").split("+", 1)[0] + "+cpu"
    state["environment"]["matplotlib"] = "not-installed-for-inference"
    write_manifest(directory, state)
    assert len(Predictor.from_bundle(directory).predict(TEXTS)) == len(TEXTS)


@pytest.mark.parametrize(
    "change",
    [
        "config",
        "class_count",
        "vocabulary_size",
        "tokenizer",
        "checkpoint_version",
        "weights_shape",
        "weights_dtype",
        "nonfinite",
        "invalid_binary",
    ],
)
def test_consistent_hashes_do_not_hide_incompatible_payloads(neural_bundle, change):
    directory, _, _, config = neural_bundle
    state = json.loads((directory / "manifest.json").read_bytes())
    if change == "config":
        write_config(
            directory / "config.toml", TrainingConfig(model=ModelConfig(8, 5, 0.1))
        )
        update_payload_hash(directory, "config.toml")
    elif change == "class_count":
        state["labels"].pop("2")
        write_manifest(directory, state)
    elif change in ("vocabulary_size", "tokenizer"):
        encoder_state = json.loads((directory / "encoder.json").read_bytes())
        if change == "vocabulary_size":
            encoder_state["vocabulary"] = Vocabulary.fit(
                [("new", "tokens")], min_frequency=1
            ).to_dict()
        else:
            encoder_state["tokenization"]["version"] = "unsupported"
        (directory / "encoder.json").write_text(
            json.dumps(encoder_state), encoding="utf-8"
        )
        update_payload_hash(directory, "encoder.json")
    else:
        path = directory / "model.pt"
        if change == "invalid_binary":
            path.write_bytes(b"not a checkpoint")
        else:
            checkpoint = torch.load(path, weights_only=True, map_location="cpu")
            weights = checkpoint["model_state_dict"]
            if change == "checkpoint_version":
                checkpoint["schema_version"] = 2
            elif change == "weights_shape":
                weights["embedding.weight"] = weights["embedding.weight"][:-1]
            elif change == "weights_dtype":
                weights["embedding.weight"] = weights["embedding.weight"].double()
            else:
                weights["embedding.weight"][1, 0] = float("nan")
            torch.save(checkpoint, path)
        update_payload_hash(directory, "model.pt")
    with pytest.raises(BundleError):
        Predictor.from_bundle(directory)


def test_changed_payload_between_verification_and_loading_is_rejected(
    neural_bundle, monkeypatch
):
    directory = neural_bundle[0]
    verify = public.verify_bundle

    def mutate_after_verification(*args, **kwargs):
        manifest = verify(*args, **kwargs)
        path = directory / "model.pt"
        path.write_bytes(b"x" * path.stat().st_size)
        return manifest

    def unexpected(*args, **kwargs):
        pytest.fail("Modified bytes must not reach the deserializer.")

    monkeypatch.setattr(public, "verify_bundle", mutate_after_verification)
    monkeypatch.setattr(torch, "load", unexpected)
    with pytest.raises(BundleError, match="checksum"):
        Predictor.from_bundle(directory)


def test_deserializer_consumes_the_verified_snapshot_even_if_file_changes(
    neural_bundle, monkeypatch
):
    directory, model, encoder, _ = neural_bundle
    expected = reference_probabilities(model, encoder, TEXTS)
    read, load = backend.read_bundle_file, torch.load
    calls = []

    def mutate_after_read(directory, name, **kwargs):
        content = read(directory, name, **kwargs)
        if name == "model.pt":
            (directory / name).write_bytes(b"changed after reading")
        return content

    def checked_load(path, **kwargs):
        assert isinstance(path, BytesIO)
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        calls.append(1)
        return load(path, **kwargs)

    monkeypatch.setattr(backend, "read_bundle_file", mutate_after_read)
    monkeypatch.setattr(torch, "load", checked_load)
    predictor = Predictor.from_bundle(directory)
    for _ in range(2):
        for result, scores in zip(
            predictor.predict(TEXTS, batch_size=1), expected, strict=True
        ):
            assert result.probabilities == pytest.approx(scores, rel=1e-6, abs=1e-7)
    assert calls == [1]


def test_manifest_pin_rejects_changed_labels(neural_bundle):
    directory = neural_bundle[0]
    path = directory / "manifest.json"
    pin = hashlib.sha256(path.read_bytes()).hexdigest()
    assert Predictor.from_bundle(directory, expected_manifest_sha256=pin)
    state = json.loads(path.read_bytes())
    state["labels"]["0"] = "changed"
    write_manifest(directory, state)
    with pytest.raises(BundleError, match="manifest checksum"):
        Predictor.from_bundle(directory, expected_manifest_sha256=pin)


def test_fresh_process_predicts_offline_outside_repo_with_only_the_moved_bundle(
    neural_bundle, tmp_path
):
    directory = neural_bundle[0]
    expected = Predictor.from_bundle(directory).predict(TEXTS, batch_size=2)
    moved = tmp_path / "delivery"
    directory.rename(moved)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    script = """import builtins, json, sys
from pathlib import Path
original_import = builtins.__import__
forbidden = {'sklearn', 'joblib', 'mlflow', 'matplotlib', 'pyarrow', 'huggingface_hub'}
def guarded_import(name, *args, **kwargs):
    assert name.split('.')[0] not in forbidden, name
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
def offline(event, args):
    assert event != 'socket.connect', 'Inference must be offline.'
sys.addaudithook(offline)
from filing_sentence_classifier.inference.predictor import Predictor
predictor = Predictor.from_bundle(Path(sys.argv[1]))
print(json.dumps([r.to_dict() for r in predictor.predict(json.loads(sys.argv[2]), batch_size=2)]))
assert not forbidden & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(moved), json.dumps(TEXTS)],
        cwd=elsewhere,
        capture_output=True,
        text=True,
        check=True,
    )
    actual = json.loads(result.stdout)
    for result, reference in zip(actual, expected, strict=True):
        scores = result.pop("probabilities")
        reference_state = reference.to_dict()
        assert scores == pytest.approx(
            reference_state.pop("probabilities"), rel=1e-6, abs=1e-7
        )
        assert result == reference_state


def test_public_import_is_lightweight_and_missing_torch_is_actionable(
    neural_bundle, tmp_path
):
    package_parent = Path(filing_sentence_classifier.__file__).resolve().parent.parent
    script = """import builtins, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from filing_sentence_classifier.inference.predictor import Predictor
from filing_sentence_classifier.artifacts import BundleError
assert not {'torch', 'sklearn', 'mlflow', 'numpy'} & sys.modules.keys()
original_import = builtins.__import__
def no_torch(name, *args, **kwargs):
    if name == 'torch':
        raise ModuleNotFoundError("No module named 'torch'", name='torch')
    return original_import(name, *args, **kwargs)
builtins.__import__ = no_torch
try:
    Predictor.from_bundle(Path(sys.argv[2]))
except BundleError as error:
    assert '[train] extra' in str(error), error
else:
    raise AssertionError('Expected a dependency error.')
"""
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            script,
            str(package_parent),
            str(neural_bundle[0]),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
