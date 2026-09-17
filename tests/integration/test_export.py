"""Export synthetic saved models, preserve their predictions, and protect outputs."""

import hashlib
import json
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

import filing_sentence_classifier
from filing_sentence_classifier import exporting
from filing_sentence_classifier.artifacts import BundleError, verify_bundle
from filing_sentence_classifier.baselines.tfidf import (
    TfidfConfig,
    fit_tfidf,
    load_tfidf_model,
    save_tfidf_model,
)
from filing_sentence_classifier.cli import app
from filing_sentence_classifier.data.cleaning import cleaning_recipe
from filing_sentence_classifier.exporting import export_bundle
from filing_sentence_classifier.inference.contracts import InputLimits
from filing_sentence_classifier.models.mean_pool_mlp import MeanPoolMLP
from filing_sentence_classifier.text.encoding import TextEncoder
from filing_sentence_classifier.text.tokenization import tokenize
from filing_sentence_classifier.text.vocabulary import Vocabulary
from filing_sentence_classifier.training.artifacts import file_inventory, json_bytes
from filing_sentence_classifier.training.checkpoints import (
    load_checkpoint,
    save_checkpoint,
)
from filing_sentence_classifier.training.config import TrainingConfig

TEXTS = (
    "We expect growth",
    "We expect investment",
    "Profit increased yesterday",
    "Revenue increased yesterday",
    "Results may differ",
    "Outcomes may differ",
)
TARGETS = (0, 0, 1, 1, 2, 2)
NEURAL_CONFIG = b"""schema_version = 2
[model]
embedding_dim = 4
hidden_dim = 3
dropout = 0.2
[training]
batch_size = 2
max_epochs = 3
patience = 2
min_delta = 0.0
learning_rate = 0.001
weight_decay = 0.0
[runtime]
seed = 17
device = "cpu"
num_workers = 0
num_threads = 1
"""
TFIDF_CONFIG = b"""[tfidf]
ngram_range = [1, 2]
min_df = 1
[logistic_regression]
c = 1.0
max_iter = 1000
tol = 0.0001
"""


def payloads(directory):
    return {
        p.relative_to(directory).as_posix(): p.read_bytes()
        for p in directory.rglob("*")
        if p.is_file()
    }


def neural_model(encoder, config):
    return MeanPoolMLP(
        len(encoder.vocabulary),
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
    )


def neural_probabilities(model, encoder):
    model.eval()
    with torch.inference_mode():
        return [
            model(
                ids := torch.tensor([encoder.encode(text).input_ids]),
                torch.ones_like(ids, dtype=torch.bool),
            )
            .softmax(dim=1)
            .tolist()[0]
            for text in TEXTS
        ]


@pytest.fixture(params=["mean_pool_mlp", "tfidf_logreg"])
def saved_run(tmp_path, request):
    """Use real serializers with small synthetic state, never the project corpus."""
    family = request.param
    run = tmp_path / "synthetic-run"
    run.mkdir()
    labels = {"0": "specific", "1": "historical", "2": "generic"}
    data_manifest = tmp_path / "data-manifest.json"
    data_manifest.write_bytes(
        json_bytes(
            {
                "schema_version": 1,
                "stage": "development_splits",
                "label_names": labels,
                "cleaning_recipe": cleaning_recipe(),
            }
        )
    )
    state = {
        "schema_version": 1,
        "data": {
            "label_names": labels,
            "manifest_sha256": hashlib.sha256(data_manifest.read_bytes()).hexdigest(),
        },
    }
    if family == "mean_pool_mlp":
        encoder = TextEncoder(Vocabulary.fit(map(tokenize, TEXTS), min_frequency=1), 3)
        config = TrainingConfig.from_toml(NEURAL_CONFIG)
        model = neural_model(encoder, config)
        save_checkpoint(
            run / "checkpoints/best.pt", model, config=config, epoch=1, macro_f1=0.5
        )
        (run / "config.toml").write_bytes(NEURAL_CONFIG)
        (run / "encoder.json").write_bytes(json_bytes(encoder.to_dict()))
        shutil.copyfile(data_manifest, run / "data-manifest.json")
        state.update(
            {
                "stage": "neural_training_run",
                "status": "completed",
                "run_id": run.name,
                "recipe": {
                    "family": family,
                    "fit_split": "train",
                    "evaluation_split": "val",
                    "selection_metric": "macro_f1",
                    "checkpoint_tie_break": "earliest_epoch",
                },
                "config": config.to_dict(),
                "model": {"num_classes": 3},
                "preprocessing": {
                    "vocabulary_size": len(encoder.vocabulary),
                    "max_length": 3,
                },
                "environment": {
                    "python": "3.13.9",
                    "packages": {
                        name: version(name)
                        for name in ("filing-sentence-classifier", "torch")
                    },
                },
            }
        )
        probabilities = neural_probabilities(model, encoder)
    else:
        config = TfidfConfig.from_toml(TFIDF_CONFIG)
        model = fit_tfidf(TEXTS, TARGETS, label_ids=(0, 1, 2), config=config)
        save_tfidf_model(model, run / "model.joblib")
        (run / "config.toml").write_bytes(TFIDF_CONFIG)
        (run / "model.json").write_bytes(
            json_bytes(
                {
                    "schema_version": 1,
                    "family": family,
                    "serialization": "joblib",
                    "model_file": "model.joblib",
                    "class_ids": [0, 1, 2],
                    "converged": True,
                    "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
                }
            )
        )
        state.update(
            {
                "stage": "baseline_run",
                "recipe": config.recipe(),
                "environment": {
                    "python": "3.13.9",
                    **{
                        name: version(name)
                        for name in (
                            "filing-sentence-classifier",
                            "scikit-learn",
                            "joblib",
                            "numpy",
                            "scipy",
                            "threadpoolctl",
                        )
                    },
                },
            }
        )
        probabilities = model.predict_proba(TEXTS).tolist()
    state["files"] = file_inventory(run)
    (run / "manifest.json").write_bytes(json_bytes(state))
    return run, data_manifest, family, probabilities


def export_args(saved_run, output):
    run, data, family, _ = saved_run
    return dict(
        run_dir=run,
        output_dir=output,
        model_id="delivery-v1",
        data_manifest=data if family == "tfidf_logreg" else None,
    )


def test_export_preserves_bytes_environment_and_predictions_after_source_removal(
    saved_run,
    tmp_path,
):
    run, data, family, expected_probabilities = saved_run
    before = payloads(run)
    source_manifest = json.loads(before["manifest.json"])
    output = export_bundle(**export_args(saved_run, tmp_path / "bundle"))
    assert payloads(run) == before
    bundle = verify_bundle(output)
    assert (
        bundle.source_manifest_sha256
        == hashlib.sha256(before["manifest.json"]).hexdigest()
    )
    assert bundle.source_run_id == run.name
    assert bundle.environment["python"] == source_manifest["environment"]["python"]
    for name in bundle.files:
        original_name = "checkpoints/best.pt" if name == "model.pt" else name
        assert (output / name).read_bytes() == before[original_name]
    shutil.rmtree(run)
    data.unlink()
    moved = tmp_path / "moved-bundle"
    output.rename(moved)
    assert verify_bundle(moved) == bundle
    if family == "mean_pool_mlp":
        encoder = TextEncoder.from_dict(
            json.loads((moved / "encoder.json").read_bytes())
        )
        config = TrainingConfig.from_toml((moved / "config.toml").read_bytes())
        model = neural_model(encoder, config)
        load_checkpoint(moved / "model.pt", model, expected_config=config)
        actual = neural_probabilities(model, encoder)
        from filing_sentence_classifier.inference.predictor import Predictor

        predictions = Predictor.from_bundle(moved).predict(TEXTS, batch_size=1)
        assert [list(result.probabilities) for result in predictions] == actual
    else:
        model = load_tfidf_model(
            moved / "model.joblib", expected_sha256=bundle.files["model.joblib"].sha256
        )
        actual = model.predict_proba(TEXTS).tolist()
    assert actual == expected_probabilities


def test_cli_repeats_identically_and_does_not_rewrite_files(saved_run, tmp_path):
    run, data, family, _ = saved_run
    output = tmp_path / "bundle"
    args = [
        "export",
        "--run-dir",
        str(run),
        "--output-dir",
        str(output),
        "--model-id",
        "delivery-v1",
        "--max-characters",
        "500",
        "--max-batch-size",
        "8",
        "--manifest-sha256",
        hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest(),
    ]
    if family == "tfidf_logreg":
        args += ["--data-manifest", str(data)]
    runner = CliRunner()
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Verified inference bundle" in result.output
    before = payloads(output)
    mtimes = {p.name: p.stat().st_mtime_ns for p in output.iterdir()}
    assert verify_bundle(output).limits == InputLimits(500, 8)
    assert runner.invoke(app, args).exit_code == 0
    assert payloads(output) == before
    assert {p.name: p.stat().st_mtime_ns for p in output.iterdir()} == mtimes
    other = export_bundle(
        **export_args(saved_run, tmp_path / "other"), limits=InputLimits(500, 8)
    )
    assert payloads(other) == before


def test_export_works_in_a_fresh_process_without_site_packages_or_dataset_rows(
    saved_run, tmp_path
):
    run, data, family, _ = saved_run
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    package_parent = Path(filing_sentence_classifier.__file__).resolve().parent.parent
    script = """import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from filing_sentence_classifier.exporting import export_bundle
from filing_sentence_classifier.artifacts import verify_bundle
result = export_bundle(Path(sys.argv[2]), Path(sys.argv[3]), model_id='isolated-v1', data_manifest=Path(sys.argv[4]))
assert verify_bundle(result).model_id == 'isolated-v1'
assert not {'torch', 'sklearn', 'joblib', 'numpy', 'scipy', 'mlflow', 'pyarrow', 'huggingface_hub'} & sys.modules.keys()
"""
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            script,
            str(package_parent),
            str(run),
            str(tmp_path / "isolated"),
            str(data),
        ],
        cwd=elsewhere,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "change", ["corrupt", "missing", "extra", "limits", "model_id", "file", "symlink"]
)
def test_existing_destinations_are_never_replaced(saved_run, tmp_path, change):
    output = export_bundle(**export_args(saved_run, tmp_path / "bundle"))
    kwargs = export_args(saved_run, output)
    if change == "corrupt":
        (output / "config.toml").write_bytes(b"corrupted")
    elif change == "missing":
        (output / "manifest.json").unlink()
    elif change == "extra":
        (output / "notes.txt").write_text("keep")
    elif change in ("file", "symlink"):
        moved = tmp_path / "original"
        output.rename(moved)
        if change == "file":
            output.write_bytes(b"keep")
        else:
            output.symlink_to(moved, target_is_directory=True)
    elif change == "limits":
        kwargs["limits"] = InputLimits(max_batch_size=1)
    else:
        kwargs["model_id"] = "changed-v2"
    before = payloads(output) if output.is_dir() else output.read_bytes()
    with pytest.raises(BundleError):
        export_bundle(**kwargs)
    assert (payloads(output) if output.is_dir() else output.read_bytes()) == before
    assert not list(tmp_path.glob(".bundle.export*"))


@pytest.mark.parametrize(
    "change",
    [
        "source_pin",
        "bytes",
        "data_hash",
        "labels",
        "cleaning",
        "config",
        "version",
        "family",
        "symlink",
    ],
)
def test_invalid_source_or_metadata_leaves_no_bundle(saved_run, tmp_path, change):
    run, data, family, _ = saved_run
    manifest_path = run / "manifest.json"
    state = json.loads(manifest_path.read_bytes())
    kwargs = export_args(saved_run, tmp_path / "bundle")
    if change == "source_pin":
        kwargs["expected_manifest_sha256"] = "0" * 64
    elif change == "bytes":
        path = run / ("encoder.json" if family == "mean_pool_mlp" else "model.joblib")
        content = path.read_bytes()
        path.write_bytes(b"x" * len(content))
    elif change == "data_hash":
        state["data"]["manifest_sha256"] = "0" * 64
    elif change == "labels":
        state["data"]["label_names"]["0"] = "changed"
    elif change == "cleaning":
        metadata = json.loads(data.read_bytes())
        metadata["cleaning_recipe"]["version"] = "2"
        data.write_bytes(json_bytes(metadata))
        state["data"]["manifest_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
        kwargs["data_manifest"] = data
    elif change == "config":
        path = run / "config.toml"
        path.write_bytes(
            path.read_bytes()
            .replace(b"0.001", b"0.002")
            .replace(b"c = 1.0", b"c = 2.0")
        )
        state["files"]["config.toml"] = file_inventory(run)["config.toml"]
    elif change == "version":
        state["schema_version"] = True
    elif change == "family":
        state["recipe"]["family"] = "majority"
    else:
        path = run / "config.toml"
        path.rename(tmp_path / "external-config")
        path.symlink_to(tmp_path / "external-config")
    manifest_path.write_bytes(json_bytes(state))
    with pytest.raises(BundleError):
        export_bundle(**kwargs)
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle.export*"))


def test_family_metadata_and_missing_data_manifest_are_checked(saved_run, tmp_path):
    run, _, family, _ = saved_run
    kwargs = export_args(saved_run, tmp_path / "bundle")
    if family == "mean_pool_mlp":
        path = run / "manifest.json"
        state = json.loads(path.read_bytes())
        state["status"] = "failed"
        path.write_bytes(json_bytes(state))
        with pytest.raises(BundleError, match="completed"):
            export_bundle(**kwargs)
        state["status"] = "completed"
        state["preprocessing"]["vocabulary_size"] += 1
        path.write_bytes(json_bytes(state))
    else:
        with pytest.raises(BundleError, match="--data-manifest"):
            export_bundle(**(kwargs | {"data_manifest": None}))
        model_path = run / "model.json"
        model = json.loads(model_path.read_bytes())
        model["class_ids"] = [2, 0, 1]
        model_path.write_bytes(json_bytes(model))
        path = run / "manifest.json"
        state = json.loads(path.read_bytes())
        state["files"]["model.json"] = file_inventory(run)["model.json"]
        path.write_bytes(json_bytes(state))
    with pytest.raises(BundleError):
        export_bundle(**kwargs)
    assert not (tmp_path / "bundle").exists()


def test_write_failure_cleans_staging_and_lock(saved_run, tmp_path, monkeypatch):
    original = exporting._copy_payload

    def fail_after_copy(source, target, expected):
        original(source, target, expected)
        raise OSError("injected write failure")

    monkeypatch.setattr(exporting, "_copy_payload", fail_after_copy)
    with pytest.raises(BundleError, match="injected write failure"):
        export_bundle(**export_args(saved_run, tmp_path / "bundle"))
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle.export*"))


def test_overlapping_paths_and_concurrent_export_are_rejected(saved_run, tmp_path):
    run, _, _, _ = saved_run
    for output in (run, run / "bundle", tmp_path):
        with pytest.raises(BundleError, match="overlap"):
            export_bundle(**export_args(saved_run, output))
    lock = tmp_path / ".bundle.export.lock"
    lock.write_text("another export")
    with pytest.raises(BundleError, match="lock"):
        export_bundle(**export_args(saved_run, tmp_path / "bundle"))
    assert lock.read_text() == "another export"
    assert not (tmp_path / "bundle").exists()
