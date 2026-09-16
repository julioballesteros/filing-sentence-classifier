"""Exercise the complete training command with small offline artifacts."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from filing_sentence_classifier.cli import app
from filing_sentence_classifier.data.dataloader import create_dataloader
from filing_sentence_classifier.data.dataset import SentenceDataset
from filing_sentence_classifier.data.loading import load_split
from filing_sentence_classifier.data.vocabulary import build_vocabulary
from filing_sentence_classifier.models.mean_pool_mlp import MeanPoolMLP
from filing_sentence_classifier.text.encoding import TextEncoder
from filing_sentence_classifier.training.checkpoints import load_checkpoint
from filing_sentence_classifier.training.config import TrainingConfig
from filing_sentence_classifier.training.engine import validate_epoch


@pytest.fixture
def inputs(development_artifact: Path, tmp_path: Path):
    vocabulary = build_vocabulary(
        development_artifact, tmp_path / "vocabulary", min_frequency=1
    )
    config = tmp_path / "config.toml"
    config.write_text(
        """schema_version = 2
[model]
embedding_dim = 8
hidden_dim = 4
dropout = 0.2
[training]
batch_size = 13
max_epochs = 3
patience = 2
min_delta = 0.0
learning_rate = 0.03
weight_decay = 0.0
[runtime]
seed = 17
device = "cpu"
num_workers = 0
num_threads = 1
""",
        encoding="utf-8",
    )
    return development_artifact, vocabulary, config


def invoke(
    inputs,
    output: Path,
    hash_seed: str = "1",
    *,
    fail: bool = False,
    tracking_dir: Path | None = None,
):
    data, vocabulary, config = inputs
    # Forbid preprocessing fitting and published-source/test reads at the boundary.
    script = """import sys
from pathlib import Path
from filing_sentence_classifier.text.vocabulary import Vocabulary
def forbidden(*args, **kwargs):
    raise AssertionError("Training must reuse the saved vocabulary.")
Vocabulary.fit = staticmethod(forbidden)
read = Path.read_bytes
def development_only(path):
    assert path.suffix != ".parquet" and "raw" not in path.parts
    return read(path)
Path.read_bytes = development_only
"""
    if tracking_dir is None:
        script += """import builtins
original_import = builtins.__import__
def no_mlflow(name, *args, **kwargs):
    assert name.split('.')[0] != 'mlflow', 'Untracked training must not import MLflow.'
    return original_import(name, *args, **kwargs)
builtins.__import__ = no_mlflow
"""
    if fail:
        script += """from filing_sentence_classifier.training.run import TrainingRunError, run_training
def abort(row):
    raise RuntimeError("injected epoch observer failure")
try:
    run_training(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), on_epoch=abort)
except TrainingRunError as error:
    print(error, file=sys.stderr)
    sys.exit(1)
"""
        args = [str(data), str(output), str(config), str(vocabulary)]
    else:
        script += "from filing_sentence_classifier.cli import app\napp()\n"
        args = [
            "train",
            "--data-dir",
            str(data),
            "--vocabulary-dir",
            str(vocabulary),
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ]
        if tracking_dir is not None:
            args.extend(
                ["--mlflow-dir", str(tracking_dir), "--experiment-name", "tests"]
            )
    return subprocess.run(
        [sys.executable, "-c", script, *args],
        cwd=output.parent,
        env={
            **os.environ,
            "PYTHONHASHSEED": hash_seed,
            "MPLCONFIGDIR": os.environ.get(
                "MPLCONFIGDIR", str(output.parent / "mpl-cache")
            ),
        },
        capture_output=True,
        text=True,
        timeout=45,
    )


def verify_inventory(output: Path):
    manifest = json.loads((output / "manifest.json").read_bytes())
    for name, metadata in manifest["files"].items():
        content = (output / name).read_bytes()
        assert metadata == {
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    return manifest


@pytest.mark.training
@pytest.mark.parametrize("tracking", [False, True])
def test_cli_reproduces_history_predictions_and_weights_and_saves_reloadable_run(
    inputs, tmp_path, tracking
):
    if tracking:
        pytest.importorskip("mlflow")
    data, vocabulary, config_path = inputs
    before = {path.name: path.read_bytes() for path in data.iterdir()}
    first, second = tmp_path / "first", tmp_path / "second"
    result = invoke(
        inputs, first, tracking_dir=tmp_path / "mlflow" if tracking else None
    )
    assert result.returncode == 0, result.stderr
    assert "Epoch 01" in result.stdout and "Completed training run" in result.stdout
    manifest = verify_inventory(first)
    assert manifest["status"] == "completed"
    assert (
        manifest["source"]["git"] is None
    )  # The command also works outside a checkout.
    assert manifest["source"]["code_sha256"]
    for name, expected_hash in manifest["source"]["code_sha256"].items():
        assert (
            hashlib.sha256(
                (
                    first / "source" / "src" / "filing_sentence_classifier" / name
                ).read_bytes()
            ).hexdigest()
            == expected_hash
        )
    assert (first / "config.toml").read_bytes() == config_path.read_bytes()
    assert (first / "learning-curves.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    history = [
        json.loads(line) for line in (first / "history.jsonl").read_text().splitlines()
    ]
    scores = [row["validation_metrics"]["macro_f1"] for row in history]
    summary = json.loads((first / "summary.json").read_bytes())
    assert summary["best_epoch"] == scores.index(max(scores)) + 1
    assert summary["epochs_completed"] == len(history)
    assert summary["validation_macro_f1"] == max(scores)
    metrics = json.loads((first / "metrics.val.json").read_bytes())
    assert metrics["metrics"]["macro_f1"] == max(scores)
    if tracking:
        from mlflow import MlflowClient

        connection = manifest["tracking"]
        client = MlflowClient(tracking_uri=connection["tracking_uri"])
        run = client.get_run(connection["run_id"])
        assert run.info.status == "FINISHED"
        assert run.data.params["config.model.dropout"] == "0.2"
        assert (
            run.data.params["data.manifest_sha256"]
            == manifest["data"]["manifest_sha256"]
        )
        assert run.data.params["preprocessing.max_length"] == "128"
        assert run.data.metrics["summary/validation_macro_f1"] == max(scores)
        assert run.data.metrics["summary/best_epoch"] == summary["best_epoch"]
        remote_history = client.get_metric_history(run.info.run_id, "val/macro_f1")
        assert [(row.step, row.value) for row in remote_history] == list(
            enumerate(scores, 1)
        )
        # The uploaded manifest and every inventoried artifact match the local run.
        downloaded = Path(client.download_artifacts(run.info.run_id, ""))
        assert (downloaded / "manifest.json").read_bytes() == (
            first / "manifest.json"
        ).read_bytes()
        assert verify_inventory(downloaded) == manifest

    # A second fresh process changes Python's hash seed, keeping the training seed.
    repeated = invoke(inputs, second, "2")
    assert repeated.returncode == 0, repeated.stderr
    for name in (
        "history.jsonl",
        "predictions.val.jsonl",
        "metrics.val.json",
        "encoder.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    weights = [
        torch.load(path / "checkpoints/best.pt", weights_only=True)
        for path in (first, second)
    ]
    assert weights[0]["epoch"] == weights[1]["epoch"]
    assert all(
        torch.equal(value, weights[1]["model_state_dict"][name])
        for name, value in weights[0]["model_state_dict"].items()
    )

    # Restore using only saved settings/encoder/weights, then check the frozen val.
    config = TrainingConfig.from_toml((first / "config.toml").read_bytes())
    encoder = TextEncoder.from_dict(json.loads((first / "encoder.json").read_bytes()))
    val = load_split(data, "val")
    with torch.random.fork_rng(devices=[]):
        model = MeanPoolMLP(
            len(encoder.vocabulary),
            embedding_dim=config.model.embedding_dim,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
        )
        load_checkpoint(first / "checkpoints/best.pt", model, expected_config=config)
        checked = validate_epoch(
            model,
            create_dataloader(SentenceDataset(val, encoder), batch_size=13),
            label_ids=val.label_ids,
            expected_sample_ids=val.sample_ids,
        )
    assert checked.metrics.macro_f1 == summary["validation_macro_f1"]
    saved = [
        json.loads(line)
        for line in (first / "predictions.val.jsonl").read_text().splitlines()
    ]
    assert [row["predicted_label"] for row in saved] == list(checked.predicted_labels)
    assert {path.name: path.read_bytes() for path in data.iterdir()} == before
    assert invoke(inputs, first).returncode == 1
    assert verify_inventory(first) == manifest


@pytest.mark.training
@pytest.mark.parametrize("failure", ["epoch_tracking", "cleanup", "interrupt"])
def test_tracking_failures_and_interruptions_preserve_local_evidence(
    inputs, tmp_path, failure
):
    mlflow = pytest.importorskip("mlflow")
    data, vocabulary, config = inputs
    output = tmp_path / "failed-tracked"
    script = """import sys
from pathlib import Path
from filing_sentence_classifier.training.run import run_training
from filing_sentence_classifier.training.tracking import MLflowTracker
failure = sys.argv[1]
tracker = MLflowTracker(Path(sys.argv[6]), 'failure-tests')
if failure == 'epoch_tracking':
    def broken_logging(row):
        raise OSError('injected metric failure')
    tracker.log_epoch = broken_logging
if failure == 'cleanup':
    from mlflow import MlflowClient
    def broken_copy(*args, **kwargs):
        raise OSError('injected artifact failure')
    MlflowClient.log_artifacts = broken_copy
def observer(row):
    if failure == 'interrupt':
        raise KeyboardInterrupt('injected interruption')
    raise ValueError('original observer failure')
run_training(
    Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]),
    tracker=tracker, on_epoch=observer,
)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            failure,
            str(data),
            str(output),
            str(config),
            str(vocabulary),
            str(tmp_path / "mlflow"),
        ],
        env={**os.environ, "MLFLOW_DISABLE_TELEMETRY": "true"},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode != 0
    manifest = verify_inventory(output)
    assert manifest["status"] == "failed"
    assert len((output / "history.jsonl").read_text().splitlines()) == 1
    assert (output / "checkpoints/best.pt").is_file()
    assert not (output / "summary.json").exists()
    if failure == "cleanup":
        assert manifest["error"]["message"] == "original observer failure"
        assert manifest["tracking_error"]["message"] == "injected artifact failure"
    elif failure == "epoch_tracking":
        assert manifest["error"]["message"] == "injected metric failure"
    else:
        assert manifest["error"]["type"] == "KeyboardInterrupt"
    connection = manifest["tracking"]
    client = mlflow.MlflowClient(tracking_uri=connection["tracking_uri"])
    run = client.get_run(connection["run_id"])
    assert run.info.status == ("KILLED" if failure == "interrupt" else "FAILED")
    if failure != "cleanup":
        downloaded = Path(client.download_artifacts(run.info.run_id, ""))
        assert verify_inventory(downloaded) == manifest


@pytest.mark.training
def test_failed_run_keeps_history_checkpoint_and_error(inputs, tmp_path):
    output = tmp_path / "failed"
    result = invoke(inputs, output, fail=True)
    assert result.returncode == 1
    assert "injected epoch observer failure" in result.stderr
    manifest = verify_inventory(output)
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "RuntimeError"
    assert len((output / "history.jsonl").read_text().splitlines()) == 1
    assert (output / "checkpoints/best.pt").is_file()
    assert not (output / "summary.json").exists()


@pytest.mark.parametrize("change", ["provenance", "checksum", "config"])
def test_bad_inputs_are_recorded_as_failed_without_training(inputs, tmp_path, change):
    data, vocabulary, config = inputs
    if change == "provenance":
        manifest_path = vocabulary / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["data"]["train_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif change == "checksum":
        with (vocabulary / "vocabulary.json").open("a") as file:
            file.write(" ")
    else:
        config.write_text(config.read_text().replace("patience = 2", "patience = 0"))
    output = tmp_path / "invalid"
    result = CliRunner().invoke(
        app,
        [
            "train",
            "--data-dir",
            str(data),
            "--vocabulary-dir",
            str(vocabulary),
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 1, result.output
    manifest = verify_inventory(output)
    assert manifest["status"] == "failed"
    assert not (output / "checkpoints").exists()
