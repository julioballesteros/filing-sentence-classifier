"""Check the optional local tracking boundary without fitting a model."""

import json
import os
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from filing_sentence_classifier.cli import app
from filing_sentence_classifier.training.tracking import MLflowTracker


@pytest.fixture
def mlflow(monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    return pytest.importorskip("mlflow")


def test_local_tracking_does_not_adopt_or_modify_the_active_run(mlflow, tmp_path):
    previous_uri = mlflow.get_tracking_uri()
    unrelated_uri = f"sqlite:///{tmp_path / 'unrelated.db'}"
    mlflow.set_tracking_uri(unrelated_uri)
    try:
        with mlflow.start_run() as active:
            run_dir = tmp_path / "run"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text('{"status": "completed"}\n')
            tracker = MLflowTracker(tmp_path / "tracking", "experiment")
            metadata = tracker.start(run_dir)
            tracker.log_summary({"validation_macro_f1": 0.7, "best_epoch": 3})
            tracker.finish(run_dir, "FINISHED")
            assert mlflow.get_tracking_uri() == unrelated_uri
            assert mlflow.active_run().info.run_id == active.info.run_id
            assert metadata["run_id"] != active.info.run_id
            assert mlflow.get_run(active.info.run_id).data.metrics == {}
            assert mlflow.get_run(active.info.run_id).info.status == "RUNNING"
            # Another instance reuses the experiment and creates a distinct run.
            another = MLflowTracker(tmp_path / "tracking", "experiment")
            repeated = another.start(run_dir)
            assert repeated["experiment_id"] == metadata["experiment_id"]
            assert repeated["run_id"] != metadata["run_id"]
            another.finish(run_dir, "FAILED")
    finally:
        mlflow.set_tracking_uri(previous_uri)


@pytest.mark.parametrize("relative", ["run", "run/tracking", "."])
def test_overlapping_storage_is_rejected_before_creation(tmp_path, relative):
    tracker = MLflowTracker(tmp_path / relative)
    with pytest.raises(ValueError, match="overlap"):
        tracker.start(tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_existing_remote_artifact_location_is_rejected(mlflow, tmp_path):
    directory = tmp_path / "tracking"
    directory.mkdir()
    client = mlflow.MlflowClient(tracking_uri=f"sqlite:///{directory / 'mlflow.db'}")
    experiment_id = client.create_experiment("remote", artifact_location="s3://unused")
    with pytest.raises(ValueError, match="different artifact location"):
        MLflowTracker(directory, "remote").start(tmp_path / "run")
    assert client.search_runs([experiment_id]) == []


def test_artifact_copy_failure_marks_mlflow_failed(mlflow, tmp_path, monkeypatch):
    tracker = MLflowTracker(tmp_path / "tracking")
    metadata = tracker.start(tmp_path / "run")

    def broken_copy(*args, **kwargs):
        raise OSError("artifact copy failed")

    monkeypatch.setattr(mlflow.MlflowClient, "log_artifacts", broken_copy)
    with pytest.raises(OSError, match="artifact copy failed"):
        tracker.finish(tmp_path / "run", "FINISHED")
    client = mlflow.MlflowClient(tracking_uri=metadata["tracking_uri"])
    assert client.get_run(metadata["run_id"]).info.status == "FAILED"


def test_bad_training_inputs_close_the_tracked_attempt(mlflow, tmp_path):
    output = tmp_path / "invalid-run"
    result = CliRunner().invoke(
        app,
        [
            "train",
            "--data-dir",
            "unused",
            "--vocabulary-dir",
            "unused",
            "--config",
            str(tmp_path / "missing.toml"),
            "--output-dir",
            str(output),
            "--mlflow-dir",
            str(tmp_path / "tracking"),
        ],
    )
    assert result.exit_code == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "FileNotFoundError"
    connection = manifest["tracking"]
    client = mlflow.MlflowClient(tracking_uri=connection["tracking_uri"])
    assert client.get_run(connection["run_id"]).info.status == "FAILED"


def test_tracking_dependency_is_optional_and_missing_extra_is_actionable(tmp_path):
    script = """import builtins, sys
from pathlib import Path
original_import = builtins.__import__
def missing_mlflow(name, *args, **kwargs):
    if name.split('.')[0] == 'mlflow':
        raise ModuleNotFoundError('MLflow is not installed', name='mlflow')
    return original_import(name, *args, **kwargs)
builtins.__import__ = missing_mlflow
from filing_sentence_classifier.training.tracking import MLflowTracker
assert 'mlflow' not in sys.modules
from filing_sentence_classifier.cli import app
sys.argv = ['fsc', 'train', '--data-dir', 'unused', '--vocabulary-dir', 'unused',
            '--config', 'unused.toml', '--output-dir', sys.argv[1],
            '--mlflow-dir', sys.argv[2]]
app()
"""
    output = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, "-c", script, str(output), str(tmp_path / "tracking")],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "MLFLOW_DISABLE_TELEMETRY": "true"},
    )
    assert result.returncode == 1
    assert "[tracking] extra" in result.stderr
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    assert not (tmp_path / "tracking").exists()


@pytest.mark.parametrize(
    "args",
    [
        ["--experiment-name", "example"],
        ["--mlflow-dir", "unused", "--experiment-name", " "],
    ],
)
def test_invalid_tracking_options_are_cli_errors(args):
    result = CliRunner().invoke(
        app,
        [
            "train",
            "--data-dir",
            "unused",
            "--vocabulary-dir",
            "unused",
            "--config",
            "unused",
            "--output-dir",
            "unused",
            *args,
        ],
    )
    assert result.exit_code == 2
    assert "nonblank experiment name" in result.output
