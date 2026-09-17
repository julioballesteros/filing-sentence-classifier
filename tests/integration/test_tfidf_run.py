"""Exercise the TF-IDF CLI, input boundary, and complete frozen run artifacts."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from filing_sentence_classifier.baselines import tfidf_run
from filing_sentence_classifier.baselines.majority import BaselineError
from filing_sentence_classifier.baselines.tfidf import load_tfidf_model, predict_labels
from filing_sentence_classifier.cli import app
from filing_sentence_classifier.data.loading import DataLoadError, load_split


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(
        "[tfidf]\nngram_range = [1, 2]\nmin_df = 2\n[logistic_regression]\nc = 1.0\nmax_iter = 1000\ntol = 0.0001\n"
    )
    return path


def payloads(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in directory.iterdir()}


def test_cli_saves_pipeline_and_matches_common_evaluation(
    development_artifact: Path,
    config_path: Path,
    tmp_path: Path,
) -> None:
    original_data = payloads(development_artifact)
    output = tmp_path / "run"
    args = [
        "baseline",
        "tfidf",
        "--data-dir",
        str(development_artifact),
        "--config",
        str(config_path),
        "--output-dir",
        str(output),
    ]
    runner = CliRunner()
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    before = payloads(output)
    manifest = json.loads(before["manifest.json"])
    assert set(before) == {
        "manifest.json",
        "model.json",
        "model.joblib",
        "config.toml",
        "predictions.val.jsonl",
        "metrics.val.json",
    }
    assert before["config.toml"] == config_path.read_bytes()
    assert manifest["recipe"]["tfidf"]["ngram_range"] == [1, 2]
    assert manifest["recipe"]["logistic_regression"]["C"] == 1.0
    assert manifest["recipe"]["thread_limit"] == 1
    val = load_split(development_artifact, "val")
    assert manifest["data"]["manifest_sha256"] == val.manifest_sha256
    assert manifest["data"]["val_sha256"] == val.records_sha256
    for name, metadata in manifest["files"].items():
        assert metadata["size_bytes"] == len(before[name])
        assert metadata["sha256"] == hashlib.sha256(before[name]).hexdigest()
    restored = load_tfidf_model(
        output / "model.joblib",
        expected_sha256=manifest["files"]["model.joblib"]["sha256"],
    )
    predictions = [
        json.loads(line) for line in before["predictions.val.jsonl"].splitlines()
    ]
    assert [row["sample_id"] for row in predictions] == list(val.sample_ids)
    assert tuple(row["predicted_label"] for row in predictions) == predict_labels(
        restored, val.texts
    )
    evaluation = runner.invoke(
        app,
        [
            "evaluate",
            "--data-dir",
            str(development_artifact),
            "--predictions",
            str(output / "predictions.val.jsonl"),
        ],
    )
    assert evaluation.exit_code == 0, evaluation.output
    assert json.loads(evaluation.stdout) == json.loads(before["metrics.val.json"])
    mtimes = {path.name: path.stat().st_mtime_ns for path in output.iterdir()}
    repeated = runner.invoke(app, args)
    assert repeated.exit_code == 0, repeated.output
    assert payloads(output) == before
    assert {path.name: path.stat().st_mtime_ns for path in output.iterdir()} == mtimes
    second = tfidf_run.run_tfidf_baseline(
        development_artifact, tmp_path / "second", config_path
    )
    assert payloads(second) == before
    assert payloads(development_artifact) == original_data
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; "
            "from filing_sentence_classifier.baselines.tfidf import load_tfidf_model, predict_labels; "
            "from filing_sentence_classifier.baselines.tfidf_run import run_tfidf_baseline; "
            "run_tfidf_baseline(Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])); "
            "model=load_tfidf_model(Path(sys.argv[1]), expected_sha256=sys.argv[2]); "
            "print(predict_labels(model, ['Plan 1-2.'])[0])",
            str(output / "model.joblib"),
            manifest["files"]["model.joblib"]["sha256"],
            str(development_artifact),
            str(tmp_path / "child-run"),
            str(config_path),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert int(child.stdout) == predict_labels(restored, ["Plan 1-2."])[0]
    child_files = payloads(tmp_path / "child-run")
    for name in (
        "config.toml",
        "model.json",
        "predictions.val.jsonl",
        "metrics.val.json",
    ):
        assert child_files[name] == before[name]
    child_manifest = json.loads(child_files["manifest.json"])
    assert {key: value for key, value in child_manifest.items() if key != "files"} == {
        key: value for key, value in manifest.items() if key != "files"
    }
    for name, metadata in child_manifest["files"].items():
        assert metadata == {
            "size_bytes": len(child_files[name]),
            "sha256": hashlib.sha256(child_files[name]).hexdigest(),
        }
    # Pickle memoization can encode equal strings as different shared references
    # across processes. Require identical learned state, not identical new pickle
    # bytes; each published file still has its own verified checksum.
    child_model = load_tfidf_model(
        tmp_path / "child-run/model.joblib",
        expected_sha256=child_manifest["files"]["model.joblib"]["sha256"],
    )
    for name in ("tfidf", "logreg"):
        assert child_model.named_steps[name].get_params(
            deep=False
        ) == restored.named_steps[name].get_params(deep=False)
    assert (
        child_model.named_steps["tfidf"].vocabulary_
        == restored.named_steps["tfidf"].vocabulary_
    )
    for name, attribute in (
        ("tfidf", "idf_"),
        ("logreg", "classes_"),
        ("logreg", "coef_"),
        ("logreg", "intercept_"),
        ("logreg", "n_iter_"),
    ):
        np.testing.assert_array_equal(
            getattr(child_model.named_steps[name], attribute),
            getattr(restored.named_steps[name], attribute),
        )
    np.testing.assert_array_equal(
        child_model.predict_proba(val.texts), restored.predict_proba(val.texts)
    )


def test_fit_is_finished_before_val_is_loaded_and_no_raw_data_is_read(
    development_artifact: Path,
    config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = load_split(development_artifact, "train")
    original_fit, original_load, read = (
        tfidf_run.fit_tfidf,
        tfidf_run.load_split,
        Path.read_bytes,
    )
    events = []

    def fit(texts, targets, **kwargs):
        assert texts == train.texts and targets == train.targets
        events.append("fit")
        return original_fit(texts, targets, **kwargs)

    def load(directory, split, **kwargs):
        events.append(split)
        return original_load(directory, split, **kwargs)

    def no_raw(path):
        assert path.suffix != ".parquet"
        assert "raw" not in path.parts and "interim" not in path.parts
        return read(path)

    monkeypatch.setattr(tfidf_run, "fit_tfidf", fit)
    monkeypatch.setattr(tfidf_run, "load_split", load)
    monkeypatch.setattr(Path, "read_bytes", no_raw)
    tfidf_run.run_tfidf_baseline(development_artifact, tmp_path / "run", config_path)
    assert events == ["train", "fit", "val"]


def test_invalid_inputs_or_nonconvergence_leave_no_run(
    development_artifact: Path,
    config_path: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    with pytest.raises(DataLoadError, match="Manifest checksum"):
        tfidf_run.run_tfidf_baseline(
            development_artifact, output, config_path, expected_manifest_sha256="0" * 64
        )
    config_path.write_text(
        config_path.read_text().replace("max_iter = 1000", "max_iter = 1")
    )
    result = CliRunner().invoke(
        app,
        [
            "baseline",
            "tfidf",
            "--data-dir",
            str(development_artifact),
            "--config",
            str(config_path),
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 1
    assert "did not converge" in result.output
    assert not output.exists()
    assert not list(tmp_path.glob(".tfidf-*"))


@pytest.mark.parametrize("change", ["config", "model", "evaluation_failure"])
def test_existing_results_are_preserved_on_changes_or_failure(
    development_artifact: Path,
    config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    output = tfidf_run.run_tfidf_baseline(
        development_artifact, tmp_path / "run", config_path
    )
    if change == "model":
        (output / "model.joblib").write_bytes(b"changed")
    elif change == "config":
        config_path.write_text(config_path.read_text().replace("c = 1.0", "c = 2.0"))
    else:

        def fail(*args, **kwargs):
            raise ValueError("synthetic failure")

        monkeypatch.setattr(tfidf_run, "evaluate_prediction_file", fail)
    before = payloads(output)
    with pytest.raises(BaselineError):
        tfidf_run.run_tfidf_baseline(development_artifact, output, config_path)
    assert payloads(output) == before
    assert not list(tmp_path.glob(".tfidf-*"))
