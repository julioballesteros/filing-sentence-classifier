"""Exercise common evaluation with synthetic predictions and saved partitions."""

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from filing_sentence_classifier.cli import app
from filing_sentence_classifier.data.loading import load_split


def prediction_file(artifact: Path, path: Path) -> Path:
    split = load_split(artifact, "val")
    # A synthetic correctness fixture, not a measured model result.
    rows = [
        {"sample_id": row.sample_id, "predicted_label": row.label}
        for row in reversed(split.records)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_cli_outputs_json_metrics_with_provenance(
    development_artifact: Path, tmp_path: Path
) -> None:
    predictions = prediction_file(development_artifact, tmp_path / "predictions.jsonl")
    args = [
        "evaluate",
        "--data-dir",
        str(development_artifact),
        "--predictions",
        str(predictions),
    ]
    runner = CliRunner()
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["split"] == "val"
    assert report["metrics"]["accuracy"] == report["metrics"]["macro_f1"] == 1.0
    assert report["metrics"]["label_ids"] == [0, 1, 2]
    assert report["confusion_matrix_axes"] == {"rows": "true", "columns": "predicted"}
    assert report["primary_metric"] == "macro_f1"
    assert report["zero_division"] == 0
    assert (
        report["data_manifest_sha256"]
        == hashlib.sha256(
            (development_artifact / "manifest.json").read_bytes()
        ).hexdigest()
    )
    assert (
        report["predictions_sha256"]
        == hashlib.sha256(predictions.read_bytes()).hexdigest()
    )
    assert runner.invoke(app, args).stdout == result.stdout


@pytest.mark.parametrize(
    "case", ["missing_id", "bad_json", "wrong_schema", "float_label", "other_split"]
)
def test_cli_reports_invalid_predictions_without_metrics(
    development_artifact: Path, tmp_path: Path, case: str
) -> None:
    predictions = prediction_file(development_artifact, tmp_path / "predictions.jsonl")
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    if case == "missing_id":
        rows.pop()
    elif case == "wrong_schema":
        rows[0]["label"] = rows[0].pop("predicted_label")
    elif case == "float_label":
        rows[0]["predicted_label"] = 0.0
    elif case == "other_split":
        rows[0]["sample_id"] = load_split(development_artifact, "train").sample_ids[0]
    predictions.write_text(
        "not JSON"
        if case == "bad_json"
        else "".join(json.dumps(row) + "\n" for row in rows)
    )
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--data-dir",
            str(development_artifact),
            "--predictions",
            str(predictions),
        ],
    )
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert '"metrics"' not in result.output


def test_cli_rejects_test_before_loading_any_files() -> None:
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--data-dir",
            "missing",
            "--predictions",
            "missing.jsonl",
            "--split",
            "test",
        ],
    )
    assert result.exit_code == 2
    assert "not one of" in result.output
