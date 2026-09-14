"""Exercise a complete baseline run against synthetic frozen development data."""

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from typer.testing import CliRunner

from filing_sentence_classifier.baselines import run
from filing_sentence_classifier.baselines.majority import (
    BaselineError,
    MajorityClassifier,
)
from filing_sentence_classifier.cli import app
from filing_sentence_classifier.data.loading import DataLoadError, load_split


def artifact_bytes(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in directory.iterdir()}


def test_cli_saves_a_reproducible_run_and_uses_common_evaluation(
    development_artifact: Path, tmp_path: Path
) -> None:
    train, val = (load_split(development_artifact, name) for name in ("train", "val"))
    before = artifact_bytes(development_artifact)
    output = tmp_path / "run"
    args = [
        "baseline",
        "majority",
        "--data-dir",
        str(development_artifact),
        "--output-dir",
        str(output),
        "--manifest-sha256",
        train.manifest_sha256,
    ]
    runner = CliRunner()
    response = runner.invoke(app, args)
    assert response.exit_code == 0, response.output
    payloads = artifact_bytes(output)
    assert set(payloads) == {
        "model.json",
        "predictions.val.jsonl",
        "metrics.val.json",
        "manifest.json",
    }
    model = MajorityClassifier.from_dict(json.loads(payloads["model.json"]))
    counts = Counter(train.targets)
    label = min(counts, key=lambda item: (-counts[item], item))
    assert model.majority_label == label
    assert model.class_counts == tuple(counts[item] for item in train.label_ids)
    rows = [json.loads(line) for line in payloads["predictions.val.jsonl"].splitlines()]
    assert [row["sample_id"] for row in rows] == list(val.sample_ids)
    assert {row["predicted_label"] for row in rows} == {label}
    report = json.loads(payloads["metrics.val.json"])
    correct = val.targets.count(label)
    assert report["metrics"]["accuracy"] == pytest.approx(correct / len(val.records))
    assert report["metrics"]["macro_f1"] == pytest.approx(
        2 * correct / (len(val.records) + correct) / len(val.label_ids)
    )
    evaluated = runner.invoke(
        app,
        [
            "evaluate",
            "--data-dir",
            str(development_artifact),
            "--predictions",
            str(output / "predictions.val.jsonl"),
        ],
    )
    assert evaluated.exit_code == 0, evaluated.output
    assert json.loads(evaluated.stdout) == report
    manifest = json.loads(payloads["manifest.json"])
    assert manifest["data"]["manifest_sha256"] == train.manifest_sha256
    assert manifest["data"]["train_sha256"] == train.records_sha256
    assert manifest["data"]["val_sha256"] == val.records_sha256
    assert manifest["recipe"]["fit_split"] == "train"
    assert manifest["recipe"]["evaluation_split"] == "val"
    assert manifest["recipe"]["seed"] is None
    for name, metadata in manifest["files"].items():
        assert metadata["size_bytes"] == len(payloads[name])
        assert metadata["sha256"] == hashlib.sha256(payloads[name]).hexdigest()
    mtimes = {path.name: path.stat().st_mtime_ns for path in output.iterdir()}
    assert runner.invoke(app, args).exit_code == 0
    assert artifact_bytes(output) == payloads
    assert mtimes == {path.name: path.stat().st_mtime_ns for path in output.iterdir()}
    rebuilt = run.run_majority_baseline(development_artifact, tmp_path / "rebuilt")
    assert artifact_bytes(rebuilt) == payloads
    assert artifact_bytes(development_artifact) == before
    restored = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys; "
            "from pathlib import Path; "
            "from filing_sentence_classifier.baselines.majority import MajorityClassifier; "
            "model = MajorityClassifier.from_dict(json.loads(Path(sys.argv[1]).read_text())); "
            "print(json.dumps(model.predict(['A new sentence.'])))",
            str(output / "model.json"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(restored.stdout) == [label]


def test_fits_only_train_before_loading_validation_and_never_reads_raw_data(
    development_artifact: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = load_split(development_artifact, "train")
    original_fit = MajorityClassifier.fit
    original_load = run.load_split
    read = Path.read_bytes
    events = []

    def fit(cls, targets, *, label_ids):
        assert tuple(targets) == train.targets
        assert tuple(label_ids) == train.label_ids
        events.append("fit")
        return original_fit(targets, label_ids=label_ids)

    def load(directory, split, **kwargs):
        events.append(split)
        return original_load(directory, split, **kwargs)

    def no_source_reads(path):
        assert path.suffix != ".parquet"
        assert "raw" not in path.parts and "interim" not in path.parts
        return read(path)

    monkeypatch.setattr(MajorityClassifier, "fit", classmethod(fit))
    monkeypatch.setattr(run, "load_split", load)
    monkeypatch.setattr(Path, "read_bytes", no_source_reads)
    run.run_majority_baseline(development_artifact, tmp_path / "run")
    assert events == ["train", "fit", "val"]


@pytest.mark.parametrize(
    "filename", ["model.json", "metrics.val.json", "predictions.val.jsonl"]
)
def test_changed_existing_run_is_never_overwritten(
    development_artifact: Path, tmp_path: Path, filename: str
) -> None:
    output = run.run_majority_baseline(development_artifact, tmp_path / "run")
    (output / filename).write_text("changed")
    before = artifact_bytes(output)
    with pytest.raises(BaselineError, match="Existing run differs"):
        run.run_majority_baseline(development_artifact, output)
    assert artifact_bytes(output) == before
    assert not list(tmp_path.glob(".majority-*"))


def test_corrupt_data_and_wrong_pin_do_not_publish_a_run(
    development_artifact: Path, tmp_path: Path
) -> None:
    output = tmp_path / "run"
    with pytest.raises(DataLoadError, match="Manifest checksum"):
        run.run_majority_baseline(
            development_artifact, output, expected_manifest_sha256="0" * 64
        )
    (development_artifact / "val.jsonl").write_text("changed")
    result = CliRunner().invoke(
        app,
        [
            "baseline",
            "majority",
            "--data-dir",
            str(development_artifact),
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 1
    assert "Integrity check failed" in result.output
    assert not output.exists()


def test_evaluation_failure_leaves_no_partial_run(
    development_artifact: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args, **kwargs):
        raise ValueError("synthetic evaluation failure")

    monkeypatch.setattr(run, "evaluate_prediction_file", fail)
    output = tmp_path / "run"
    with pytest.raises(BaselineError, match="synthetic evaluation failure"):
        run.run_majority_baseline(development_artifact, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".majority-*"))
