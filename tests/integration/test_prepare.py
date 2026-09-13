"""Exercise real Parquet preparation, traceability, and artifact integrity offline."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from filing_sentence_classifier import cli
from filing_sentence_classifier.data.prepare import (
    PreparationError,
    prepare_training_data,
)
from filing_sentence_classifier.data.spec import DatasetSource, SourceFile

LABELS = {0: "specific", 1: "historical", 2: "generic"}


@pytest.fixture
def training_source(tmp_path: Path) -> tuple[DatasetSource, SourceFile]:
    rows = [
        {"text": " Company\u0092s  plan ", "label": 0, "label_text": "specific"},
        {"text": "Company’s plan", "label": 0, "label_text": "specific"},
        {"text": "Conflicting text", "label": 1, "label_text": "historical"},
        {"text": "Conflicting text", "label": 2, "label_text": "generic"},
        {"text": None, "label": 0, "label_text": "specific"},
        {"text": "A different plan", "label": 0, "label_text": "specific"},
    ]
    revision = "b" * 40
    path = tmp_path / "raw" / revision / "data/train.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    content = path.read_bytes()
    train_file = SourceFile(
        "data/train.parquet", len(content), hashlib.sha256(content).hexdigest()
    )
    # Deliberately absent: preparation must not open or verify other source files.
    test_file = SourceFile("data/test.parquet", 123, "c" * 64)
    return DatasetSource(
        "tests/synthetic", revision, (train_file, test_file)
    ), train_file


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def artifact_bytes(path: Path) -> dict[str, bytes]:
    return {file.name: file.read_bytes() for file in path.iterdir()}


def test_cli_preparation_preserves_identity_and_reproduces_every_file(
    tmp_path: Path,
    training_source: tuple[DatasetSource, SourceFile],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, train_file = training_source
    monkeypatch.setattr(cli, "FLS_SOURCE", dataset)
    monkeypatch.setattr(cli, "FLS_TRAIN_FILE", train_file)
    monkeypatch.setattr(cli, "LABEL_NAMES", LABELS)
    args = [
        "prepare-data",
        "--raw-dir",
        str(tmp_path / "raw"),
        "--output-dir",
        str(tmp_path / "interim"),
    ]
    runner = CliRunner()
    response = runner.invoke(cli.app, args)
    assert response.exit_code == 0, response.output
    artifact = tmp_path / "interim" / dataset.revision / "clean-v1"
    assert str(artifact) in response.output
    contents = artifact_bytes(artifact)
    assert set(contents) == {
        "records.jsonl",
        "excluded.jsonl",
        "changes.jsonl",
        "manifest.json",
    }
    kept = read_jsonl(artifact / "records.jsonl")
    excluded = read_jsonl(artifact / "excluded.jsonl")
    changes = read_jsonl(artifact / "changes.jsonl")
    assert [row["source_row"] for row in kept] == [0, 1, 5]
    assert [row["source_row"] for row in excluded] == [2, 3, 4]
    assert [row["source_row"] for row in changes] == [0]
    assert len({row["sample_id"] for row in kept + excluded}) == 6
    assert changes[0]["sample_id"] == kept[0]["sample_id"]
    assert kept[0]["group_id"] == kept[1]["group_id"]
    assert all(row["label"] == 0 and row["label_text"] == "specific" for row in kept)
    assert "split" not in kept[0]
    manifest = json.loads(contents["manifest.json"])
    assert manifest["counts"]["input"] == 6
    assert manifest["counts"]["kept"] == manifest["counts"]["excluded"] == 3
    assert manifest["counts"]["changed"] == 1
    assert manifest["counts"]["kept_groups"] == 2
    assert manifest["source"]["file"]["sha256"] == train_file.sha256
    for name, metadata in manifest["files"].items():
        assert hashlib.sha256(contents[name]).hexdigest() == metadata["sha256"]
        assert len(contents[name]) == metadata["size_bytes"]
    for row in kept + excluded:
        identity = {"source": manifest["source"], "source_row": row["source_row"]}
        expected_id = hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        assert row["sample_id"] == expected_id

    mtimes = {path.name: path.stat().st_mtime_ns for path in artifact.iterdir()}
    assert runner.invoke(cli.app, args).exit_code == 0
    assert contents == artifact_bytes(artifact)
    assert mtimes == {path.name: path.stat().st_mtime_ns for path in artifact.iterdir()}
    rebuilt = prepare_training_data(
        dataset, train_file, LABELS, tmp_path / "raw", tmp_path / "rebuilt"
    )
    assert contents == artifact_bytes(rebuilt)
    raw_path = tmp_path / "raw" / dataset.revision / train_file.path
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == train_file.sha256


@pytest.mark.parametrize(
    "name", ["records.jsonl", "excluded.jsonl", "changes.jsonl", "manifest.json"]
)
def test_existing_corrupt_artifact_is_not_overwritten(
    tmp_path: Path, training_source: tuple[DatasetSource, SourceFile], name: str
) -> None:
    dataset, train_file = training_source
    args = (dataset, train_file, LABELS, tmp_path / "raw", tmp_path / "interim")
    artifact = prepare_training_data(*args)
    (artifact / name).write_bytes(b"corrupt")
    before = artifact_bytes(artifact)
    with pytest.raises(PreparationError, match="Left unchanged"):
        prepare_training_data(*args)
    assert artifact_bytes(artifact) == before


def test_source_integrity_failure_prevents_parsing_and_publication(
    tmp_path: Path,
    training_source: tuple[DatasetSource, SourceFile],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, train_file = training_source
    path = tmp_path / "raw" / dataset.revision / train_file.path
    content = path.read_bytes()
    path.write_bytes(b"X" + content[1:])

    def forbidden_read(*args: object, **kwargs: object) -> None:
        pytest.fail("An unverified input must not be parsed.")

    monkeypatch.setattr(pq, "read_table", forbidden_read)
    with pytest.raises(PreparationError, match="integrity check failed"):
        prepare_training_data(
            dataset, train_file, LABELS, tmp_path / "raw", tmp_path / "interim"
        )
    assert not (tmp_path / "interim").exists()


@pytest.mark.parametrize("case", ["invalid_parquet", "wrong_columns", "all_invalid"])
def test_unusable_source_fails_without_publishing_an_artifact(
    tmp_path: Path, training_source: tuple[DatasetSource, SourceFile], case: str
) -> None:
    dataset, train_file = training_source
    path = tmp_path / "raw" / dataset.revision / train_file.path
    if case == "invalid_parquet":
        path.write_bytes(b"not parquet")
        message = "Could not read"
    elif case == "wrong_columns":
        pq.write_table(pa.table({"sentence": ["Plan"]}), path)
        message = "Unexpected training columns"
    else:
        pq.write_table(
            pa.table({"text": [None], "label": [0], "label_text": ["specific"]}), path
        )
        message = "no eligible training records"
    content = path.read_bytes()
    train_file = replace(
        train_file, size_bytes=len(content), sha256=hashlib.sha256(content).hexdigest()
    )
    dataset = replace(dataset, files=(train_file,))
    with pytest.raises(PreparationError, match=message):
        prepare_training_data(
            dataset, train_file, LABELS, tmp_path / "raw", tmp_path / "interim"
        )
    assert not (tmp_path / "interim").exists()


def test_output_cannot_be_written_into_the_source_snapshot(
    tmp_path: Path, training_source: tuple[DatasetSource, SourceFile]
) -> None:
    dataset, train_file = training_source
    with pytest.raises(PreparationError, match="outside the source snapshot"):
        prepare_training_data(
            dataset, train_file, LABELS, tmp_path / "raw", tmp_path / "raw"
        )
    assert not (tmp_path / "raw" / dataset.revision / "clean-v1").exists()


def test_failed_write_leaves_no_partial_artifact(
    tmp_path: Path,
    training_source: tuple[DatasetSource, SourceFile],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, train_file = training_source
    write_bytes = Path.write_bytes

    def fail_on_exclusions(path: Path, data: bytes) -> int:
        if path.name == "excluded.jsonl":
            raise OSError("simulated disk failure")
        return write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_on_exclusions)
    with pytest.raises(PreparationError, match="simulated disk failure"):
        prepare_training_data(
            dataset, train_file, LABELS, tmp_path / "raw", tmp_path / "interim"
        )
    assert list((tmp_path / "interim" / dataset.revision).iterdir()) == []
