"""Exercise preparation-to-partition provenance and text-only overlap checks offline."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from filing_sentence_classifier import cli
from filing_sentence_classifier.data import partition
from filing_sentence_classifier.data.prepare import prepare_training_data
from filing_sentence_classifier.data.spec import DatasetSource, SourceFile
from filing_sentence_classifier.data.split import SplitError

LABELS = {0: "specific", 1: "historical", 2: "generic"}
Source = tuple[DatasetSource, SourceFile, SourceFile, Path]


def file_spec(path: Path, name: str) -> SourceFile:
    content = path.read_bytes()
    return SourceFile(name, len(content), hashlib.sha256(content).hexdigest())


def artifact_bytes(path: Path) -> dict[str, bytes]:
    return {file.name: file.read_bytes() for file in path.iterdir()}


def jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def source(tmp_path: Path) -> Source:
    revision = "d" * 40
    snapshot = tmp_path / "data/raw" / revision
    (snapshot / "data").mkdir(parents=True)
    train = [
        {
            "text": f"Company\u0092s plan class {label} item {index}.",
            "label": label,
            "label_text": LABELS[label],
        }
        for label in LABELS
        for index in range(30)
    ]
    train.append(
        {
            "text": " Company’s  plan class 0 item 0. ",
            "label": 0,
            "label_text": LABELS[0],
        }
    )
    train.extend(
        {"text": "Conflicting annotation", "label": label, "label_text": LABELS[label]}
        for label in (0, 2)
    )
    pq.write_table(pa.Table.from_pylist(train), snapshot / "data/train.parquet")
    # Nonsensical labels are intentional: the reserved labels must not be read.
    test = pa.table(
        {
            "text": [
                "Company’s plan class 0 item 0.",
                "\tCompany’s  plan class 0 item 0.\n",
                "Reserved-only example.",
            ],
            "label": ["UNREAD_LABEL"] * 3,
            "label_text": ["UNREAD_LABEL_TEXT"] * 3,
        }
    )
    pq.write_table(test, snapshot / "data/test.parquet")
    train_file = file_spec(snapshot / "data/train.parquet", "data/train.parquet")
    test_file = file_spec(snapshot / "data/test.parquet", "data/test.parquet")
    dataset = DatasetSource("tests/synthetic", revision, (train_file, test_file))
    prepared = prepare_training_data(
        dataset, train_file, LABELS, tmp_path / "data/raw", tmp_path / "data/interim"
    )
    return dataset, train_file, test_file, prepared


def create(tmp_path: Path, source: Source) -> Path:
    dataset, train_file, test_file, prepared = source
    return partition.create_development_split(
        dataset,
        train_file,
        test_file,
        LABELS,
        prepared,
        tmp_path / "data/raw",
        tmp_path / "data/processed",
    )


def test_cli_freezes_assignments_and_reproduces_outputs_with_no_reserved_labels(
    tmp_path: Path, source: Source, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, train_file, test_file, prepared = source
    monkeypatch.setattr(cli, "FLS_SOURCE", dataset)
    monkeypatch.setattr(cli, "FLS_TRAIN_FILE", train_file)
    monkeypatch.setattr(cli, "FLS_TEST_FILE", test_file)
    monkeypatch.setattr(cli, "LABEL_NAMES", LABELS)
    monkeypatch.chdir(tmp_path)
    read_table = pq.read_table
    reads = []

    def text_only(*args: object, **kwargs: object) -> pa.Table:
        assert kwargs.get("columns") == ["text"]
        reads.append(kwargs["columns"])
        return read_table(*args, **kwargs)

    monkeypatch.setattr(pq, "read_table", text_only)
    before = artifact_bytes(prepared)
    runner = CliRunner()
    response = runner.invoke(cli.app, ["split-data"])
    assert response.exit_code == 0, response.output
    assert reads == [["text"]]
    artifact = tmp_path / "data/processed" / dataset.revision / "split-v1"
    output = artifact_bytes(artifact)
    train, val = jsonl(artifact / "train.jsonl"), jsonl(artifact / "val.jsonl")
    excluded = jsonl(artifact / "excluded.jsonl")
    assignments = jsonl(artifact / "assignments.jsonl")
    assert [row["source_row"] for row in excluded] == [0, 90]
    assert {row["reason"] for row in excluded} == {"published_test_overlap"}
    assert len(train) + len(val) == len(assignments) == 89
    assert not {row["group_id"] for row in train} & {row["group_id"] for row in val}
    kept_by_id = {row["sample_id"]: row for row in jsonl(prepared / "records.jsonl")}
    assert {row["sample_id"] for row in (*train, *val, *excluded)} == set(kept_by_id)
    for name, records in (("train", train), ("val", val)):
        assert {row["label"] for row in records} == set(LABELS)
        assert {row["sample_id"] for row in records} == {
            row["sample_id"] for row in assignments if row["split"] == name
        }
        assert all(row == kept_by_id[row["sample_id"]] for row in records)
    manifest = json.loads(output["manifest.json"])
    assert (
        manifest["parent"]["manifest_sha256"]
        == hashlib.sha256(before["manifest.json"]).hexdigest()
    )
    assert manifest["recipe"]["seed"] == 2026
    assert manifest["recipe"]["validation_fold"] == 0
    assert manifest["overlap_check"]["test_rows"] == 3
    assert manifest["overlap_check"]["test_repeated_occurrences"] == 1
    assert manifest["overlap_check"]["excluded_development_groups"] == 1
    for name, metadata in manifest["files"].items():
        assert hashlib.sha256(output[name]).hexdigest() == metadata["sha256"]
        assert len(output[name]) == metadata["size_bytes"]
    assert b"UNREAD_LABEL" not in b"".join(output.values())
    assert b"Reserved-only" not in b"".join(output.values())
    mtimes = {file.name: file.stat().st_mtime_ns for file in artifact.iterdir()}
    assert runner.invoke(cli.app, ["split-data"]).exit_code == 0
    assert artifact_bytes(artifact) == output
    assert mtimes == {file.name: file.stat().st_mtime_ns for file in artifact.iterdir()}
    rebuilt = partition.create_development_split(
        dataset,
        train_file,
        test_file,
        LABELS,
        prepared,
        tmp_path / "data/raw",
        tmp_path / "rebuilt",
    )
    assert artifact_bytes(rebuilt) == output
    assert artifact_bytes(prepared) == before
    for file in (train_file, test_file):
        assert (
            file_spec(tmp_path / "data/raw" / dataset.revision / file.path, file.path)
            == file
        )


@pytest.mark.parametrize("name", ["records.jsonl", "excluded.jsonl", "changes.jsonl"])
def test_corrupt_parent_files_fail_before_reserved_access(
    tmp_path: Path, source: Source, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    source[3].joinpath(name).write_bytes(b"corrupted")

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Reserved text should not be read for invalid preparation inputs.")

    monkeypatch.setattr(partition, "_reserved_text_groups", forbidden)
    with pytest.raises(SplitError, match="integrity check failed"):
        create(tmp_path, source)
    assert not (tmp_path / "data/processed").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("group_id", "wrong"),
        ("sample_id", "wrong"),
        ("label", 99),
        ("text", "Not yet  normalized"),
    ],
)
def test_invalid_record_contract_is_rejected_even_with_updated_checksums(
    tmp_path: Path, source: Source, field: str, value: object
) -> None:
    prepared = source[3]
    rows = jsonl(prepared / "records.jsonl")
    rows[0][field] = value
    content = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (prepared / "records.jsonl").write_bytes(content)
    manifest = json.loads((prepared / "manifest.json").read_text())
    manifest["files"]["records.jsonl"] = {
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SplitError, match="Prepared"):
        create(tmp_path, source)
    assert not (tmp_path / "data/processed").exists()


@pytest.mark.parametrize("change", ["source", "recipe"])
def test_parent_provenance_and_policy_must_match(
    tmp_path: Path, source: Source, change: str
) -> None:
    path = source[3] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[change] = {}
    path.write_text(json.dumps(manifest))
    with pytest.raises(SplitError, match="Prepared"):
        create(tmp_path, source)


def test_corrupt_reserved_file_is_not_parsed(
    tmp_path: Path, source: Source, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, _, test_file, _ = source
    (tmp_path / "data/raw" / dataset.revision / test_file.path).write_bytes(b"corrupt")

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("A corrupt source must not be parsed.")

    monkeypatch.setattr(pq, "read_table", forbidden)
    with pytest.raises(SplitError, match="integrity check failed"):
        create(tmp_path, source)


@pytest.mark.parametrize("text", [None, ""])
def test_invalid_reserved_text_stops_without_exposing_or_dropping_it(
    tmp_path: Path, source: Source, text: str | None
) -> None:
    dataset, train_file, test_file, prepared = source
    path = tmp_path / "data/raw" / dataset.revision / test_file.path
    pq.write_table(pa.table({"text": [text]}), path)
    test_file = file_spec(path, test_file.path)
    dataset = replace(dataset, files=(train_file, test_file))
    with pytest.raises(SplitError, match="frozen text contract") as error:
        create(tmp_path, (dataset, train_file, test_file, prepared))
    assert "Secret" not in str(error.value)
    assert not (tmp_path / "data/processed").exists()


def test_reserved_unicode_warning_is_counted_without_repair_or_row_exclusion(
    tmp_path: Path, source: Source
) -> None:
    dataset, train_file, test_file, prepared = source
    path = tmp_path / "data/raw" / dataset.revision / test_file.path
    pq.write_table(
        pa.table(
            {
                "text": [
                    "Secret\u0099 example",
                    "Secret\u0099 example",
                    "Company’s plan class 0 item 0.",
                ]
            }
        ),
        path,
    )
    test_file = file_spec(path, test_file.path)
    dataset = replace(dataset, files=(train_file, test_file))
    artifact = create(tmp_path, (dataset, train_file, test_file, prepared))
    outputs = artifact_bytes(artifact)
    summary = json.loads(outputs["manifest.json"])["overlap_check"]
    assert summary["test_rows"] == 3
    assert summary["test_groups"] == 2
    assert summary["test_text_warning_rows"] == 2
    assert summary["test_repeated_occurrences"] == 1
    assert b"Secret" not in b"".join(outputs.values())
    assert file_spec(path, test_file.path) == test_file


def test_output_corruption_is_not_overwritten(tmp_path: Path, source: Source) -> None:
    artifact = create(tmp_path, source)
    (artifact / "val.jsonl").write_bytes(b"modified")
    before = artifact_bytes(artifact)
    with pytest.raises(SplitError, match="left unchanged"):
        create(tmp_path, source)
    assert artifact_bytes(artifact) == before


def test_output_cannot_be_nested_under_inputs(tmp_path: Path, source: Source) -> None:
    dataset, train_file, test_file, prepared = source
    for output in (prepared, tmp_path / "data/raw"):
        with pytest.raises(SplitError, match="outside both input"):
            partition.create_development_split(
                dataset,
                train_file,
                test_file,
                LABELS,
                prepared,
                tmp_path / "data/raw",
                output,
            )


def test_failed_write_does_not_publish_partial_splits(
    tmp_path: Path, source: Source, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_bytes = Path.write_bytes

    def fail(path: Path, content: bytes) -> int:
        if path.name == "val.jsonl":
            raise OSError("simulated write failure")
        return write_bytes(path, content)

    monkeypatch.setattr(Path, "write_bytes", fail)
    with pytest.raises(SplitError, match="Could not read or store"):
        create(tmp_path, source)
    assert list((tmp_path / "data/processed" / source[0].revision).iterdir()) == []
