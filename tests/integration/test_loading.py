"""Verify the saved artifact consumer independently from source access."""

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from filing_sentence_classifier.data.loading import DataLoadError, SplitName, load_split


def read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def replace_rows(artifact: Path, name: str, rows: list[dict[str, object]]) -> None:
    content = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (artifact / name).write_bytes(content)
    manifest = json.loads((artifact / "manifest.json").read_text())
    manifest["files"][name] = {
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("name", [SplitName.TRAIN, SplitName.VAL])
def test_loading_preserves_records_and_reads_only_the_requested_partition(
    development_artifact: Path, monkeypatch: pytest.MonkeyPatch, name: SplitName
) -> None:
    expected = read_rows(development_artifact / f"{name}.jsonl")
    manifest_hash = hashlib.sha256(
        (development_artifact / "manifest.json").read_bytes()
    ).hexdigest()
    original_read = Path.read_bytes
    reads = []

    def bounded_read(path: Path) -> bytes:
        assert path.parent == development_artifact
        assert path.name in {
            "manifest.json",
            "assignments.jsonl",
            "excluded.jsonl",
            f"{name}.jsonl",
        }
        reads.append(path.name)
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", bounded_read)
    result = load_split(
        development_artifact, name, expected_manifest_sha256=manifest_hash
    )
    assert set(reads) == {
        "manifest.json",
        "assignments.jsonl",
        "excluded.jsonl",
        f"{name}.jsonl",
    }
    assert result.name == name
    assert result.manifest_sha256 == manifest_hash
    assert result.label_ids == (0, 1, 2)
    assert result.label_names == ("specific", "historical", "generic")
    assert result.sample_ids == tuple(row["sample_id"] for row in expected)
    assert result.texts == tuple(row["text"] for row in expected)
    assert result.targets == tuple(row["label"] for row in expected)
    assert [row.source_row for row in result.records] == [
        row["source_row"] for row in expected
    ]
    with pytest.raises(FrozenInstanceError):
        result.records[0].label = 9


def test_other_partition_text_is_not_required(development_artifact: Path) -> None:
    (development_artifact / "train.jsonl").unlink()
    assert load_split(development_artifact, "val").records


@pytest.mark.parametrize("name", ["val.jsonl", "assignments.jsonl", "excluded.jsonl"])
def test_corrupted_files_are_rejected(development_artifact: Path, name: str) -> None:
    (development_artifact / name).write_bytes(b"corrupted")
    with pytest.raises(DataLoadError, match="Integrity check failed"):
        load_split(development_artifact, "val")


def test_reserved_split_and_wrong_manifest_are_rejected(
    development_artifact: Path,
) -> None:
    with pytest.raises(DataLoadError, match="Only saved train and val"):
        load_split(Path("nonexistent"), "test")
    with pytest.raises(DataLoadError, match="Manifest checksum"):
        load_split(development_artifact, "val", expected_manifest_sha256="0" * 64)


@pytest.mark.parametrize(
    "change", ["duplicate_id", "crossed_group", "excluded_group", "wrong_split"]
)
def test_inconsistent_ledgers_are_rejected_even_with_valid_checksums(
    development_artifact: Path, change: str
) -> None:
    rows = read_rows(development_artifact / "assignments.jsonl")
    train = next(row for row in rows if row["split"] == "train")
    val = next(row for row in rows if row["split"] == "val")
    if change == "duplicate_id":
        rows.append(dict(rows[0]))
    elif change == "crossed_group":
        val["group_id"] = train["group_id"]
    elif change == "wrong_split":
        val["split"] = "test"
    else:
        excluded = read_rows(development_artifact / "excluded.jsonl")
        excluded[0]["group_id"] = train["group_id"]
        replace_rows(development_artifact, "excluded.jsonl", excluded)
    replace_rows(development_artifact, "assignments.jsonl", rows)
    with pytest.raises(DataLoadError):
        load_split(development_artifact, "val")


@pytest.mark.parametrize("change", ["text", "label", "source_id", "order"])
def test_inconsistent_records_are_rejected_even_with_valid_checksums(
    development_artifact: Path, change: str
) -> None:
    rows = read_rows(development_artifact / "val.jsonl")
    if change == "text":
        rows[0]["text"] = "Tampered text."
    elif change == "label":
        rows[0]["label"] = True
    elif change == "source_id":
        rows[0]["sample_id"] = "f" * 64
    else:
        rows.reverse()
    replace_rows(development_artifact, "val.jsonl", rows)
    with pytest.raises(DataLoadError):
        load_split(development_artifact, "val")


@pytest.mark.parametrize("change", ["stage", "counts", "class_counts", "label_names"])
def test_manifest_contract_is_checked(development_artifact: Path, change: str) -> None:
    path = development_artifact / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == "stage":
        manifest["stage"] = "unprepared"
    elif change == "counts":
        manifest["counts"]["val"]["rows"] += 1
    elif change == "class_counts":
        manifest["counts"]["val"]["by_label"]["0"] += 1
    else:
        manifest["label_names"]["01"] = "ambiguous label ID"
    path.write_text(json.dumps(manifest))
    with pytest.raises(DataLoadError):
        load_split(development_artifact, "val")
