"""Read frozen development partitions without preparation or training dependencies."""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast


class DataLoadError(RuntimeError):
    """A saved partition failed its integrity or record contract."""


class SplitName(StrEnum):
    TRAIN = "train"
    VAL = "val"


@dataclass(frozen=True)
class SentenceRecord:
    sample_id: str
    source_row: int
    group_id: str
    text: str
    label: int
    label_text: str


@dataclass(frozen=True)
class LoadedSplit:
    name: SplitName
    records: tuple[SentenceRecord, ...]
    label_ids: tuple[int, ...]
    label_names: tuple[str, ...]
    manifest_sha256: str
    records_sha256: str

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(record.text for record in self.records)

    @property
    def targets(self) -> tuple[int, ...]:
        return tuple(record.label for record in self.records)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(record.sample_id for record in self.records)


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DataLoadError("Expected a JSON object in the split artifact.")
    return cast(dict[str, object], value)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _is_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise DataLoadError(f"Missing regular artifact file: {path}")
    return path.read_bytes()


def _verified(directory: Path, name: str, files: dict[str, object]) -> bytes:
    metadata = _object(files[name])
    content = _read(directory / name)
    if (
        type(metadata.get("size_bytes")) is not int
        or len(content) != metadata["size_bytes"]
        or _sha256(content) != metadata.get("sha256")
    ):
        raise DataLoadError(f"Integrity check failed: {name}")
    return content


def _identity(
    row: dict[str, object], source: dict[str, object]
) -> tuple[str, int, str]:
    sample_id, source_row, group_id = (
        row.get("sample_id"),
        row.get("source_row"),
        row.get("group_id"),
    )
    if (
        not _is_hash(sample_id)
        or not _is_hash(group_id)
        or type(source_row) is not int
        or source_row < 0
    ):
        raise DataLoadError("Invalid sample identity in the split artifact.")
    identity = json.dumps(
        {"source": source, "source_row": source_row},
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    if sample_id != _sha256(identity.encode("utf-8")):
        raise DataLoadError("Sample ID does not match the source identity.")
    return sample_id, source_row, cast(str, group_id)


def load_split(
    directory: Path,
    split: SplitName | str,
    *,
    expected_manifest_sha256: str | None = None,
) -> LoadedSplit:
    """Load one saved train/val partition, preserving its exact text and row order.

    Verify the selected JSONL, assignments, and exclusion ledger against their
    manifest. Validate identities, class counts, and group separation using the
    ledgers, without reading the other partition's text. Raw data, reserved test,
    parent artifacts, and current preparation code are not required. An optional
    manifest hash pins the precise artifact expected by a training/evaluation run.
    """
    if split not in (SplitName.TRAIN, SplitName.VAL):
        raise DataLoadError("Only saved train and val partitions can be loaded.")
    if expected_manifest_sha256 is not None and not _is_hash(expected_manifest_sha256):
        raise DataLoadError("Expected manifest checksum must be a SHA-256 hex digest.")
    try:
        return _load(
            directory.expanduser().resolve(), SplitName(split), expected_manifest_sha256
        )
    except (OSError, ValueError) as exc:
        raise DataLoadError(f"Could not read the split artifact: {exc}") from exc


def _load(directory: Path, split: SplitName, expected_hash: str | None) -> LoadedSplit:
    manifest_bytes = _read(directory / "manifest.json")
    manifest_hash = _sha256(manifest_bytes)
    if expected_hash is not None and manifest_hash != expected_hash:
        raise DataLoadError("Manifest checksum does not match the requested artifact.")
    manifest = _object(json.loads(manifest_bytes))
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("stage") != "development_splits"
    ):
        raise DataLoadError("Expected a version-1 development split artifact.")
    source = _object(manifest.get("source"))
    if source.get("split") != "train":
        raise DataLoadError(
            "Development records must originate from published training data."
        )
    if _object(manifest.get("recipe")).get("grouping") != "exact_cleaned_text_sha256":
        raise DataLoadError("Unsupported group identity contract.")
    names = _object(manifest.get("label_names"))
    if not names or any(
        not key.isascii()
        or not key.isdecimal()
        or str(int(key)) != key
        or not isinstance(name, str)
        or not name.strip()
        for key, name in names.items()
    ):
        raise DataLoadError("Invalid label mapping in the split manifest.")
    label_ids = tuple(sorted(int(key) for key in names))
    label_names = tuple(cast(str, names[str(label)]) for label in label_ids)
    files = _object(manifest.get("files"))
    if set(files) != {
        "train.jsonl",
        "val.jsonl",
        "assignments.jsonl",
        "excluded.jsonl",
    }:
        raise DataLoadError("Unexpected split artifact file set.")
    counts = _object(manifest.get("counts"))
    assignments = [
        _object(json.loads(line))
        for line in _verified(directory, "assignments.jsonl", files).splitlines()
    ]
    exclusions = [
        _object(json.loads(line))
        for line in _verified(directory, "excluded.jsonl", files).splitlines()
    ]
    by_split: dict[SplitName, dict[str, tuple[int, str]]] = {
        name: {} for name in SplitName
    }
    group_owner: dict[str, str] = {}
    sample_ids: set[str] = set()
    source_rows: set[int] = set()
    for rows, excluded in ((assignments, False), (exclusions, True)):
        for row in rows:
            expected_fields = {
                "sample_id",
                "source_row",
                "group_id",
                "reason" if excluded else "split",
            }
            if set(row) != expected_fields:
                raise DataLoadError("Unexpected assignment or exclusion fields.")
            sample_id, source_row, group_id = _identity(row, source)
            if sample_id in sample_ids or source_row in source_rows:
                raise DataLoadError("Duplicate identity across assignments/exclusions.")
            sample_ids.add(sample_id)
            source_rows.add(source_row)
            if excluded:
                if row["reason"] != "published_test_overlap":
                    raise DataLoadError("Unexpected split exclusion reason.")
                owner = "excluded"
            else:
                if row["split"] not in ("train", "val"):
                    raise DataLoadError("Unexpected development split assignment.")
                owner = row["split"]
                by_split[SplitName(owner)][sample_id] = (source_row, group_id)
            if group_id in group_owner and group_owner[group_id] != owner:
                raise DataLoadError(
                    "A group crosses partitions or overlaps exclusions."
                )
            group_owner[group_id] = owner
    if counts.get("prepared") != len(assignments) + len(exclusions) or counts.get(
        "excluded_test_overlap"
    ) != len(exclusions):
        raise DataLoadError("Assignment/exclusion counts do not match the manifest.")
    for name, index in by_split.items():
        expected = _object(counts.get(name))
        if (
            not index
            or expected.get("rows") != len(index)
            or expected.get("groups") != len({group for _, group in index.values()})
        ):
            raise DataLoadError(f"Assignment counts do not match the {name} manifest.")

    content = _verified(directory, f"{split}.jsonl", files)
    records = []
    observed: dict[str, tuple[int, str]] = {}
    group_labels: dict[str, int] = {}
    previous_row = -1
    for line in content.splitlines():
        row = _object(json.loads(line))
        if set(row) != {
            "sample_id",
            "source_row",
            "group_id",
            "text",
            "label",
            "label_text",
        }:
            raise DataLoadError("Unexpected sentence record fields.")
        sample_id, source_row, group_id = _identity(row, source)
        text, label, label_text = row["text"], row["label"], row["label_text"]
        if (
            not isinstance(text, str)
            or not text.strip()
            or type(label) is not int
            or label not in label_ids
            or label_text != names[str(label)]
        ):
            raise DataLoadError("Sentence record violates the text or label contract.")
        if group_id != _sha256(text.encode("utf-8")):
            raise DataLoadError("Group ID does not match the saved text.")
        if sample_id in observed or source_row <= previous_row:
            raise DataLoadError(
                "Sentence records must have unique IDs and ascending source rows."
            )
        if group_id in group_labels and group_labels[group_id] != label:
            raise DataLoadError("Sentence group contains conflicting labels.")
        previous_row = source_row
        group_labels[group_id] = label
        observed[sample_id] = (source_row, group_id)
        records.append(
            SentenceRecord(
                sample_id, source_row, group_id, text, label, cast(str, label_text)
            )
        )
    if observed != by_split[split]:
        raise DataLoadError(
            "Sentence records do not match the saved split assignments."
        )
    actual_counts = Counter(record.label for record in records)
    if _object(counts.get(split)).get("by_label") != {
        str(label): actual_counts[label] for label in label_ids
    }:
        raise DataLoadError("Class counts do not match the split manifest.")
    return LoadedSplit(
        split, tuple(records), label_ids, label_names, manifest_hash, _sha256(content)
    )
