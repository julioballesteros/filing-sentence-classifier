"""Verify inputs and persist development assignments with a text-only overlap check."""

import hashlib
import json
import platform
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from filing_sentence_classifier.data.cleaning import (
    C1_REPLACEMENTS,
    clean_text,
    cleaning_recipe,
)
from filing_sentence_classifier.data.spec import DatasetSource, SourceFile
from filing_sentence_classifier.data.split import (
    SPLIT_VERSION,
    PreparedRecord,
    SplitError,
    comparison_key,
    split_recipe,
    split_records,
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _jsonl(records: Sequence[object]) -> bytes:
    return "".join(_json(record) + "\n" for record in records).encode("utf-8")


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SplitError("Expected a JSON object in the prepared artifact.")
    return cast(dict[str, object], value)


def _read_regular(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise SplitError(f"Missing regular input file: {path}")
    return path.read_bytes()


def _read_verified(path: Path, metadata: Mapping[str, object]) -> bytes:
    content = _read_regular(path)
    if len(content) != metadata.get("size_bytes") or _sha256(content) != metadata.get(
        "sha256"
    ):
        raise SplitError(f"Input integrity check failed: {path}")
    return content


def _read_prepared(
    directory: Path, source: Mapping[str, object], label_names: Mapping[int, str]
) -> tuple[dict[str, object], bytes, tuple[PreparedRecord, ...]]:
    manifest_bytes = _read_regular(directory / "manifest.json")
    manifest = _object(json.loads(manifest_bytes))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("stage") != "cleaned_published_train"
    ):
        raise SplitError("Expected a version-1 cleaned training artifact.")
    if manifest.get("source") != source:
        raise SplitError("Prepared source does not match the pinned training file.")
    if manifest.get("label_names") != {
        str(label): name for label, name in label_names.items()
    }:
        raise SplitError("Prepared label mapping does not match the selected dataset.")
    if manifest.get("recipe") != cleaning_recipe() or _object(
        manifest.get("code_sha256")
    ).get("cleaning.py") != _sha256(
        Path(__file__).with_name("cleaning.py").read_bytes()
    ):
        raise SplitError(
            "Prepared cleaning rules differ from the frozen overlap rules."
        )
    files = _object(manifest.get("files"))
    if set(files) != {"records.jsonl", "excluded.jsonl", "changes.jsonl"}:
        raise SplitError("Prepared manifest has an unexpected file set.")
    payloads = {
        name: _read_verified(directory / name, _object(metadata))
        for name, metadata in files.items()
    }
    records = []
    for line in payloads["records.jsonl"].splitlines():
        row = _object(json.loads(line))
        if set(row) != {
            "sample_id",
            "source_row",
            "group_id",
            "text",
            "label",
            "label_text",
        }:
            raise SplitError("Unexpected prepared record fields.")
        if (
            type(row["source_row"]) is not int
            or row["source_row"] < 0
            or type(row["label"]) is not int
            or row["label"] not in label_names
            or row["label_text"] != label_names[row["label"]]
            or not isinstance(row["text"], str)
            or not row["text"]
        ):
            raise SplitError("Prepared record violates the field or label contract.")
        try:
            cleaned = clean_text(row["text"])
        except ValueError as exc:
            raise SplitError("Prepared record contains unresolved Unicode.") from exc
        if cleaned.operations:
            raise SplitError("Prepared text is not in the frozen cleaned form.")
        if row["group_id"] != _sha256(row["text"].encode("utf-8")):
            raise SplitError("Prepared group ID does not match its cleaned text.")
        identity = {"source": source, "source_row": row["source_row"]}
        if row["sample_id"] != _sha256(_json(identity).encode("utf-8")):
            raise SplitError("Prepared sample ID does not match its source identity.")
        records.append(cast(PreparedRecord, row))
    if _object(manifest.get("counts")).get("kept") != len(records):
        raise SplitError("Prepared row count does not match its manifest.")
    return manifest, manifest_bytes, tuple(records)


def _reserved_text_groups(content: bytes) -> tuple[set[str], int, int]:
    """Read only the reserved text column and return hashes plus an aggregate count.

    Never return/log test text, row identities, or labels. Empty or non-string
    text stops the operation. Unknown Unicode receives no new encoding repairs;
    count affected rows and retain all texts in the comparison and published test.
    """
    try:
        table = pq.read_table(pa.BufferReader(content), columns=["text"])
    except (pa.ArrowException, ValueError):
        raise SplitError(
            "Could not read the reserved text column; no split was published."
        ) from None
    keys = set()
    warning_rows = 0
    for text in table.column("text").to_pylist():
        if not isinstance(text, str) or not text.strip():
            raise SplitError(
                "Reserved text violates the frozen text contract; no split was published."
            )
        warning_rows += any(
            ("\u0080" <= char <= "\u009f" and char not in C1_REPLACEMENTS)
            or char == "\ufffd"
            for char in text
        )
        keys.add(comparison_key(text))
    if table.num_rows == 0:
        raise SplitError("The reserved source is empty; no split was published.")
    return keys, table.num_rows, warning_rows


def _publish(destination: Path, payloads: Mapping[str, bytes]) -> Path:
    if destination.is_symlink():
        raise SplitError("The split destination must not be a symlink.")
    if destination.exists():
        if not destination.is_dir() or {
            path.name for path in destination.iterdir()
        } != set(payloads):
            raise SplitError(
                "Existing split artifact differs; left unchanged. Choose another --output-dir."
            )
        for name, content in payloads.items():
            path = destination / name
            if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
                raise SplitError(
                    "Existing split artifact differs; left unchanged. Choose another --output-dir."
                )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".split-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        for name, content in payloads.items():
            (staging / name).write_bytes(content)
        staging.rename(destination)
    return destination


def create_development_split(
    dataset: DatasetSource,
    train_file: SourceFile,
    test_file: SourceFile,
    label_names: Mapping[int, str],
    prepared_dir: Path,
    raw_dir: Path = Path("data/raw"),
    output_dir: Path = Path("data/processed"),
) -> Path:
    """Materialize verified train/val records, assignments, and overlap exclusions."""
    try:
        return _create_split(
            dataset,
            train_file,
            test_file,
            label_names,
            prepared_dir,
            raw_dir,
            output_dir,
        )
    except (OSError, ValueError) as exc:
        raise SplitError(
            "Could not read or store split inputs/artifacts; check their format and integrity."
        ) from exc


def _create_split(
    dataset: DatasetSource,
    train_file: SourceFile,
    test_file: SourceFile,
    label_names: Mapping[int, str],
    prepared_dir: Path,
    raw_dir: Path,
    output_dir: Path,
) -> Path:
    if (
        train_file not in dataset.files
        or test_file not in dataset.files
        or train_file == test_file
    ):
        raise SplitError(
            "Select distinct training and test files from the pinned source."
        )
    prepared_dir = prepared_dir.expanduser().resolve()
    snapshot = (raw_dir.expanduser().resolve() / dataset.revision).resolve()
    destination = (
        output_dir.expanduser().resolve() / dataset.revision / f"split-v{SPLIT_VERSION}"
    )
    resolved_destination = destination.resolve()
    if any(
        resolved_destination.is_relative_to(parent)
        or parent.is_relative_to(resolved_destination)
        for parent in (prepared_dir, snapshot)
    ):
        raise SplitError("Split outputs must be stored outside both input artifacts.")
    source = {
        "repo_id": dataset.repo_id,
        "revision": dataset.revision,
        "split": "train",
        "file": asdict(train_file),
    }
    parent, parent_bytes, records = _read_prepared(prepared_dir, source, label_names)
    test_bytes = _read_verified(snapshot / test_file.path, asdict(test_file))
    test_groups, test_rows, test_warning_rows = _reserved_text_groups(test_bytes)
    split = split_records(records, test_groups, tuple(label_names))
    overlap_groups = {record["group_id"] for record in split.excluded}
    if {record["group_id"] for record in (*split.train, *split.val)} & test_groups:
        raise SplitError("A reserved text group remains in the development partitions.")

    assignments = sorted(
        [
            {
                "sample_id": record["sample_id"],
                "source_row": record["source_row"],
                "group_id": record["group_id"],
                "split": name,
            }
            for name, rows in (("train", split.train), ("val", split.val))
            for record in rows
        ],
        key=lambda row: cast(int, row["source_row"]),
    )
    exclusions = [
        {
            "sample_id": record["sample_id"],
            "source_row": record["source_row"],
            "group_id": record["group_id"],
            "reason": "published_test_overlap",
        }
        for record in split.excluded
    ]
    payloads = {
        "train.jsonl": _jsonl(split.train),
        "val.jsonl": _jsonl(split.val),
        "assignments.jsonl": _jsonl(assignments),
        "excluded.jsonl": _jsonl(exclusions),
    }
    counts = {
        name: {
            "rows": len(rows),
            "groups": len({row["group_id"] for row in rows}),
            "by_label": {
                str(label): sum(row["label"] == label for row in rows)
                for label in sorted(label_names)
            },
        }
        for name, rows in (("train", split.train), ("val", split.val))
    }
    manifest = {
        "schema_version": 1,
        "stage": "development_splits",
        "source": source,
        "reserved_test_file": asdict(test_file),
        "parent": {"manifest_sha256": _sha256(parent_bytes), "files": parent["files"]},
        "label_names": parent["label_names"],
        "recipe": split_recipe(),
        "cleaning_recipe": parent["recipe"],
        "counts": {
            "prepared": len(records),
            "excluded_test_overlap": len(split.excluded),
            **counts,
        },
        "overlap_check": {
            "columns_read": ["text"],
            "test_rows": test_rows,
            "test_groups": len(test_groups),
            "test_repeated_occurrences": test_rows - len(test_groups),
            "test_text_warning_rows": test_warning_rows,
            "excluded_development_groups": len(overlap_groups),
            "remaining_development_test_groups_shared": 0,
            "train_val_groups_shared": 0,
        },
        "code_sha256": {
            name: _sha256(Path(__file__).with_name(name).read_bytes())
            for name in ("audit.py", "cleaning.py", "split.py", "partition.py")
        },
        "environment": {
            "python": platform.python_version(),
            "unicode": unicodedata.unidata_version,
            **{
                name: version(name)
                for name in (
                    "filing-sentence-classifier",
                    "scikit-learn",
                    "numpy",
                    "scipy",
                    "pyarrow",
                )
            },
        },
        "files": {
            name: {"sha256": _sha256(content), "size_bytes": len(content)}
            for name, content in payloads.items()
        },
    }
    payloads["manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return _publish(destination, payloads)
