"""Materialize traceable cleaned training records from a verified Parquet file."""

import hashlib
import json
import platform
import unicodedata
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from filing_sentence_classifier.data.cleaning import (
    CLEANING_VERSION,
    clean_records,
    cleaning_recipe,
)
from filing_sentence_classifier.data.spec import DatasetSource, SourceFile


class PreparationError(RuntimeError):
    """The training source or prepared artifact could not be verified or stored."""


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def prepare_training_data(
    dataset: DatasetSource,
    train_file: SourceFile,
    label_names: Mapping[int, str],
    raw_dir: Path = Path("data/raw"),
    output_dir: Path = Path("data/interim"),
) -> Path:
    """Write records, exclusions, changes, and a reproducibility manifest.

    Read only the explicitly selected training file, without network access. No
    development splits or test-overlap checks are performed. All outputs preserve
    source order and omit timestamps/absolute paths. Repeated runs compare every
    output byte; an existing artifact is never silently replaced.
    """
    try:
        return _prepare(dataset, train_file, label_names, raw_dir, output_dir)
    except OSError as exc:
        raise PreparationError(
            f"Could not read or store preparation files: {exc}"
        ) from exc


def _prepare(
    dataset: DatasetSource,
    train_file: SourceFile,
    label_names: Mapping[int, str],
    raw_dir: Path,
    output_dir: Path,
) -> Path:
    if train_file not in dataset.files:
        raise PreparationError("The training file is not part of the selected source.")
    snapshot = raw_dir.expanduser().resolve() / dataset.revision
    source_path = snapshot / train_file.path
    destination = (
        output_dir.expanduser().resolve()
        / dataset.revision
        / f"clean-v{CLEANING_VERSION}"
    )
    if destination.is_relative_to(snapshot) or snapshot.is_relative_to(destination):
        raise PreparationError(
            "Prepared data must be stored outside the source snapshot."
        )
    if source_path.is_symlink() or not source_path.is_file():
        raise PreparationError(
            f"Missing regular training file: {source_path}. Run download-data first."
        )
    source_bytes = source_path.read_bytes()
    if (
        len(source_bytes) != train_file.size_bytes
        or _sha256(source_bytes) != train_file.sha256
    ):
        raise PreparationError(f"Training file integrity check failed: {source_path}")

    # Parse the exact bytes just verified, without scanning a directory or test.
    try:
        table = pq.read_table(pa.BufferReader(source_bytes))
    except (pa.ArrowException, ValueError) as exc:
        raise PreparationError(
            f"Could not read the training Parquet file: {exc}"
        ) from exc
    if (
        set(table.column_names) != {"text", "label", "label_text"}
        or len(table.column_names) != 3
    ):
        raise PreparationError(
            "Unexpected training columns; expected text, label, label_text."
        )
    original_records = table.to_pylist()
    result = clean_records(original_records, label_names)
    if not result.records:
        raise PreparationError(
            "Cleaning left no eligible training records; no artifact was published."
        )

    source = {
        "repo_id": dataset.repo_id,
        "revision": dataset.revision,
        "split": "train",
        "file": asdict(train_file),
    }

    def identify(record: dict[str, object]) -> dict[str, object]:
        identity = {"source": source, "source_row": record["source_row"]}
        return {"sample_id": _sha256(_json(identity).encode("utf-8")), **record}

    payloads = {
        name: "".join(
            _json(identify(asdict(record))) + "\n" for record in records
        ).encode("utf-8")
        for name, records in (
            ("records.jsonl", result.records),
            ("excluded.jsonl", result.excluded),
            ("changes.jsonl", result.changes),
        )
    }
    manifest = {
        "schema_version": 1,
        "stage": "cleaned_published_train",
        "source": source,
        "recipe": cleaning_recipe(),
        "label_names": {
            str(label): name for label, name in sorted(label_names.items())
        },
        "sample_id": "sha256_of_sorted_utf8_json(source,source_row); zero-based source row",
        "code_sha256": {
            name: _sha256(Path(__file__).with_name(name).read_bytes())
            for name in ("audit.py", "cleaning.py", "prepare.py")
        },
        "environment": {
            "python": platform.python_version(),
            "unicode": unicodedata.unidata_version,
            "pyarrow": version("pyarrow"),
            "filing-sentence-classifier": version("filing-sentence-classifier"),
        },
        "counts": {
            "input": len(original_records),
            "kept": len(result.records),
            "excluded": len(result.excluded),
            "changed": len(result.changes),
            "kept_groups": len({record.group_id for record in result.records}),
            "kept_by_label": {
                str(label): sum(record.label == label for record in result.records)
                for label in sorted(label_names)
            },
            "exclusions_by_reason": dict(
                Counter(
                    reason for record in result.excluded for reason in record.reasons
                )
            ),
            "changes_by_operation": dict(
                Counter(
                    operation
                    for record in result.changes
                    for operation in record.operations
                )
            ),
        },
        "files": {
            name: {"sha256": _sha256(content), "size_bytes": len(content)}
            for name, content in payloads.items()
        },
    }
    payloads["manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if destination.is_symlink():
        raise PreparationError(
            f"Prepared destination must not be a symlink: {destination}"
        )
    if destination.exists():
        if not destination.is_dir() or {
            path.name for path in destination.iterdir()
        } != set(payloads):
            raise PreparationError(
                f"Existing artifact differs: {destination}. Left unchanged; choose another --output-dir."
            )
        for name, content in payloads.items():
            path = destination / name
            if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
                raise PreparationError(
                    f"Existing artifact differs: {path}. Left unchanged; choose another --output-dir."
                )
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".clean-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        for name, content in payloads.items():
            (staging / name).write_bytes(content)
        staging.rename(destination)
    return destination
