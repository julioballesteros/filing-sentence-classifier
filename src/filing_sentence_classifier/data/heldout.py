"""Prepare the complete published test without development eligibility filters."""

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from filing_sentence_classifier.data.cleaning import clean_text, cleaning_recipe
from filing_sentence_classifier.data.loading import SentenceRecord
from filing_sentence_classifier.data.spec import DatasetSource, SourceFile
from filing_sentence_classifier.evaluation.metrics import EvaluationError


def test_preparation_recipe() -> dict[str, object]:
    """An explicit evaluation adapter, declared before reading test labels.

    U+0099 was identified during the earlier text-only overlap check. Its
    Windows-1252 interpretation is the trademark sign. This adapter does not
    modify clean-v1, training data, or the frozen inference bundles.
    """
    return {
        "version": "published-test-v1",
        "before_clean_v1": {"U+0099": "U+2122"},
        "base_cleaning": cleaning_recipe(),
        "invalid_input": "fail_entire_campaign_without_dropping_rows",
        "duplicates_and_conflicting_labels": "retain_all_published_rows",
        "labels": "unchanged",
        "order": "published_source_row",
    }


def jsonl_bytes(rows: Sequence[object]) -> bytes:
    """Serialize records with the same deterministic identity convention as train."""
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")


@dataclass(frozen=True)
class PreparedTest:
    records: tuple[SentenceRecord, ...]
    label_ids: tuple[int, ...]
    changes: tuple[dict[str, object], ...]
    manifest: dict[str, object]
    name: str = "test"

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(row.sample_id for row in self.records)

    @property
    def targets(self) -> tuple[int, ...]:
        return tuple(row.label for row in self.records)

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(row.text for row in self.records)


def prepare_published_test(
    content: bytes,
    dataset: DatasetSource,
    test_file: SourceFile,
    label_names: Mapping[int, str],
    *,
    expected_rows: int,
) -> PreparedTest:
    """Verify and parse exact source bytes, retaining every row and label.

    This is called only after the campaign has verified its frozen selection.
    It deliberately does not call clean_records, deduplicate, or choose labels.
    Original text remains recoverable from the immutable source and row index.
    """
    if (
        test_file not in dataset.files
        or len(content) != test_file.size_bytes
        or hashlib.sha256(content).hexdigest() != test_file.sha256
    ):
        raise EvaluationError("Published test source integrity check failed.")
    if type(expected_rows) is not int or expected_rows < 1:
        raise EvaluationError("Expected a positive published test row count.")
    try:
        table = pq.read_table(pa.BufferReader(content))
    except (pa.ArrowException, ValueError) as exc:
        raise EvaluationError("Could not parse the published test Parquet.") from exc
    if (
        len(table.column_names) != 3
        or set(table.column_names) != {"text", "label", "label_text"}
        or table.num_rows != expected_rows
    ):
        raise EvaluationError(
            "Published test schema or row count differs from selection."
        )
    source = {
        "repo_id": dataset.repo_id,
        "revision": dataset.revision,
        "split": "test",
        "file": asdict(test_file),
    }
    records = []
    changes = []
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(table.to_pylist()):
        if (
            not isinstance(row["text"], str)
            or type(row["label"]) is not int
            or row["label"] not in label_names
            or row["label_text"] != label_names[row["label"]]
        ):
            raise EvaluationError(
                f"Published test row {index} violates the field contract."
            )
        text = row["text"].replace("\u0099", "\u2122")
        try:
            cleaned = clean_text(text)
            cleaned.text.encode("utf-8")
        except (ValueError, UnicodeError) as exc:
            raise EvaluationError(
                f"Unsupported Unicode in published test row {index}."
            ) from exc
        if not cleaned.text:
            raise EvaluationError(
                f"Published test row {index} is empty after cleaning."
            )
        identity = json.dumps(
            {"source": source, "source_row": index}, ensure_ascii=False, sort_keys=True
        )
        sample_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        group_id = hashlib.sha256(cleaned.text.encode("utf-8")).hexdigest()
        records.append(
            SentenceRecord(
                sample_id,
                index,
                group_id,
                cleaned.text,
                row["label"],
                row["label_text"],
            )
        )
        groups[group_id].append(row["label"])
        operations = (
            ("repair_c1_trademark",) if text != row["text"] else ()
        ) + cleaned.operations
        if operations:
            changes.append(
                {"sample_id": sample_id, "source_row": index, "operations": operations}
            )
    manifest = {
        "schema_version": 1,
        "stage": "prepared_published_test",
        "source": source,
        "recipe": test_preparation_recipe(),
        "label_names": {str(key): value for key, value in sorted(label_names.items())},
        "sample_id": "sha256_of_sorted_utf8_json(source,source_row); zero-based source row",
        "counts": {
            "input": expected_rows,
            "kept": len(records),
            "excluded": 0,
            "changed": len(changes),
            "by_label": dict(
                sorted(Counter(str(row.label) for row in records).items())
            ),
            "unique_groups": len(groups),
            "duplicate_groups": sum(len(labels) > 1 for labels in groups.values()),
            "duplicate_extra_rows": len(records) - len(groups),
            "conflicting_label_groups": sum(
                len(set(labels)) > 1 for labels in groups.values()
            ),
        },
    }
    return PreparedTest(
        tuple(records), tuple(sorted(label_names)), tuple(changes), manifest
    )
