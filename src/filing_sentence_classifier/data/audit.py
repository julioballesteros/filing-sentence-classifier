"""Deterministic, read-only checks for sentence classification records."""

import math
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RowIssue:
    """A finding identified by its zero-based position in the source records."""

    row: int
    field: str
    code: str
    severity: Literal["error", "warning"] = "error"


@dataclass(frozen=True)
class DuplicateGroup:
    """All rows sharing a text key, including the first occurrence."""

    key: str
    rows: tuple[int, ...]
    labels: tuple[int, ...]

    @property
    def has_label_conflict(self) -> bool:
        """Whether the group contains multiple distinct, valid label IDs."""
        return len(self.labels) > 1


@dataclass(frozen=True)
class AuditReport:
    row_count: int
    class_counts: dict[int, int]
    issues: tuple[RowIssue, ...]
    exact_duplicates: tuple[DuplicateGroup, ...]
    normalized_duplicates: tuple[DuplicateGroup, ...]

    @property
    def invalid_rows(self) -> tuple[int, ...]:
        return tuple(
            sorted({issue.row for issue in self.issues if issue.severity == "error"})
        )


def normalize_duplicate_text(text: str) -> str:
    """Apply NFC and collapse whitespace; preserve case, punctuation and numbers.

    This is a duplicate-detection key, not model preprocessing or fuzzy matching.
    """
    return " ".join(unicodedata.normalize("NFC", text).split())


def _is_null(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _duplicate_groups(
    groups: Mapping[str, list[tuple[int, int | None]]],
) -> tuple[DuplicateGroup, ...]:
    duplicates = [
        DuplicateGroup(
            key=key,
            rows=tuple(row for row, _ in members),
            labels=tuple(sorted({label for _, label in members if label is not None})),
        )
        for key, members in groups.items()
        if len(members) > 1
    ]
    return tuple(
        sorted(duplicates, key=lambda group: (-len(group.rows), group.rows[0]))
    )


def audit_records(
    records: Iterable[Mapping[str, object]], label_names: Mapping[int, str]
) -> AuditReport:
    """Inspect records without filtering, relabeling, or changing input data.

    Each record should contain text, label and label_text. Pass Python scalar
    values, representing nulls as None or float NaN. Class counts include every
    valid label ID, even if another field in that row is invalid. Duplicate groups
    include all nonempty string texts; invalid label IDs do not count as conflicts.
    Normalized duplicate groups include the exact duplicate groups.
    """
    class_counts = dict.fromkeys(sorted(label_names), 0)
    issues: list[RowIssue] = []
    exact: dict[str, list[tuple[int, int | None]]] = defaultdict(list)
    normalized: dict[str, list[tuple[int, int | None]]] = defaultdict(list)
    row_count = 0

    for row, record in enumerate(records):
        row_count += 1
        for field in ("text", "label", "label_text"):
            if field not in record:
                issues.append(RowIssue(row, field, "missing_field"))
            elif _is_null(record[field]):
                issues.append(RowIssue(row, field, "null_value"))

        label = record.get("label")
        valid_label: int | None = None
        if label is not None and not _is_null(label):
            # bool and integral floats must not silently become class IDs.
            if type(label) is not int:
                issues.append(RowIssue(row, "label", "invalid_type"))
            elif label not in label_names:
                issues.append(RowIssue(row, "label", "unknown_label"))
            else:
                valid_label = label
                class_counts[label] += 1

        label_text = record.get("label_text")
        if label_text is not None and not _is_null(label_text):
            if not isinstance(label_text, str):
                issues.append(RowIssue(row, "label_text", "invalid_type"))
            elif valid_label is not None and label_text != label_names[valid_label]:
                issues.append(RowIssue(row, "label_text", "label_text_mismatch"))

        text = record.get("text")
        if text is None or _is_null(text):
            continue
        if not isinstance(text, str):
            issues.append(RowIssue(row, "text", "invalid_type"))
            continue
        if not text.strip():
            issues.append(RowIssue(row, "text", "empty_text"))
            continue
        if any("\u0080" <= char <= "\u009f" or char == "\ufffd" for char in text):
            issues.append(RowIssue(row, "text", "suspicious_unicode", "warning"))
        exact[text].append((row, valid_label))
        normalized[normalize_duplicate_text(text)].append((row, valid_label))

    return AuditReport(
        row_count=row_count,
        class_counts=class_counts,
        issues=tuple(issues),
        exact_duplicates=_duplicate_groups(exact),
        normalized_duplicates=_duplicate_groups(normalized),
    )
