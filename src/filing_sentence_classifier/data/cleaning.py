"""Versioned, deterministic cleaning rules for development records."""

import hashlib
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

from filing_sentence_classifier.data.audit import audit_records

CLEANING_VERSION = "1"
# Only the six C1 characters observed in the FLS training audit are repaired.
# Their contexts support these Windows-1252 punctuation interpretations.
C1_REPLACEMENTS = {
    "\u0092": "\u2019",
    "\u0093": "\u201c",
    "\u0094": "\u201d",
    "\u0095": "\u2022",
    "\u0096": "\u2013",
    "\u0097": "\u2014",
}


def cleaning_recipe() -> dict[str, object]:
    """Describe the fixed policy embedded in every preparation manifest."""
    return {
        "version": CLEANING_VERSION,
        "operations_in_order": [
            "repair_c1_punctuation",
            "unicode_nfc",
            "collapse_whitespace",
        ],
        "c1_replacements": {
            f"U+{ord(source):04X}": f"U+{ord(target):04X}"
            for source, target in C1_REPLACEMENTS.items()
        },
        "invalid_records": "exclude_with_reasons",
        "unresolved_unicode": "exclude_other_c1_and_replacement_character",
        "conflicting_labels": "quarantine_entire_cleaned_text_group",
        "same_label_duplicates": "retain_rows_with_shared_group_id",
        "group_id": "sha256_of_cleaned_text_utf8",
        "near_duplicates": "no_automatic_merging",
        "labels": "unchanged",
        "length_filter": None,
    }


@dataclass(frozen=True)
class CleanedText:
    text: str
    operations: tuple[str, ...]


@dataclass(frozen=True)
class CleanedRecord:
    source_row: int
    text: str
    label: int
    label_text: str
    group_id: str


@dataclass(frozen=True)
class ExcludedRecord:
    source_row: int
    reasons: tuple[str, ...]
    group_id: str | None = None


@dataclass(frozen=True)
class ChangedRecord:
    source_row: int
    operations: tuple[str, ...]


@dataclass(frozen=True)
class CleaningResult:
    records: tuple[CleanedRecord, ...]
    excluded: tuple[ExcludedRecord, ...]
    changes: tuple[ChangedRecord, ...]


def clean_text(text: str) -> CleanedText:
    """Clean text without using labels or fitting any parameters.

    Preserve case, punctuation, numbers, negation, and word order. Reject unknown
    C1 characters and U+FFFD before whitespace collapsing can hide them. This
    label-independent function can also be reused at inference time.
    """
    if any(
        ("\u0080" <= char <= "\u009f" and char not in C1_REPLACEMENTS)
        or char == "\ufffd"
        for char in text
    ):
        raise ValueError("Text contains unresolved Unicode characters.")
    operations: list[str] = []
    repaired = text.translate(
        {ord(source): target for source, target in C1_REPLACEMENTS.items()}
    )
    if repaired != text:
        operations.append("repair_c1_punctuation")
    normalized = unicodedata.normalize("NFC", repaired)
    if normalized != repaired:
        operations.append("unicode_nfc")
    collapsed = " ".join(normalized.split())
    if collapsed != normalized:
        operations.append("collapse_whitespace")
    return CleanedText(collapsed, tuple(operations))


def clean_records(
    records: Iterable[Mapping[str, object]], label_names: Mapping[int, str]
) -> CleaningResult:
    """Clean published training records, retaining their original row positions.

    Invalid fields are never coerced. After cleaning, quarantine every member of
    any text group with conflicting valid label IDs; never select a label by vote.
    Changes include transformed rows subsequently quarantined for a label conflict.
    Source mappings are left unchanged. Nulls follow audit_records' scalar contract.
    """
    rows = list(records)
    audit = audit_records(rows, label_names)
    invalid: dict[int, list[str]] = defaultdict(list)
    for issue in audit.issues:
        if issue.severity == "error":
            invalid[issue.row].append(f"{issue.field}:{issue.code}")

    candidates: list[CleanedRecord] = []
    excluded: list[ExcludedRecord] = []
    changes: list[ChangedRecord] = []
    group_labels: dict[str, set[int]] = defaultdict(set)
    for source_row, row in enumerate(rows):
        if source_row in invalid:
            excluded.append(
                ExcludedRecord(source_row, tuple(sorted(invalid[source_row])))
            )
            continue
        # The audit has validated these scalar types and the label mapping.
        try:
            cleaned = clean_text(cast(str, row["text"]))
        except ValueError:
            excluded.append(ExcludedRecord(source_row, ("text:unresolved_unicode",)))
            continue
        if not cleaned.text:
            excluded.append(ExcludedRecord(source_row, ("text:empty_after_cleaning",)))
            continue
        label = cast(int, row["label"])
        group_id = hashlib.sha256(cleaned.text.encode("utf-8")).hexdigest()
        candidates.append(
            CleanedRecord(
                source_row, cleaned.text, label, cast(str, row["label_text"]), group_id
            )
        )
        group_labels[group_id].add(label)
        if cleaned.operations:
            changes.append(ChangedRecord(source_row, cleaned.operations))

    kept: list[CleanedRecord] = []
    for candidate in candidates:
        if len(group_labels[candidate.group_id]) > 1:
            excluded.append(
                ExcludedRecord(
                    candidate.source_row, ("conflicting_labels",), candidate.group_id
                )
            )
        else:
            kept.append(candidate)
    return CleaningResult(
        tuple(kept),
        tuple(sorted(excluded, key=lambda record: record.source_row)),
        tuple(changes),
    )
