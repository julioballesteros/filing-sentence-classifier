"""Check audit behavior with synthetic records and no external data access."""

from copy import deepcopy

import pytest

from filing_sentence_classifier.data.audit import (
    audit_records,
    normalize_duplicate_text,
)

LABELS = {0: "specific", 1: "historical", 2: "generic"}


def record(
    text: object, label: object = 0, label_text: object = "specific"
) -> dict[str, object]:
    return {"text": text, "label": label, "label_text": label_text}


def test_empty_input_and_missing_classes() -> None:
    empty = audit_records([], LABELS)
    assert empty.row_count == 0
    assert empty.class_counts == {0: 0, 1: 0, 2: 0}
    assert empty.issues == empty.exact_duplicates == empty.normalized_duplicates == ()

    report = audit_records([record("We will expand.")], LABELS)
    assert report.row_count == 1
    assert report.class_counts == {0: 1, 1: 0, 2: 0}
    assert report.invalid_rows == ()


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({}, {(field, "missing_field") for field in ("text", "label", "label_text")}),
        (record(None), {("text", "null_value")}),
        (record(float("nan")), {("text", "null_value")}),
        (record(12), {("text", "invalid_type")}),
        (record(" \t\n"), {("text", "empty_text")}),
        (record("Plan", None), {("label", "null_value")}),
        (record("Plan", True), {("label", "invalid_type")}),
        (record("Plan", 0.0), {("label", "invalid_type")}),
        (record("Plan", "0"), {("label", "invalid_type")}),
        (record("Plan", 7), {("label", "unknown_label")}),
        (record("Plan", 0, None), {("label_text", "null_value")}),
        (record("Plan", 0, 0), {("label_text", "invalid_type")}),
        (record("Plan", 0, "historical"), {("label_text", "label_text_mismatch")}),
    ],
)
def test_invalid_records_are_reported_without_coercion(
    row: dict[str, object], expected: set[tuple[str, str]]
) -> None:
    report = audit_records([row], LABELS)
    assert {(issue.field, issue.code) for issue in report.issues} == expected
    assert report.invalid_rows == (0,)


def test_unicode_findings_are_warnings_not_automatic_exclusions() -> None:
    report = audit_records(
        [record("The company\u0092s plan."), record("Loss of \ufffd text.")], LABELS
    )
    assert len(report.issues) == 2
    assert {issue.code for issue in report.issues} == {"suspicious_unicode"}
    assert {issue.severity for issue in report.issues} == {"warning"}
    assert report.invalid_rows == ()


def test_duplicates_preserve_row_positions_and_distinguish_label_conflicts() -> None:
    rows = [
        record("We will expand."),
        record("We will expand."),
        record(" We  will\texpand. ", 2, "generic"),
        record("Other text", 1, "historical"),
        record("Other text", 1, "historical"),
    ]
    before = deepcopy(rows)
    report = audit_records(iter(rows), LABELS)

    assert rows == before
    assert report == audit_records(rows, LABELS)
    assert report.class_counts == {0: 2, 1: 2, 2: 1}
    assert [group.rows for group in report.exact_duplicates] == [(0, 1), (3, 4)]
    assert not any(group.has_label_conflict for group in report.exact_duplicates)
    assert report.normalized_duplicates[0].rows == (0, 1, 2)
    assert report.normalized_duplicates[0].labels == (0, 2)
    assert report.normalized_duplicates[0].has_label_conflict


def test_missing_texts_are_not_duplicates_and_bad_labels_are_not_conflicts() -> None:
    report = audit_records(
        [
            record(None),
            record(None),
            record("Plan"),
            record("Plan", "0"),
            record("Plan", 99),
        ],
        LABELS,
    )
    assert len(report.exact_duplicates) == 1
    group = report.exact_duplicates[0]
    assert group.rows == (2, 3, 4)
    assert group.labels == (0,)
    assert not group.has_label_conflict
    assert report.invalid_rows == (0, 1, 3, 4)


def test_normalization_is_conservative() -> None:
    assert normalize_duplicate_text(" Cafe\u0301\n plans\t2027. ") == "Café plans 2027."
    original = "We will not expand in 2027."
    assert normalize_duplicate_text(original) == original
    variants = [
        "we will not expand in 2027.",
        "We will expand in 2027.",
        "We will not expand in 2028.",
        "We will not expand in 2027!",
    ]
    assert all(normalize_duplicate_text(text) != original for text in variants)
