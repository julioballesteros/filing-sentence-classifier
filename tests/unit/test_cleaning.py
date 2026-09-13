"""Verify cleaning decisions and preservation of classification information."""

from copy import deepcopy

import pytest

from filing_sentence_classifier.data.cleaning import clean_records, clean_text

LABELS = {0: "specific", 1: "historical", 2: "generic"}


def record(text: object, label: int = 0) -> dict[str, object]:
    return {"text": text, "label": label, "label_text": LABELS[label]}


def test_text_repairs_are_targeted_ordered_and_idempotent() -> None:
    text = " \u0093Cafe\u0301\u0094\u0092s\u0095 plan \u0096  2027\u0097no\nexpansion. "
    cleaned = clean_text(text)
    assert cleaned.text == "“Café”’s• plan – 2027—no expansion."
    assert cleaned.operations == (
        "repair_c1_punctuation",
        "unicode_nfc",
        "collapse_whitespace",
    )
    repeated = clean_text(cleaned.text)
    assert repeated.text == cleaned.text
    assert repeated.operations == ()


@pytest.mark.parametrize("character", ["\u0081", "\u0085", "\u0091", "\ufffd"])
def test_unknown_unicode_is_not_guessed_or_hidden_by_whitespace(character: str) -> None:
    with pytest.raises(ValueError, match="unresolved Unicode"):
        clean_text(f"Next{character}year.")


def test_clean_text_preserves_meaning_and_long_records() -> None:
    text = "We will NOT expand in 2027; expected costs are $3.4 million. " * 400
    cleaned = clean_text(text.rstrip())
    assert cleaned.text == text.rstrip()
    assert cleaned.operations == ()
    assert clean_text("We will expand in 2028!").text == "We will expand in 2028!"


def test_invalid_records_are_excluded_with_reasons_and_no_coercion() -> None:
    rows = [
        record(None),
        record(" \n"),
        {"text": "Plan", "label": True, "label_text": "specific"},
        {"text": "Plan", "label": 0, "label_text": "historical"},
        record("The \ufffd plan"),
        record("A valid plan"),
    ]
    result = clean_records(rows, LABELS)
    assert [row.source_row for row in result.records] == [5]
    assert [(row.source_row, row.reasons) for row in result.excluded] == [
        (0, ("text:null_value",)),
        (1, ("text:empty_text",)),
        (2, ("label:invalid_type",)),
        (3, ("label_text:label_text_mismatch",)),
        (4, ("text:unresolved_unicode",)),
    ]
    assert result.changes == ()


def test_conflicts_created_by_cleaning_quarantine_all_members_without_voting() -> None:
    rows = [
        record("Company\u0092s  plan", 0),
        record("Company’s plan", 2),
        record("Company’s plan", 2),
        record("Unrelated statement", 1),
    ]
    result = clean_records(rows, LABELS)
    assert [row.source_row for row in result.records] == [3]
    assert [row.source_row for row in result.excluded] == [0, 1, 2]
    assert {row.reasons for row in result.excluded} == {("conflicting_labels",)}
    assert len({row.group_id for row in result.excluded}) == 1
    assert result.excluded[0].group_id is not None
    assert [row.source_row for row in result.changes] == [0]


def test_duplicate_rows_and_labels_are_retained_without_merging_similar_texts() -> None:
    rows = [
        record("We will expand in 2027."),
        record("We will expand in 2027."),
        record("We will not expand in 2027.", 2),
        record("We will expand in 2028."),
        record("we will expand in 2027."),
    ]
    before = deepcopy(rows)
    result = clean_records(iter(rows), LABELS)
    assert rows == before
    assert result == clean_records(rows, LABELS)
    assert [row.source_row for row in result.records] == list(range(5))
    assert [row.label for row in result.records] == [0, 0, 2, 0, 0]
    assert result.records[0].group_id == result.records[1].group_id
    assert len({row.group_id for row in result.records}) == 4
    assert result.excluded == result.changes == ()


def test_empty_input_has_no_results() -> None:
    result = clean_records([], LABELS)
    assert result.records == result.excluded == result.changes == ()
