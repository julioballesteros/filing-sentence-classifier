"""Check grouped stratification, exclusions, and deterministic row assignments."""

from copy import deepcopy

import pytest

from filing_sentence_classifier.data.cleaning import clean_text
from filing_sentence_classifier.data.split import (
    PreparedRecord,
    SplitError,
    comparison_key,
    split_records,
)

LABELS = (0, 1, 2)


def test_comparison_key_uses_frozen_repairs_without_training_exclusion_rules() -> None:
    import hashlib

    assert comparison_key(" Company\u0092s  plan. ") == comparison_key(
        "Company’s plan."
    )
    original = "Company\u0099 will not expand in 2027."
    assert comparison_key(original) == hashlib.sha256(original.encode()).hexdigest()
    assert comparison_key(original) != comparison_key(original.replace("\u0099", "™"))
    assert comparison_key(original) != comparison_key(original.replace("not ", ""))
    with pytest.raises(ValueError, match="unresolved Unicode"):
        clean_text(original)


def record(source_row: int, label: int, group: str) -> PreparedRecord:
    return {
        "sample_id": f"sample-{source_row}",
        "source_row": source_row,
        "group_id": group,
        "label": label,
        "label_text": f"class-{label}",
        "text": f"Text for {group}.",
    }


@pytest.fixture
def rows() -> list[PreparedRecord]:
    return [
        record(label * 100 + index, label, f"{label}-{index}")
        for label in LABELS
        for index in range(30)
    ]


def test_split_is_balanced_disjoint_complete_and_independent_of_input_order(
    rows: list[PreparedRecord],
) -> None:
    rows.extend(record(300 + label, label, f"{label}-0") for label in LABELS)
    original = deepcopy(rows)
    result = split_records(rows, set(), LABELS)
    assert rows == original
    assert result == split_records(list(reversed(rows)), set(), LABELS)
    assert result == split_records(rows, set(), LABELS)
    assert result.excluded == ()
    assert not {row["group_id"] for row in result.train} & {
        row["group_id"] for row in result.val
    }
    all_rows = sorted((*result.train, *result.val), key=lambda row: row["source_row"])
    assert all_rows == sorted(rows, key=lambda row: row["source_row"])
    for label in LABELS:
        assert 5 <= sum(row["label"] == label for row in result.val) <= 7
    for partition in (result.train, result.val):
        assert [row["source_row"] for row in partition] == sorted(
            row["source_row"] for row in partition
        )


def test_overlap_excludes_the_entire_group_and_keeps_nonmatches(
    rows: list[PreparedRecord],
) -> None:
    rows.append(record(300, 0, "0-0"))
    result = split_records(rows, {"0-0", "reserved-only"}, LABELS)
    assert [row["source_row"] for row in result.excluded] == [0, 300]
    assert len(result.train) + len(result.val) == len(rows) - 2
    assert all(row["group_id"] != "0-0" for row in (*result.train, *result.val))


@pytest.mark.parametrize("field", ["sample_id", "source_row"])
def test_duplicate_identities_are_rejected(
    rows: list[PreparedRecord], field: str
) -> None:
    corrupted = dict(rows[1])
    corrupted[field] = rows[0][field]
    rows[1] = corrupted
    with pytest.raises(SplitError, match="duplicate"):
        split_records(rows, set(), LABELS)


def test_conflicting_labels_are_not_hidden_by_test_exclusion(
    rows: list[PreparedRecord],
) -> None:
    rows.append(record(300, 2, "0-0"))
    with pytest.raises(SplitError, match="conflicting labels"):
        split_records(rows, {"0-0"}, LABELS)


def test_class_support_is_checked_after_overlap_removal(
    rows: list[PreparedRecord],
) -> None:
    with pytest.raises(
        SplitError, match="Label 0 needs at least 5 eligible groups; found 4"
    ):
        split_records(rows, {f"0-{index}" for index in range(26)}, LABELS)
    with pytest.raises(SplitError, match="Label 2 needs"):
        split_records([row for row in rows if row["label"] != 2], set(), LABELS)


def test_dominant_groups_fail_balance_checks_instead_of_being_split() -> None:
    rows = [
        record(
            label * 100 + index,
            label,
            f"{label}-large" if index < 95 else f"{label}-{index}",
        )
        for label in LABELS
        for index in range(100)
    ]
    with pytest.raises(SplitError, match="balance|absent"):
        split_records(rows, set(), LABELS)


def test_empty_input_and_unexpected_labels_are_rejected(
    rows: list[PreparedRecord],
) -> None:
    with pytest.raises(SplitError, match="requires records"):
        split_records([], set(), LABELS)
    with pytest.raises(SplitError, match="unexpected label"):
        split_records(rows, set(), (0, 1))
