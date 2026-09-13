"""Reproducible, group-disjoint development splits over cleaned records."""

import hashlib
from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import TypedDict

from sklearn.model_selection import StratifiedGroupKFold  # type: ignore[import-untyped]

from filing_sentence_classifier.data.audit import normalize_duplicate_text
from filing_sentence_classifier.data.cleaning import C1_REPLACEMENTS

SPLIT_VERSION = "1"
SPLIT_SEED = 2026
N_SPLITS = 5
# At most five percentage points away from the target, overall and per class.
MAX_FRACTION_DEVIATION = 0.05


class PreparedRecord(TypedDict):
    sample_id: str
    source_row: int
    text: str
    label: int
    label_text: str
    group_id: str


class SplitError(RuntimeError):
    """Inputs or the fixed partition do not satisfy the splitting contract."""


@dataclass(frozen=True)
class DevelopmentSplit:
    train: tuple[PreparedRecord, ...]
    val: tuple[PreparedRecord, ...]
    excluded: tuple[PreparedRecord, ...]


def split_recipe() -> dict[str, object]:
    return {
        "version": SPLIT_VERSION,
        "algorithm": "sklearn.model_selection.StratifiedGroupKFold",
        "n_splits": N_SPLITS,
        "validation_fold": 0,
        "shuffle": True,
        "seed": SPLIT_SEED,
        "input_order": "ascending_source_row",
        "validation_target": 1 / N_SPLITS,
        "max_fraction_deviation_overall_and_per_class": MAX_FRACTION_DEVIATION,
        "grouping": "exact_cleaned_text_sha256",
        "test_overlap": "exclude_entire_development_group",
        "overlap_key": "known_c1_replacements_then_NFC_and_whitespace_then_sha256_utf8",
        "unresolved_test_unicode": "preserve_without_additional_encoding_repairs_and_report_count",
        "near_duplicates": "not_merged",
    }


def comparison_key(text: str) -> str:
    """Normalize text for comparison without applying training eligibility rules.

    Repair only the six predeclared C1 characters. Other characters receive NFC
    and whitespace normalization only, without guessing an encoding replacement.
    Every nonempty reserved text remains in the overlap check and published test.
    For eligible cleaned training text, this key equals its existing group_id.
    """
    repaired = text.translate(
        {ord(source): target for source, target in C1_REPLACEMENTS.items()}
    )
    return hashlib.sha256(
        normalize_duplicate_text(repaired).encode("utf-8")
    ).hexdigest()


def split_records(
    records: Sequence[PreparedRecord],
    test_group_ids: Collection[str],
    labels: Collection[int],
) -> DevelopmentSplit:
    """Remove test-overlapping groups and take fold zero of the fixed splitter.

    Only precomputed test text hashes enter this function; test labels are never
    inputs. Preserve every eligible occurrence, its ID, label, and cleaned text.
    Sorting makes assignments independent of the caller's iteration order. Do not
    search other seeds/folds if the declared balance checks fail.
    """
    ordered = sorted(records, key=lambda record: record["source_row"])
    if not ordered or not labels:
        raise SplitError("Splitting requires records and an explicit label set.")
    for field in ("sample_id", "source_row"):
        if len({record[field] for record in ordered}) != len(ordered):
            raise SplitError(f"Prepared records contain duplicate {field} values.")
    group_labels: dict[str, set[int]] = defaultdict(set)
    for record in ordered:
        if record["label"] not in labels:
            raise SplitError("Prepared records contain an unexpected label.")
        group_labels[record["group_id"]].add(record["label"])
    if any(len(values) > 1 for values in group_labels.values()):
        raise SplitError("Prepared records contain conflicting labels within a group.")

    forbidden = set(test_group_ids)
    eligible = tuple(
        record for record in ordered if record["group_id"] not in forbidden
    )
    excluded = tuple(record for record in ordered if record["group_id"] in forbidden)
    for label in sorted(labels):
        groups = {record["group_id"] for record in eligible if record["label"] == label}
        if len(groups) < N_SPLITS:
            raise SplitError(
                f"Label {label} needs at least {N_SPLITS} eligible groups; found {len(groups)}."
            )

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED
    )
    train_indices, val_indices = next(
        splitter.split(
            [[0]] * len(eligible),
            [record["label"] for record in eligible],
            [record["group_id"] for record in eligible],
        )
    )
    train = tuple(eligible[index] for index in train_indices)
    val = tuple(eligible[index] for index in val_indices)
    if {record["group_id"] for record in train} & {
        record["group_id"] for record in val
    }:
        raise SplitError("A group crosses the development partitions.")
    assigned = [record["sample_id"] for record in (*train, *val)]
    if len(assigned) != len(eligible) or set(assigned) != {
        record["sample_id"] for record in eligible
    }:
        raise SplitError(
            "Partition assignments do not cover eligible records exactly once."
        )

    fractions = {"overall": len(val) / len(eligible)}
    for label in sorted(labels):
        val_count = sum(record["label"] == label for record in val)
        total = sum(record["label"] == label for record in eligible)
        if val_count == 0 or val_count == total:
            raise SplitError(f"Label {label} is absent from a development partition.")
        fractions[f"label {label}"] = val_count / total
    for name, fraction in fractions.items():
        if abs(fraction - 1 / N_SPLITS) > MAX_FRACTION_DEVIATION + 1e-12:
            raise SplitError(
                f"Fixed fold has unsuitable validation balance for {name}: {fraction:.2%}. "
                "Review the grouping policy; no alternative seed or fold was tried."
            )
    return DevelopmentSplit(train, val, excluded)
