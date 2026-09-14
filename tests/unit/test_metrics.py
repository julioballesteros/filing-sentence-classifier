"""Verify metric definitions against hand-calculated classification examples."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from filing_sentence_classifier.evaluation.metrics import (
    EvaluationError,
    classification_metrics,
)


def test_metrics_match_manual_values_and_confusion_matrix_orientation() -> None:
    result = classification_metrics(
        [0, 0, 0, 1, 1, 2], [0, 1, 1, 1, 2, 2], label_ids=[0, 1, 2]
    )
    assert result.sample_count == 6
    assert result.accuracy == 0.5
    assert result.macro_f1 == pytest.approx(47 / 90)
    assert result.confusion_matrix == ((1, 2, 0), (0, 1, 1), (0, 0, 1))
    assert [item.support for item in result.per_class] == [3, 2, 1]
    assert [item.precision for item in result.per_class] == pytest.approx(
        [1, 1 / 3, 1 / 2]
    )
    assert [item.recall for item in result.per_class] == pytest.approx(
        [1 / 3, 1 / 2, 1]
    )
    assert [item.f1 for item in result.per_class] == pytest.approx(
        [1 / 2, 2 / 5, 2 / 3]
    )
    json.dumps(asdict(result), allow_nan=False)


def test_absent_classes_stay_in_macro_average_and_have_zero_metrics() -> None:
    result = classification_metrics([0, 0], [0, 0], label_ids=[0, 1, 2])
    assert result.accuracy == 1.0
    assert result.macro_f1 == pytest.approx(1 / 3)
    assert result.confusion_matrix == ((2, 0, 0), (0, 0, 0), (0, 0, 0))
    for item in result.per_class[1:]:
        assert (item.precision, item.recall, item.f1, item.support) == (0, 0, 0, 0)


def test_declared_label_order_is_preserved() -> None:
    result = classification_metrics(
        [0, 0, 0, 1, 1, 2], [0, 1, 1, 1, 2, 2], label_ids=[2, 0, 1]
    )
    assert result.label_ids == (2, 0, 1)
    assert [item.label for item in result.per_class] == [2, 0, 1]
    assert result.confusion_matrix == ((1, 0, 0), (0, 1, 2), (1, 0, 1))


@pytest.mark.parametrize(
    "truth,predictions,labels",
    [
        ([], [], [0, 1, 2]),
        ([0, 1], [0], [0, 1, 2]),
        ([0], [9], [0, 1, 2]),
        ([9], [0], [0, 1, 2]),
        ([0], [0], []),
        ([0], [0], [0, 0]),
        ([0], [True], [0, 1, 2]),
        ([0], [0.0], [0, 1, 2]),
        ([0], ["0"], [0, 1, 2]),
        ([0], [[0]], [0, 1, 2]),
        ([0], [float("nan")], [0, 1, 2]),
    ],
)
def test_invalid_vectors_are_not_truncated_coerced_or_ignored(
    truth: list[int], predictions: list[int], labels: list[int]
) -> None:
    with pytest.raises(EvaluationError):
        classification_metrics(truth, predictions, label_ids=labels)


def test_numpy_integer_outputs_and_iterators_are_supported() -> None:
    result = classification_metrics(
        np.array([0, 1, 2], dtype=np.int64),
        iter(np.array([0, 1, 2], dtype=np.int32)),
        label_ids=range(3),
    )
    assert result.accuracy == result.macro_f1 == 1.0
    assert type(result.sample_count) is int
    assert all(type(item.support) is int for item in result.per_class)
    json.dumps(asdict(result), allow_nan=False)


def test_metrics_are_computed_over_the_full_split_not_averaged_per_batch() -> None:
    complete = classification_metrics([0, 1], [0, 0], label_ids=[0, 1])
    batches = [
        classification_metrics([target], [0], label_ids=[0, 1]).macro_f1
        for target in (0, 1)
    ]
    assert complete.macro_f1 == pytest.approx(1 / 3)
    assert complete.macro_f1 != pytest.approx(sum(batches) / 2)
