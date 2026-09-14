"""Classification metrics with an explicit class order and zero-division policy."""

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral

from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


class EvaluationError(ValueError):
    """Predictions do not satisfy the evaluation contract."""


@dataclass(frozen=True)
class ClassMetrics:
    label: int
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ClassificationMetrics:
    label_ids: tuple[int, ...]
    sample_count: int
    accuracy: float
    macro_f1: float
    per_class: tuple[ClassMetrics, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]


def _integers(values: Iterable[int], field: str) -> tuple[int, ...]:
    result = tuple(values)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in result
    ):
        raise EvaluationError(
            f"{field} must contain integer class IDs, not scores or coerced labels."
        )
    # Accept NumPy integer outputs while returning ordinary Python JSON scalars.
    return tuple(int(value) for value in result)


def classification_metrics(
    y_true: Iterable[int], y_pred: Iterable[int], *, label_ids: Iterable[int]
) -> ClassificationMetrics:
    """Evaluate complete prediction vectors in the explicitly declared class order.

    Include every declared class in macro-F1, even if absent from truth or
    predictions. Undefined precision/recall/F1 are zero (zero_division=0).
    Confusion-matrix rows are true classes and columns predicted classes.
    Reject empty input, unknown labels, and unequal lengths; never truncate pairs.
    Compute once over the complete split, rather than averaging batch F1 scores.
    """
    labels = _integers(label_ids, "label_ids")
    if not labels or len(set(labels)) != len(labels):
        raise EvaluationError("label_ids must be nonempty and unique.")
    truth, predicted = _integers(y_true, "y_true"), _integers(y_pred, "y_pred")
    if not truth or len(truth) != len(predicted):
        raise EvaluationError(
            "Truth and predictions must have the same nonzero length."
        )
    if (set(truth) | set(predicted)) - set(labels):
        raise EvaluationError("Truth or predictions contain undeclared class IDs.")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, average=None, zero_division=0
    )
    per_class = tuple(
        ClassMetrics(label, float(p), float(r), float(f), int(n))
        for label, p, r, f, n in zip(
            labels, precision, recall, f1, support, strict=True
        )
    )
    matrix = confusion_matrix(truth, predicted, labels=labels)
    return ClassificationMetrics(
        label_ids=labels,
        sample_count=len(truth),
        accuracy=float(accuracy_score(truth, predicted)),
        macro_f1=sum(item.f1 for item in per_class) / len(labels),
        per_class=per_class,
        confusion_matrix=tuple(tuple(int(count) for count in row) for row in matrix),
    )
