"""Exercise model-independent prediction identity alignment."""

import pytest

from filing_sentence_classifier.data.loading import (
    LoadedSplit,
    SentenceRecord,
    SplitName,
)
from filing_sentence_classifier.evaluation.evaluate import (
    Prediction,
    evaluate_predictions,
)
from filing_sentence_classifier.evaluation.metrics import EvaluationError


@pytest.fixture
def validation() -> LoadedSplit:
    return LoadedSplit(
        SplitName.VAL,
        tuple(
            SentenceRecord(
                f"id-{label}",
                label,
                f"group-{label}",
                "Synthetic text",
                label,
                f"class-{label}",
            )
            for label in range(3)
        ),
        (0, 1, 2),
        ("class-0", "class-1", "class-2"),
        "a" * 64,
        "b" * 64,
    )


def test_predictions_align_by_id_regardless_of_order(validation: LoadedSplit) -> None:
    predictions = [Prediction("id-2", 2), Prediction("id-0", 0), Prediction("id-1", 1)]
    result = evaluate_predictions(validation, iter(predictions))
    assert result.accuracy == result.macro_f1 == 1.0
    assert result == evaluate_predictions(validation, list(reversed(predictions)))


@pytest.mark.parametrize(
    "case", ["duplicate", "missing", "unexpected", "wrong_label", "bad_id"]
)
def test_invalid_prediction_sets_are_rejected(
    validation: LoadedSplit, case: str
) -> None:
    predictions = [Prediction(f"id-{label}", label) for label in range(3)]
    if case == "duplicate":
        predictions.append(predictions[0])
    elif case == "missing":
        predictions.pop()
    elif case == "unexpected":
        predictions[0] = Prediction("another-split-id", 0)
    elif case == "wrong_label":
        predictions[0] = Prediction("id-0", 99)
    else:
        predictions[0] = Prediction("", 0)
    with pytest.raises(EvaluationError):
        evaluate_predictions(validation, predictions)
