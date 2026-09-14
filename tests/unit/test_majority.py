"""Check the training-only decision rule and portable majority model state."""

import json

import pytest

from filing_sentence_classifier.baselines.majority import (
    BaselineError,
    MajorityClassifier,
)


def test_fit_counts_rows_and_predictions_ignore_text_features() -> None:
    model = MajorityClassifier.fit(iter([2, 0, 2, 1, 2]), label_ids=[0, 1, 2, 3])
    assert model.class_counts == (1, 1, 3, 0)
    assert model.majority_label == 2
    assert model.predict(iter(["Past results.", "Future plans.", ""])) == (2, 2, 2)
    assert model.predict([]) == ()


def test_tie_uses_smallest_id_regardless_of_class_or_row_order() -> None:
    for targets in ([2, 1, 2, 1], [1, 2, 1, 2]):
        model = MajorityClassifier.fit(targets, label_ids=[2, 1, 0])
        assert model.majority_label == 1
        assert model.label_ids == (2, 1, 0)


@pytest.mark.parametrize(
    "targets,labels",
    [
        ([], [0, 1]),
        ([1], []),
        ([1], [1, 1]),
        ([3], [0, 1]),
        ([True], [0, 1]),
        ([1.0], [0, 1]),
        (["1"], [0, 1]),
        ([1], [True]),
    ],
)
def test_invalid_training_inputs_are_rejected(targets: list, labels: list) -> None:
    with pytest.raises(BaselineError):
        MajorityClassifier.fit(targets, label_ids=labels)


def test_json_round_trip_preserves_counts_and_predictions() -> None:
    model = MajorityClassifier.fit([2, 2, 1], label_ids=[2, 0, 1])
    restored = MajorityClassifier.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored == model
    assert restored.predict(["A new sentence."]) == (2,)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("family", "other"),
        ("tie_break", "first_seen"),
        ("majority_label", 0),
        ("majority_label", True),
        ("class_counts", [0, 0]),
        ("class_counts", [-1, 3]),
        ("class_counts", [1]),
        ("label_ids", [1, 1]),
    ],
)
def test_invalid_saved_state_is_rejected(field: str, value: object) -> None:
    state = MajorityClassifier.fit([1, 1, 0], label_ids=[0, 1]).to_dict()
    state[field] = value
    with pytest.raises(BaselineError):
        MajorityClassifier.from_dict(state)
