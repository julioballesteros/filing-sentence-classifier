"""Public input boundaries, immutable class order, and coherent output scores."""

import json
from dataclasses import FrozenInstanceError

import pytest

from filing_sentence_classifier.inference import contracts
from filing_sentence_classifier.inference.contracts import (
    InputLimits,
    LabelSet,
    Prediction,
    PredictionContractError,
    prepare_texts,
    validate_model_id,
)

LABELS = LabelSet(("specific", "historical", "generic"))


def test_labels_use_explicit_ids_not_dictionary_insertion_order():
    labels = LabelSet.from_dict({"2": "generic", "0": "specific", "1": "historical"})
    assert labels == LABELS
    assert labels.to_dict() == {"0": "specific", "1": "historical", "2": "generic"}
    assert LabelSet.from_dict(json.loads(json.dumps(labels.to_dict()))) == labels
    with pytest.raises(FrozenInstanceError):
        labels.names = ("changed", "labels")


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"1": "a", "2": "b"},
        {0: "a", 1: "b"},
        {"00": "a", "1": "b"},
        {"0": "a", "1": "a"},
        {"0": "a", "1": " "},
        {"0": "a", "1": 1},
    ],
)
def test_invalid_label_mapping_is_rejected(value):
    with pytest.raises(PredictionContractError):
        LabelSet.from_dict(value)


@pytest.mark.parametrize("names", [["a", "b"], ("a",), ("a", " b"), ("a", [])])
def test_labels_require_an_immutable_valid_order(names):
    with pytest.raises(PredictionContractError):
        LabelSet(names)


@pytest.mark.parametrize(
    "value",
    ["", "../model", "/model", "a/b", "a\\b", "model name", ".hidden", "x" * 129, None],
)
def test_model_ids_are_portable_version_identifiers(value):
    with pytest.raises(PredictionContractError):
        validate_model_id(value)
    validate_model_id("mean-pool-mlp-v1")


def test_request_preparation_reuses_cleaning_and_preserves_order_and_duplicates():
    raw = ["  Cafe\u0301\u0092s\tforecast.\n", "We expect growth.", "We expect growth."]
    original = raw.copy()
    assert prepare_texts(raw) == (
        "Café’s forecast.",
        "We expect growth.",
        "We expect growth.",
    )
    assert raw == original
    assert prepare_texts([]) == ()
    assert prepare_texts(()) == ()


@pytest.mark.parametrize(
    "value", ["one sentence", b"bytes", None, 17, iter(["sentence"])]
)
def test_batch_must_be_a_bounded_sequence(value):
    with pytest.raises(PredictionContractError, match="sequence"):
        prepare_texts(value)


@pytest.mark.parametrize(
    "text", [None, 1, b"bytes", "", " \t\n", "bad\ufffd", "bad\u0085", "bad\ud800"]
)
def test_invalid_item_rejects_the_whole_request_with_index_and_without_echo(text):
    with pytest.raises(PredictionContractError, match="index 1") as error:
        prepare_texts(["Valid first sentence.", text])
    assert "Valid first sentence" not in str(error.value)
    assert "bad" not in str(error.value)


def test_character_limit_applies_before_cleaning_and_does_not_truncate():
    limits = InputLimits(max_characters=4, max_batch_size=2)
    assert prepare_texts(["Café", "ab c"], limits=limits) == ("Café", "ab c")
    with pytest.raises(PredictionContractError, match="index 0.*max_characters=4"):
        prepare_texts([" a   "], limits=limits)
    assert prepare_texts(["a " * 128]) == (("a " * 128).strip(),)


def test_oversized_requests_are_rejected_before_preprocessing(monkeypatch):
    def unexpected(*args):
        pytest.fail("Oversized batches must not start preprocessing.")

    monkeypatch.setattr(contracts, "clean_text", unexpected)
    with pytest.raises(PredictionContractError, match="max_batch_size=1"):
        prepare_texts(["First.", "Second."], limits=InputLimits(max_batch_size=1))


@pytest.mark.parametrize("field", ["max_characters", "max_batch_size"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "5", None])
def test_input_limits_reject_invalid_values(field, value):
    state = InputLimits().to_dict() | {field: value}
    with pytest.raises(PredictionContractError):
        InputLimits.from_dict(state)


def test_input_limit_schema_is_complete_and_round_trips():
    limits = InputLimits(100, 4)
    assert InputLimits.from_dict(json.loads(json.dumps(limits.to_dict()))) == limits
    for state in ({"max_characters": 100}, limits.to_dict() | {"unknown": 1}, None):
        with pytest.raises(PredictionContractError):
            InputLimits.from_dict(state)


def test_prediction_derives_consistent_label_and_serializes_explicit_class_scores():
    result = Prediction("model-v1", LABELS, (0.1, 0.2, 0.7), truncated=True)
    assert (result.label_id, result.label) == (2, "generic")
    assert json.loads(json.dumps(result.to_dict())) == {
        "schema_version": 1,
        "model_id": "model-v1",
        "label_id": 2,
        "label": "generic",
        "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
        "truncated": True,
    }
    tie = Prediction("tie-v1", LabelSet(("z", "a", "b")), (0.5, 0.5, 0.0))
    assert (tie.label_id, tie.label) == (0, "z")
    with pytest.raises(FrozenInstanceError):
        result.truncated = False


def test_probability_tolerance_accepts_float32_rounding_without_renormalizing():
    values = (0.1, 0.2, 0.70000003)
    result = Prediction("model-v1", LABELS, values)
    assert tuple(result.to_dict()["probabilities"].values()) == values


@pytest.mark.parametrize(
    "scores",
    [
        (0.2, 0.8),
        [0.1, 0.2, 0.7],
        (0.1, 0.2, 0.6),
        (0.1, 0.2, 0.70001),
        (-0.1, 0.2, 0.9),
        (True, 0, 0),
        (0, 0, float("nan")),
        (0, 0, float("inf")),
        (0, 0, 10**1000),
        ("0", 0, 1),
    ],
)
def test_invalid_scores_fail_instead_of_being_coerced_or_normalized(scores):
    with pytest.raises(PredictionContractError):
        Prediction("model-v1", LABELS, scores)


def test_prediction_rejects_invalid_labels_and_truncation_metadata():
    with pytest.raises(PredictionContractError):
        Prediction("model-v1", ("a", "b"), (0.5, 0.5))
    with pytest.raises(PredictionContractError, match="boolean"):
        Prediction("model-v1", LABELS, (0.1, 0.2, 0.7), truncated=1)
