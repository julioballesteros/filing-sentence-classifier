"""Verify fitted vocabulary ordering, filtering, lookup, and state validation."""

import json
from dataclasses import FrozenInstanceError

import pytest

from filing_sentence_classifier.text.tokenization import tokenize
from filing_sentence_classifier.text.vocabulary import (
    PAD_ID,
    PAD_TOKEN,
    UNK_ID,
    UNK_TOKEN,
    Vocabulary,
    VocabularyError,
)


def test_fit_uses_occurrences_and_filters_rare_tokens() -> None:
    documents = [("may", "grow", "may", "the"), ("will", "grow", "may", "the")]
    vocabulary = Vocabulary.fit(iter(iter(row) for row in documents))
    assert vocabulary.tokens == (PAD_TOKEN, UNK_TOKEN, "may", "grow", "the")
    assert vocabulary.counts == (0, 0, 3, 2, 2)
    assert len(vocabulary) == 5
    assert vocabulary[PAD_TOKEN] == PAD_ID == 0
    assert vocabulary[UNK_TOKEN] == UNK_ID == 1
    assert vocabulary["may"] == 2
    assert vocabulary["will"] == UNK_ID
    assert Vocabulary.fit([("may", "may")]).counts == (0, 0, 2)


def test_ties_and_size_limit_are_independent_of_input_order() -> None:
    documents = [("é", "b", "a"), ("a", "é", "b")]
    vocabulary = Vocabulary.fit(documents, max_size=4)
    assert vocabulary.tokens == (PAD_TOKEN, UNK_TOKEN, "a", "b")
    assert vocabulary["é"] == UNK_ID
    reordered = [tuple(reversed(row)) for row in reversed(documents)]
    assert Vocabulary.fit(reordered, max_size=4).to_dict() == vocabulary.to_dict()
    assert Vocabulary.fit(documents, max_size=3).tokens == (PAD_TOKEN, UNK_TOKEN, "a")
    assert Vocabulary.fit(documents).tokens[-1] == "é"


def test_minimum_frequency_is_inclusive_and_precedes_size_limit() -> None:
    documents = [("high", "high", "high", "low", "low", "once")]
    assert Vocabulary.fit(documents, min_frequency=3, max_size=5).tokens == (
        PAD_TOKEN,
        UNK_TOKEN,
        "high",
    )
    assert Vocabulary.fit(documents, min_frequency=1)["once"] != UNK_ID


def test_validation_only_tokens_do_not_change_the_fitted_vocabulary() -> None:
    vocabulary = Vocabulary.fit([tokenize("We will grow.")], min_frequency=1)
    before = vocabulary.to_dict()
    validation_tokens = tokenize("We will pivot.")
    assert vocabulary["pivot"] == UNK_ID
    assert tuple(vocabulary[token] for token in validation_tokens) == (4, 5, UNK_ID, 2)
    assert "pivot" not in vocabulary.token_to_id
    assert vocabulary.to_dict() == before
    with pytest.raises(TypeError):
        vocabulary.token_to_id["pivot"] = 99  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        vocabulary.tokens = ()  # type: ignore[misc]


def test_json_round_trip_preserves_every_id_without_refitting() -> None:
    vocabulary = Vocabulary.fit([tokenize("Café’s plans: $1,250.50.")], min_frequency=1)
    state = json.loads(json.dumps(vocabulary.to_dict()))
    restored = Vocabulary.from_dict(state)
    assert restored == vocabulary
    assert dict(restored.token_to_id) == dict(vocabulary.token_to_id)
    assert [restored[token] for token in vocabulary.tokens] == list(
        range(len(vocabulary))
    )
    assert restored["unseen"] == UNK_ID
    state["tokens"][2] = "changed"
    state["counts"][2] = 999
    state["recipe"]["special_tokens"].clear()
    assert restored.to_dict() == vocabulary.to_dict()


@pytest.mark.parametrize(
    "documents",
    [
        [],
        [()],
        [("common", "common"), ()],
        "raw text",
        ["raw text"],
        [b"raw text"],
        [("only-once",)],
        [("", "")],
        [("two words",)],
        [(12,)],
        [(PAD_TOKEN,)],
        [(UNK_TOKEN,)],
    ],
)
def test_invalid_or_empty_training_inputs_are_rejected(documents) -> None:
    with pytest.raises(VocabularyError):
        Vocabulary.fit(documents)


@pytest.mark.parametrize(
    "options",
    [
        {"min_frequency": 0},
        {"min_frequency": True},
        {"min_frequency": 1.5},
        {"max_size": 2},
        {"max_size": True},
        {"max_size": 3.5},
    ],
)
def test_invalid_limits_are_rejected(options: dict) -> None:
    with pytest.raises(VocabularyError):
        Vocabulary.fit([("word", "word")], **options)


@pytest.mark.parametrize("token", [None, 1, "", "two words"])
def test_invalid_lookup_tokens_are_rejected(token) -> None:
    vocabulary = Vocabulary.fit([("word", "word")])
    with pytest.raises(VocabularyError):
        vocabulary[token]


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("extra", "unsupported"),
        ("tokens", [UNK_TOKEN, PAD_TOKEN, "a", "b"]),
        ("tokens", [PAD_TOKEN, UNK_TOKEN, "a", "a"]),
        ("tokens", [PAD_TOKEN, UNK_TOKEN, "b", "a"]),
        ("tokens", [PAD_TOKEN, UNK_TOKEN, "", "b"]),
        ("tokens", [PAD_TOKEN, UNK_TOKEN, 1, "b"]),
        ("counts", [0, 0, 2]),
        ("counts", [1, 0, 2, 2]),
        ("counts", [0, 0, 2, -1]),
        ("counts", [0, 0, 2, True]),
        ("counts", [0, 0, 2, 1]),
        ("counts", [0, 0, 2, 3]),
    ],
)
def test_corrupt_saved_tables_are_rejected(field: str, value: object) -> None:
    state = Vocabulary.fit([("a", "b", "a", "b")]).to_dict()
    state[field] = value
    with pytest.raises(VocabularyError):
        Vocabulary.from_dict(state)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", "2"),
        ("min_frequency", True),
        ("max_size", 3),
        ("ordering", "first_seen"),
        ("frequency", "document_frequency"),
        ("special_tokens", {PAD_TOKEN: 1, UNK_TOKEN: 0}),
        ("special_tokens", {PAD_TOKEN: False, UNK_TOKEN: 1}),
        ("unknown_token_id", 0),
        ("unknown_token_id", True),
        ("max_size_includes_special_tokens", 1),
    ],
)
def test_incompatible_saved_recipes_are_rejected(field: str, value: object) -> None:
    state = json.loads(json.dumps(Vocabulary.fit([("a", "b", "a", "b")]).to_dict()))
    state["recipe"][field] = value
    with pytest.raises(VocabularyError):
        Vocabulary.from_dict(state)


def test_missing_fields_and_mutable_constructor_inputs_are_rejected() -> None:
    state = Vocabulary.fit([("word", "word")]).to_dict()
    del state["recipe"]
    with pytest.raises(VocabularyError, match="schema"):
        Vocabulary.from_dict(state)
    with pytest.raises(VocabularyError, match="immutable"):
        Vocabulary([PAD_TOKEN, UNK_TOKEN, "word"], (0, 0, 2))  # type: ignore[arg-type]
