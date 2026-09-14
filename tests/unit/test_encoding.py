"""Verify shared ID encoding, truncation boundaries, and portable state."""

import json
from dataclasses import FrozenInstanceError

import pytest

from filing_sentence_classifier.text.encoding import (
    DEFAULT_MAX_LENGTH,
    EncodedText,
    EncodingError,
    TextEncoder,
    encoding_statistics,
)
from filing_sentence_classifier.text.tokenization import tokenize
from filing_sentence_classifier.text.vocabulary import PAD_ID, UNK_ID, Vocabulary


@pytest.fixture
def vocabulary() -> Vocabulary:
    return Vocabulary.fit(
        [tokenize("We don't expect long-term growth of 12.5%.")], min_frequency=1
    )


def test_encoder_reuses_tokenization_and_the_exact_fitted_ids(
    vocabulary: Vocabulary,
) -> None:
    encoder = TextEncoder(vocabulary)
    result = encoder.encode("WE don’t expect long-term growth of 12.5%.")
    tokens = (
        "we",
        "don't",
        "expect",
        "long",
        "-",
        "term",
        "growth",
        "of",
        "12.5",
        "%",
        ".",
    )
    assert result.input_ids == tuple(vocabulary[token] for token in tokens)
    assert result.original_length == len(tokens)
    assert result.original_unknown_count == result.unknown_count == 0
    assert result.truncated_tokens == 0
    assert not result.truncated
    assert PAD_ID not in result.input_ids
    assert encoder.max_length == DEFAULT_MAX_LENGTH == 128
    with pytest.raises(FrozenInstanceError):
        result.input_ids = ()  # type: ignore[misc]


@pytest.mark.parametrize("max_length", [1, 2, 3, 4])
def test_truncation_keeps_the_prefix_and_reports_only_actual_loss(
    vocabulary: Vocabulary, max_length: int
) -> None:
    encoder = TextEncoder(vocabulary, max_length=max_length)
    result = encoder.encode("we expect growth")
    expected = tuple(vocabulary[token] for token in ("we", "expect", "growth"))
    assert result.input_ids == expected[:max_length]
    assert result.original_length == 3
    assert result.truncated == (max_length < 3)
    assert result.truncated_tokens == max(3 - max_length, 0)


def test_unknown_counts_distinguish_full_text_from_kept_tokens(
    vocabulary: Vocabulary,
) -> None:
    encoder = TextEncoder(vocabulary, max_length=2)
    before = vocabulary.to_dict()
    result = encoder.encode("growth unicorn unknownsuffix")
    assert result.input_ids == (vocabulary["growth"], UNK_ID)
    assert result.unknown_count == 1
    assert result.original_unknown_count == 2
    assert result.original_length == 3
    assert result.truncated_tokens == 1
    assert vocabulary.to_dict() == before
    assert encoder.encode("unicorn").input_ids == (UNK_ID,)
    assert encoder.encode("unknown other").input_ids == (UNK_ID, UNK_ID)


def test_short_inputs_are_not_padded_and_special_names_are_ordinary_text(
    vocabulary: Vocabulary,
) -> None:
    encoder = TextEncoder(vocabulary)
    assert encoder.encode("growth").input_ids == (vocabulary["growth"],)
    assert encoder.encode(".").input_ids == (vocabulary["."],)
    assert encoder.encode("<PAD>").input_ids == (UNK_ID, UNK_ID, UNK_ID)


@pytest.mark.parametrize("text", ["", " \n\t", "\u00a0"])
def test_blank_text_is_rejected(vocabulary: Vocabulary, text: str) -> None:
    with pytest.raises(ValueError, match="non-whitespace"):
        TextEncoder(vocabulary).encode(text)


@pytest.mark.parametrize("text", [None, 1, b"growth", ["growth"]])
def test_invalid_text_is_not_coerced(vocabulary: Vocabulary, text) -> None:
    with pytest.raises(TypeError, match="string"):
        TextEncoder(vocabulary).encode(text)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, None])
def test_invalid_limits_are_rejected(vocabulary: Vocabulary, limit) -> None:
    with pytest.raises(EncodingError, match="positive integer"):
        TextEncoder(vocabulary, max_length=limit)


def test_json_round_trip_restores_rules_ids_and_metadata(
    vocabulary: Vocabulary,
) -> None:
    encoder = TextEncoder(vocabulary, max_length=3)
    state = json.loads(json.dumps(encoder.to_dict()))
    restored = TextEncoder.from_dict(state)
    for text in ("growth", "We don’t expect growth.", "unseen new words here"):
        assert restored.encode(text) == encoder.encode(text)
    state["encoding"]["max_length"] = 99
    state["vocabulary"]["tokens"][2] = "changed"
    assert restored.to_dict() == encoder.to_dict()


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("encoding", "version", "2"),
        ("encoding", "max_length", True),
        ("encoding", "truncation", "keep_last_tokens"),
        ("encoding", "padding", True),
        ("encoding", "padding", 0),
        ("encoding", "added_special_tokens", ["<BOS>"]),
        ("tokenization", "version", "2"),
        ("tokenization", "pattern", r"\w+"),
        ("vocabulary", "schema_version", 2),
    ],
)
def test_incompatible_saved_processing_is_rejected(
    vocabulary: Vocabulary, section: str, field: str, value: object
) -> None:
    state = json.loads(json.dumps(TextEncoder(vocabulary).to_dict()))
    state[section][field] = value
    with pytest.raises(EncodingError):
        TextEncoder.from_dict(state)


def test_incomplete_and_unsupported_states_are_rejected(vocabulary: Vocabulary) -> None:
    state = TextEncoder(vocabulary).to_dict()
    with pytest.raises(EncodingError, match="schema"):
        TextEncoder.from_dict({**state, "schema_version": True})
    del state["vocabulary"]
    with pytest.raises(EncodingError, match="schema"):
        TextEncoder.from_dict(state)
    with pytest.raises(EncodingError, match="fitted Vocabulary"):
        TextEncoder({})  # type: ignore[arg-type]


def test_statistics_use_token_weighting_and_complete_input_lengths() -> None:
    records = [
        EncodedText((2, 2, 2, 2), original_length=4, original_unknown_count=0),
        EncodedText((UNK_ID,), original_length=2, original_unknown_count=2),
    ]
    stats = encoding_statistics(iter(records))
    assert stats["rows"] == 2
    assert stats["original_tokens"] == 6
    assert stats["encoded_tokens"] == 5
    assert stats["truncated_rows"] == 1
    assert stats["truncated_row_rate"] == 0.5
    assert stats["discarded_tokens"] == 1
    assert stats["discarded_token_rate"] == pytest.approx(1 / 6)
    assert stats["original_unknown_tokens"] == 2
    assert stats["original_unknown_rate"] == pytest.approx(2 / 6)
    assert stats["encoded_unknown_tokens"] == 1
    assert stats["encoded_unknown_rate"] == 0.2
    assert stats["original_lengths"] == {
        "min": 2,
        "max": 4,
        "p50": 2,
        "p90": 4,
        "p95": 4,
        "p99": 4,
    }
    assert stats["percentile_method"] == "nearest_rank"
    with pytest.raises(EncodingError, match="at least one sentence"):
        encoding_statistics([])


@pytest.mark.parametrize(
    "ids,length,unknown",
    [
        ((), 0, 0),
        ((PAD_ID,), 1, 0),
        ((True,), 1, 1),
        ((2,), 0, 0),
        ((UNK_ID,), 1, 0),
        ((2,), 1, 1),
        ((2,), 2, 2),
        ((2,), True, 0),
        ((2,), 1, False),
    ],
)
def test_invalid_encoded_metadata_is_rejected(ids, length, unknown) -> None:
    with pytest.raises(EncodingError, match="consistent counts"):
        EncodedText(ids, length, unknown)
