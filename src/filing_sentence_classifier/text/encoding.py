"""Shared, versioned text encoding for training and inference."""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil
from typing import Self

from filing_sentence_classifier.text.tokenization import tokenization_recipe, tokenize
from filing_sentence_classifier.text.vocabulary import (
    PAD_ID,
    UNK_ID,
    Vocabulary,
    VocabularyError,
)

ENCODING_VERSION = "1"
DEFAULT_MAX_LENGTH = 128


class EncodingError(ValueError):
    """Encoder configuration, saved state, or encoded metadata is invalid."""


@dataclass(frozen=True)
class EncodedText:
    """Unpadded IDs and metadata describing the complete input sentence.

    original_unknown_count includes unknown tokens discarded by truncation;
    unknown_count counts only the IDs actually returned to the model.
    """

    input_ids: tuple[int, ...]
    original_length: int
    original_unknown_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.input_ids, tuple)
            or not self.input_ids
            or any(
                type(index) is not int or index <= PAD_ID for index in self.input_ids
            )
            or type(self.original_length) is not int
            or self.original_length < len(self.input_ids)
            or type(self.original_unknown_count) is not int
            or not self.unknown_count
            <= self.original_unknown_count
            <= self.unknown_count + self.truncated_tokens
        ):
            raise EncodingError("Expected nonempty unpadded IDs and consistent counts.")

    @property
    def truncated_tokens(self) -> int:
        return self.original_length - len(self.input_ids)

    @property
    def truncated(self) -> bool:
        return self.truncated_tokens > 0

    @property
    def unknown_count(self) -> int:
        return self.input_ids.count(UNK_ID)


@dataclass(frozen=True)
class TextEncoder:
    """Tokenize prepared text, look up frozen IDs, and keep the first max_length.

    Source cleaning stays upstream. No fitting, padding, BOS/EOS tokens, or tensor
    creation takes place here. max_length is a fixed configuration value; encoding
    new inputs never adjusts it or changes the vocabulary.
    """

    vocabulary: Vocabulary
    max_length: int = DEFAULT_MAX_LENGTH

    def __post_init__(self) -> None:
        if not isinstance(self.vocabulary, Vocabulary):
            raise EncodingError("Expected a fitted Vocabulary.")
        if type(self.max_length) is not int or self.max_length < 1:
            raise EncodingError("max_length must be a positive integer.")

    def encode(self, text: str) -> EncodedText:
        """Encode one prepared sentence; reject invalid or blank text.

        Count the complete tokenized input before truncation so unknown-token and
        truncation diagnostics remain visible even for discarded suffixes.
        """
        ids = tuple(self.vocabulary[token] for token in tokenize(text))
        return EncodedText(
            input_ids=ids[: self.max_length],
            original_length=len(ids),
            original_unknown_count=ids.count(UNK_ID),
        )

    def recipe(self) -> dict[str, object]:
        return {
            "version": ENCODING_VERSION,
            "input": "prepared_text",
            "max_length": self.max_length,
            "truncation": "keep_first_tokens",
            "padding": False,
            "added_special_tokens": [],
            "empty_input": "raise_value_error",
        }

    def to_dict(self) -> dict[str, object]:
        """Export a self-contained encoder, including its fitted vocabulary."""
        return {
            "schema_version": 1,
            "encoding": self.recipe(),
            "tokenization": tokenization_recipe(),
            "vocabulary": self.vocabulary.to_dict(),
        }

    @classmethod
    def from_dict(cls, state: object) -> Self:
        """Restore without training data; reject incompatible processing rules."""
        if (
            not isinstance(state, dict)
            or set(state)
            != {"schema_version", "encoding", "tokenization", "vocabulary"}
            or type(state["schema_version"]) is not int
            or state["schema_version"] != 1
            or not isinstance(state["encoding"], dict)
            or "max_length" not in state["encoding"]
        ):
            raise EncodingError("Unsupported encoder schema.")
        try:
            encoder = cls(
                Vocabulary.from_dict(state["vocabulary"]),
                max_length=state["encoding"]["max_length"],
            )
        except VocabularyError as exc:
            raise EncodingError(f"Invalid encoder vocabulary: {exc}") from exc
        # JSON comparison also distinguishes booleans from numeric lookalikes.
        try:
            compatible = json.dumps(
                state, sort_keys=True, allow_nan=False
            ) == json.dumps(encoder.to_dict(), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise EncodingError("Encoder state must be valid JSON data.") from exc
        if not compatible:
            raise EncodingError("Unsupported encoding or tokenization recipe.")
        return encoder


def encoding_statistics(encodings: Iterable[EncodedText]) -> dict[str, object]:
    """Summarize a nonempty partition, with occurrence-weighted unknown rates.

    Length percentiles use complete inputs and the nearest-rank definition.
    This reports observations only; it never chooses or adjusts encoder limits.
    """
    records = tuple(encodings)
    if not records:
        raise EncodingError("Encoding statistics require at least one sentence.")
    lengths = sorted(record.original_length for record in records)
    original_tokens = sum(lengths)
    encoded_tokens = sum(len(record.input_ids) for record in records)
    truncated_rows = sum(record.truncated for record in records)
    original_unknown = sum(record.original_unknown_count for record in records)
    encoded_unknown = sum(record.unknown_count for record in records)
    return {
        "rows": len(records),
        "original_tokens": original_tokens,
        "encoded_tokens": encoded_tokens,
        "truncated_rows": truncated_rows,
        "truncated_row_rate": truncated_rows / len(records),
        "discarded_tokens": original_tokens - encoded_tokens,
        "discarded_token_rate": (original_tokens - encoded_tokens) / original_tokens,
        "original_unknown_tokens": original_unknown,
        "original_unknown_rate": original_unknown / original_tokens,
        "encoded_unknown_tokens": encoded_unknown,
        "encoded_unknown_rate": encoded_unknown / encoded_tokens,
        "original_lengths": {
            "min": lengths[0],
            "max": lengths[-1],
            **{
                f"p{percentile}": lengths[ceil(percentile * len(lengths) / 100) - 1]
                for percentile in (50, 90, 95, 99)
            },
        },
        "percentile_method": "nearest_rank",
    }
