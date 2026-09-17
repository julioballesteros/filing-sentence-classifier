"""Immutable prediction contracts shared by model adapters, CLI, and future APIs."""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from filing_sentence_classifier.data.cleaning import clean_text

PREDICTION_SCHEMA_VERSION = 1
PROBABILITY_SUM_TOLERANCE = 1e-6


class PredictionContractError(ValueError):
    """Inputs or adapter outputs violate the public prediction contract."""


def validate_model_id(value: str) -> None:
    """Require a portable version identifier, distinct from the schema version."""
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value) is None
    ):
        raise PredictionContractError(
            "model_id must contain 1–128 ASCII letters, digits, dots, underscores, "
            "or hyphens, starting with a letter or digit."
        )


@dataclass(frozen=True)
class LabelSet:
    """Names indexed by contiguous class IDs, independent of the source dataset."""

    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.names, tuple)
            or len(self.names) < 2
            or any(
                not isinstance(name, str) or not name.strip() or name != name.strip()
                for name in self.names
            )
            or len(set(self.names)) != len(self.names)
        ):
            raise PredictionContractError("Expected at least two unique label names.")

    def to_dict(self) -> dict[str, str]:
        return {str(index): name for index, name in enumerate(self.names)}

    @classmethod
    def from_dict(cls, state: object) -> Self:
        if not isinstance(state, dict) or set(state) != {
            str(index) for index in range(len(state))
        }:
            raise PredictionContractError(
                "Label IDs must be the strings 0 through N-1."
            )
        return cls(tuple(state[str(index)] for index in range(len(state))))


@dataclass(frozen=True)
class InputLimits:
    """Request bounds; max_characters counts raw Python Unicode characters."""

    max_characters: int = 10_000
    max_batch_size: int = 256

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (self.max_characters, self.max_batch_size)
        ):
            raise PredictionContractError("Input limits must be positive integers.")

    def to_dict(self) -> dict[str, int]:
        return {
            "max_characters": self.max_characters,
            "max_batch_size": self.max_batch_size,
        }

    @classmethod
    def from_dict(cls, state: object) -> Self:
        if not isinstance(state, dict) or set(state) != {
            "max_characters",
            "max_batch_size",
        }:
            raise PredictionContractError("Expected complete input limits.")
        return cls(state["max_characters"], state["max_batch_size"])


DEFAULT_INPUT_LIMITS = InputLimits()


def prepare_texts(
    texts: Sequence[str], *, limits: InputLimits = DEFAULT_INPUT_LIMITS
) -> tuple[str, ...]:
    """Validate a complete request and apply the existing cleaning policy once.

    Return an immutable, ordered batch of prepared text, including duplicates.
    Empty requests return (). Reject invalid requests before any adapter runs;
    errors identify the zero-based item index and never include the input text.
    A string is not a batch. Iterators are excluded so size is bounded up front.
    Token truncation belongs to the saved encoder, not these request limits.
    """
    if not isinstance(limits, InputLimits):
        raise PredictionContractError("Expected InputLimits.")
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise PredictionContractError("Expected a sequence of sentences, not a string.")
    if len(texts) > limits.max_batch_size:
        raise PredictionContractError(
            f"Request exceeds max_batch_size={limits.max_batch_size}."
        )
    prepared: list[str] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise PredictionContractError(f"Input at index {index} must be a string.")
        if len(text) > limits.max_characters:
            raise PredictionContractError(
                f"Input at index {index} exceeds max_characters={limits.max_characters}."
            )
        try:
            text.encode("utf-8")
            cleaned = clean_text(text).text
        except (UnicodeError, ValueError) as exc:
            raise PredictionContractError(
                f"Input at index {index} contains unsupported Unicode."
            ) from exc
        if not cleaned:
            raise PredictionContractError(f"Input at index {index} is blank.")
        prepared.append(cleaned)
    return tuple(prepared)


@dataclass(frozen=True)
class Prediction:
    """One uncalibrated class distribution in the bundle's class-ID order.

    Derive the label from argmax, keeping the lowest ID on exact ties. Adapters
    must reorder their class scores before construction. Scores are validated,
    never silently normalized, and serialized with explicit string class IDs.
    """

    model_id: str
    labels: LabelSet
    probabilities: tuple[float, ...]
    truncated: bool = False

    def __post_init__(self) -> None:
        validate_model_id(self.model_id)
        if not isinstance(self.labels, LabelSet):
            raise PredictionContractError("Expected a LabelSet.")
        if (
            not isinstance(self.probabilities, tuple)
            or len(self.probabilities) != len(self.labels.names)
            or any(
                type(value) not in (int, float)
                or not 0 <= value <= 1
                or not math.isfinite(value)
                for value in self.probabilities
            )
            or not math.isclose(
                math.fsum(self.probabilities),
                1.0,
                rel_tol=0.0,
                abs_tol=PROBABILITY_SUM_TOLERANCE,
            )
        ):
            raise PredictionContractError(
                "Expected one finite probability per class in [0, 1], summing to 1."
            )
        if type(self.truncated) is not bool:
            raise PredictionContractError("truncated must be a boolean.")

    @property
    def label_id(self) -> int:
        return max(range(len(self.probabilities)), key=self.probabilities.__getitem__)

    @property
    def label(self) -> str:
        return self.labels.names[self.label_id]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "model_id": self.model_id,
            "label_id": self.label_id,
            "label": self.label,
            "probabilities": {
                str(index): float(value)
                for index, value in enumerate(self.probabilities)
            },
            "truncated": self.truncated,
        }
