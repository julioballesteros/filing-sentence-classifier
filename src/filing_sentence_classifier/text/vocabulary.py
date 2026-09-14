"""Immutable word vocabularies with deterministic IDs and portable JSON state."""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Self

VOCABULARY_VERSION = "1"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1
_SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN)


class VocabularyError(ValueError):
    """Vocabulary inputs or saved state violate the vocabulary contract."""


def _validate_limits(min_frequency: int, max_size: int | None) -> None:
    if type(min_frequency) is not int or min_frequency < 1:
        raise VocabularyError("min_frequency must be a positive integer.")
    if max_size is not None and (type(max_size) is not int or max_size < 3):
        raise VocabularyError("max_size must be at least 3, including PAD and UNK.")


def _validate_token(token: str) -> None:
    if (
        not isinstance(token, str)
        or not token
        or any(character.isspace() for character in token)
    ):
        raise VocabularyError("Tokens must be nonempty strings without whitespace.")


@dataclass(frozen=True)
class Vocabulary:
    """An ID-ordered token table and its training occurrence counts.

    PAD and UNK occupy IDs 0 and 1 with zero counts. All other entries are sorted
    by descending occurrence count, then ascending Unicode token value. Lookup
    never adds entries. Fit accepts tokenized training sentences; the caller is
    responsible for partition selection and tokenization.
    """

    tokens: tuple[str, ...]
    counts: tuple[int, ...]
    min_frequency: int = 2
    max_size: int | None = None
    _token_to_id: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_limits(self.min_frequency, self.max_size)
        if (
            not isinstance(self.tokens, tuple)
            or not isinstance(self.counts, tuple)
            or len(self.tokens) < 3
            or len(self.counts) != len(self.tokens)
        ):
            raise VocabularyError("Expected aligned immutable token and count tuples.")
        for token in self.tokens:
            _validate_token(token)
        if (
            self.tokens[:2] != _SPECIAL_TOKENS
            or len(set(self.tokens)) != len(self.tokens)
            or any(type(count) is not int or count < 0 for count in self.counts)
            or self.counts[:2] != (0, 0)
            or any(count < self.min_frequency for count in self.counts[2:])
            or (self.max_size is not None and len(self.tokens) > self.max_size)
        ):
            raise VocabularyError("Invalid special tokens, training counts, or size.")
        entries = list(zip(self.tokens[2:], self.counts[2:], strict=True))
        if entries != sorted(entries, key=lambda item: (-item[1], item[0])):
            raise VocabularyError("Tokens must follow frequency and Unicode tie order.")
        object.__setattr__(
            self,
            "_token_to_id",
            MappingProxyType({token: index for index, token in enumerate(self.tokens)}),
        )

    @classmethod
    def fit(
        cls,
        documents: Iterable[Iterable[str]],
        *,
        min_frequency: int = 2,
        max_size: int | None = None,
    ) -> Self:
        """Count all training occurrences, including repetitions within a sentence.

        Filter by min_frequency, then apply max_size (including both reserved
        entries). Reject raw strings, empty sentences, reserved-token collisions,
        and a corpus with no retained ordinary tokens. No randomness is used.
        """
        _validate_limits(min_frequency, max_size)
        if isinstance(documents, (str, bytes)):
            raise VocabularyError("Expected tokenized documents, not a raw string.")
        counts: Counter[str] = Counter()
        for document in documents:
            if isinstance(document, (str, bytes)):
                raise VocabularyError("Expected tokenized documents, not raw strings.")
            empty = True
            for token in document:
                _validate_token(token)
                if token in _SPECIAL_TOKENS:
                    raise VocabularyError(
                        "Training tokens must not use reserved names."
                    )
                counts[token] += 1
                empty = False
            if empty:
                raise VocabularyError("Training documents must contain tokens.")
        entries = sorted(
            (
                (token, count)
                for token, count in counts.items()
                if count >= min_frequency
            ),
            key=lambda item: (-item[1], item[0]),
        )
        if max_size is not None:
            entries = entries[: max_size - len(_SPECIAL_TOKENS)]
        if not entries:
            raise VocabularyError("No training tokens meet min_frequency.")
        return cls(
            _SPECIAL_TOKENS + tuple(token for token, _ in entries),
            (0, 0) + tuple(count for _, count in entries),
            min_frequency,
            max_size,
        )

    def __len__(self) -> int:
        """Return the embedding table size, including PAD and UNK."""
        return len(self.tokens)

    def __getitem__(self, token: str) -> int:
        """Look up an already-tokenized string; unknown tokens return UNK_ID."""
        _validate_token(token)
        return self._token_to_id.get(token, UNK_ID)

    @property
    def token_to_id(self) -> Mapping[str, int]:
        """Expose the fitted mapping read-only, without a mutable global vocabulary."""
        return self._token_to_id

    def recipe(self) -> dict[str, object]:
        """Describe the fixed ID policy and effective vocabulary limits."""
        return {
            "version": VOCABULARY_VERSION,
            "min_frequency": self.min_frequency,
            "max_size": self.max_size,
            "max_size_includes_special_tokens": True,
            "frequency": "token_occurrences",
            "ordering": "frequency_descending_then_unicode_ascending",
            "special_tokens": {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID},
            "unknown_token_id": UNK_ID,
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize the ordered table so IDs survive a JSON round trip exactly."""
        return {
            "schema_version": 1,
            "recipe": self.recipe(),
            "tokens": list(self.tokens),
            "counts": list(self.counts),
        }

    @classmethod
    def from_dict(cls, state: object) -> Self:
        """Restore and validate fitted state without recounting a corpus."""
        if (
            not isinstance(state, dict)
            or set(state) != {"schema_version", "recipe", "tokens", "counts"}
            or type(state["schema_version"]) is not int
            or state["schema_version"] != 1
            or not isinstance(state["recipe"], dict)
            or "min_frequency" not in state["recipe"]
            or "max_size" not in state["recipe"]
            or not isinstance(state["tokens"], list)
            or not isinstance(state["counts"], list)
        ):
            raise VocabularyError("Unsupported vocabulary schema.")
        vocabulary = cls(
            tuple(state["tokens"]),
            tuple(state["counts"]),
            state["recipe"]["min_frequency"],
            state["recipe"]["max_size"],
        )
        recipe = state["recipe"]
        if (
            recipe != vocabulary.recipe()
            or type(recipe["max_size_includes_special_tokens"]) is not bool
            or type(recipe["unknown_token_id"]) is not int
            or any(
                type(index) is not int for index in recipe["special_tokens"].values()
            )
        ):
            raise VocabularyError("Unsupported vocabulary recipe.")
        return vocabulary
