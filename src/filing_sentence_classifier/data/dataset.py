"""PyTorch access to a frozen, eagerly encoded sentence partition."""

import operator
from typing import TypedDict

import torch
from torch import Tensor
from torch.utils.data import Dataset

from filing_sentence_classifier.data.loading import LoadedSplit
from filing_sentence_classifier.text.encoding import EncodedText, TextEncoder


class DatasetError(ValueError):
    """A partition cannot satisfy the supervised sentence dataset contract."""


class SentenceItem(TypedDict):
    """One unpadded CPU example; label is a scalar long tensor."""

    input_ids: Tensor
    label: Tensor
    sample_id: str
    original_length: int
    truncated: bool
    truncated_tokens: int
    unknown_count: int
    original_unknown_count: int


class SentenceDataset(Dataset[SentenceItem]):
    """Adapt a loaded partition and fitted encoder to integer-indexed examples.

    Encode once at construction, preserving the saved row order and label IDs.
    Cache immutable encodings, then create fresh CPU tensors on each access so
    in-place changes cannot affect future reads. Keep partition provenance for
    evaluation and manifests. No file I/O, fitting, shuffling, or padding occurs
    here. The encoder is not retained; the dataset can be sent to spawn workers.

    The complete partition must fit in memory. Class IDs must be contiguous from
    zero so labels can be used directly as cross-entropy targets.
    """

    def __init__(self, split: LoadedSplit, encoder: TextEncoder) -> None:
        if not isinstance(split, LoadedSplit) or not isinstance(encoder, TextEncoder):
            raise DatasetError("Expected a LoadedSplit and a fitted TextEncoder.")
        if not split.records:
            raise DatasetError("The dataset must contain at least one sentence.")
        if (
            not split.label_ids
            or any(type(label) is not int for label in split.label_ids)
            or split.label_ids != tuple(range(len(split.label_ids)))
            or len(split.label_names) != len(split.label_ids)
        ):
            raise DatasetError(
                "Class IDs must be contiguous from zero with aligned names."
            )

        encodings = []
        seen_ids: set[str] = set()
        for index, record in enumerate(split.records):
            if (
                not isinstance(record.sample_id, str)
                or not record.sample_id
                or record.sample_id in seen_ids
            ):
                raise DatasetError(f"Sample {index} must have a unique nonempty ID.")
            if (
                type(record.label) is not int
                or record.label not in split.label_ids
                or record.label_text != split.label_names[record.label]
            ):
                raise DatasetError(
                    f"Invalid class label for sample {record.sample_id}."
                )
            try:
                encodings.append(encoder.encode(record.text))
            except (TypeError, ValueError) as exc:
                raise DatasetError(
                    f"Could not encode sample {record.sample_id} at index {index}: {exc}"
                ) from exc
            seen_ids.add(record.sample_id)
        self._split = split
        self._encodings = tuple(encodings)

    @property
    def split(self) -> LoadedSplit:
        """Return the frozen source partition, including its hashes and class order."""
        return self._split

    @property
    def encodings(self) -> tuple[EncodedText, ...]:
        """Expose immutable cached results for diagnostics without encoding again."""
        return self._encodings

    def __len__(self) -> int:
        return len(self._encodings)

    def __getitem__(self, index: int) -> SentenceItem:
        """Return one example; support integer indexing, including negative indices."""
        if isinstance(index, bool):
            raise TypeError("Dataset indices must be integers, not booleans.")
        position = operator.index(index)
        record = self._split.records[position]
        encoded = self._encodings[position]
        return {
            "input_ids": torch.tensor(
                encoded.input_ids, dtype=torch.long, device="cpu"
            ),
            "label": torch.tensor(record.label, dtype=torch.long, device="cpu"),
            "sample_id": record.sample_id,
            "original_length": encoded.original_length,
            "truncated": encoded.truncated,
            "truncated_tokens": encoded.truncated_tokens,
            "unknown_count": encoded.unknown_count,
            "original_unknown_count": encoded.original_unknown_count,
        }
