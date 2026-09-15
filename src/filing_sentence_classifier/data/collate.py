"""Dynamic right padding and aligned metadata for supervised sentence batches."""

from collections.abc import Sequence
from typing import TypedDict

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from filing_sentence_classifier.data.dataset import SentenceItem
from filing_sentence_classifier.text.vocabulary import PAD_ID, UNK_ID

COLLATION_VERSION = "1"


class CollationError(ValueError):
    """Examples cannot form a valid, nonempty supervised batch."""


class SentenceBatch(TypedDict):
    """CPU tensors: inputs/mask [B, L], labels and numeric metadata [B]."""

    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor
    sample_ids: list[str]
    lengths: Tensor
    original_lengths: Tensor
    truncated: Tensor
    truncated_tokens: Tensor
    unknown_counts: Tensor
    original_unknown_counts: Tensor


def _validate_item(item: SentenceItem, position: int) -> None:
    if not isinstance(item, dict) or not SentenceItem.__required_keys__ <= item.keys():
        raise CollationError(
            f"Batch item {position} is missing required sentence fields."
        )
    ids, label = item["input_ids"], item["label"]
    if (
        not isinstance(ids, Tensor)
        or ids.device.type != "cpu"
        or ids.layout != torch.strided
        or ids.dtype != torch.long
        or ids.ndim != 1
        or ids.numel() == 0
        or bool(torch.any(ids <= PAD_ID))
    ):
        raise CollationError(
            f"Batch item {position} needs nonempty, unpadded CPU long input_ids."
        )
    if (
        not isinstance(label, Tensor)
        or label.device.type != "cpu"
        or label.layout != torch.strided
        or label.dtype != torch.long
        or label.ndim != 0
        or int(label) < 0
    ):
        raise CollationError(
            f"Batch item {position} needs a nonnegative CPU long scalar label."
        )
    if not isinstance(item["sample_id"], str) or not item["sample_id"]:
        raise CollationError(f"Batch item {position} needs a nonempty sample_id.")
    counts = (
        item["original_length"],
        item["truncated_tokens"],
        item["unknown_count"],
        item["original_unknown_count"],
    )
    if (
        any(type(count) is not int or count < 0 for count in counts)
        or item["original_length"] != ids.numel() + item["truncated_tokens"]
        or type(item["truncated"]) is not bool
        or item["truncated"] != (item["truncated_tokens"] > 0)
        or item["unknown_count"] != int((ids == UNK_ID).sum())
        or not item["unknown_count"]
        <= item["original_unknown_count"]
        <= item["unknown_count"] + item["truncated_tokens"]
    ):
        raise CollationError(
            f"Batch item {position} has inconsistent encoding metadata."
        )


def collate_sentences(items: Sequence[SentenceItem]) -> SentenceBatch:
    """Pad to the longest item, preserving input order, IDs, and metadata.

    PAD=0 is masked out; UNK=1 remains a real token. Every row contains at least
    one real token. Inputs are never modified, sorted, or truncated here. All
    tensors are newly allocated on CPU; device transfer belongs to the caller.
    """
    if not items:
        raise CollationError("Cannot collate an empty batch.")
    for position, item in enumerate(items):
        _validate_item(item, position)
    ids = pad_sequence(
        [item["input_ids"] for item in items],
        batch_first=True,
        padding_value=PAD_ID,
        padding_side="right",
    )
    return {
        "input_ids": ids,
        "attention_mask": ids != PAD_ID,
        "labels": torch.stack([item["label"] for item in items]),
        "sample_ids": [item["sample_id"] for item in items],
        "lengths": torch.tensor(
            [item["input_ids"].numel() for item in items],
            dtype=torch.long,
            device="cpu",
        ),
        "original_lengths": torch.tensor(
            [item["original_length"] for item in items], dtype=torch.long, device="cpu"
        ),
        "truncated": torch.tensor(
            [item["truncated"] for item in items], dtype=torch.bool, device="cpu"
        ),
        "truncated_tokens": torch.tensor(
            [item["truncated_tokens"] for item in items], dtype=torch.long, device="cpu"
        ),
        "unknown_counts": torch.tensor(
            [item["unknown_count"] for item in items], dtype=torch.long, device="cpu"
        ),
        "original_unknown_counts": torch.tensor(
            [item["original_unknown_count"] for item in items],
            dtype=torch.long,
            device="cpu",
        ),
    }


def collation_recipe() -> dict[str, object]:
    """Describe the versioned batch contract for future training artifacts."""
    return {
        "version": COLLATION_VERSION,
        "padding_side": "right",
        "padding_length": "longest_sequence_in_batch",
        "padding_id": PAD_ID,
        "attention_mask": "true_for_real_tokens_including_unknown",
        "input_dtype": "int64",
        "mask_dtype": "bool",
        "device": "cpu",
        "empty_sequences": "reject",
    }
