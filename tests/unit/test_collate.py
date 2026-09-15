"""Check padding, masks, alignment, and the supervised batch boundary."""

import json

import pytest
import torch

from filing_sentence_classifier.data.collate import (
    CollationError,
    SentenceBatch,
    collate_sentences,
    collation_recipe,
)
from filing_sentence_classifier.data.dataset import SentenceItem


def item(
    ids: tuple[int, ...],
    *,
    sample_id: str = "example",
    label: int = 0,
    discarded: int = 0,
    discarded_unknown: int = 0,
) -> SentenceItem:
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long, device="cpu"),
        "label": torch.tensor(label, dtype=torch.long, device="cpu"),
        "sample_id": sample_id,
        "original_length": len(ids) + discarded,
        "truncated": discarded > 0,
        "truncated_tokens": discarded,
        "unknown_count": ids.count(1),
        "original_unknown_count": ids.count(1) + discarded_unknown,
    }


def test_dynamic_padding_preserves_order_and_all_metadata() -> None:
    batch = collate_sentences(
        [
            item((9,), sample_id="short"),
            item(
                (4, 1, 8), sample_id="long", label=2, discarded=2, discarded_unknown=1
            ),
            item((1, 1), sample_id="unknown", label=1),
        ]
    )
    assert batch["input_ids"].tolist() == [[9, 0, 0], [4, 1, 8], [1, 1, 0]]
    assert batch["attention_mask"].tolist() == [
        [True, False, False],
        [True, True, True],
        [True, True, False],
    ]
    assert batch["labels"].tolist() == [0, 2, 1]
    assert batch["sample_ids"] == ["short", "long", "unknown"]
    assert batch["lengths"].tolist() == [1, 3, 2]
    assert batch["original_lengths"].tolist() == [1, 5, 2]
    assert batch["truncated"].tolist() == [False, True, False]
    assert batch["truncated_tokens"].tolist() == [0, 2, 0]
    assert batch["unknown_counts"].tolist() == [0, 1, 2]
    assert batch["original_unknown_counts"].tolist() == [0, 2, 2]
    assert torch.equal(batch["attention_mask"].sum(dim=1), batch["lengths"])
    for name, value in batch.items():
        if isinstance(value, torch.Tensor):
            assert value.device.type == "cpu"
            assert not value.requires_grad
            expected_dtype = (
                torch.bool if name in {"attention_mask", "truncated"} else torch.long
            )
            assert value.dtype == expected_dtype
            assert value.shape == (
                (3, 3) if name in {"input_ids", "attention_mask"} else (3,)
            )


def test_single_unknown_token_keeps_both_batch_dimensions() -> None:
    batch = collate_sentences([item((1,))])
    assert batch["input_ids"].shape == batch["attention_mask"].shape == (1, 1)
    assert batch["attention_mask"].tolist() == [[True]]
    assert batch["labels"].shape == (1,)


def test_collator_does_not_truncate_or_mutate_items() -> None:
    original = item((2,) * 140)
    batch = collate_sentences([original, original])
    assert batch["input_ids"].shape == (2, 140)
    assert batch["sample_ids"] == ["example", "example"]
    batch["input_ids"][0].zero_()
    batch["labels"].fill_(99)
    batch["sample_ids"][0] = "changed"
    assert original["input_ids"].tolist() == [2] * 140
    assert original["label"].item() == 0
    assert original["sample_id"] == "example"
    assert batch["input_ids"][1].tolist() == [2] * 140


def test_masked_pooling_is_invariant_to_padding_and_other_batch_members() -> None:
    # Nonzero PAD embeddings make accidental inclusion of padding visible.
    embeddings = torch.arange(40, dtype=torch.float64).reshape(10, 4) + 1

    def mean(batch: SentenceBatch) -> torch.Tensor:
        mask = batch["attention_mask"].unsqueeze(-1)
        return (embeddings[batch["input_ids"]] * mask).sum(1) / mask.sum(1)

    short = item((1, 5))
    alone = collate_sentences([short])
    mixed = collate_sentences([item((8, 7, 4, 3)), short])
    assert torch.equal(mean(alone)[0], mean(mixed)[1])
    extended = dict(mixed)
    extended["input_ids"] = torch.nn.functional.pad(mixed["input_ids"], (0, 5), value=0)
    extended["attention_mask"] = torch.nn.functional.pad(
        mixed["attention_mask"], (0, 5), value=False
    )
    assert torch.equal(mean(mixed), mean(extended))


def test_cpu_contract_ignores_a_different_default_device() -> None:
    examples = [item((2,)), item((3, 4))]
    with torch.device("meta"):
        batch = collate_sentences(examples)
    assert all(
        value.device.type == "cpu"
        for value in batch.values()
        if isinstance(value, torch.Tensor)
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_ids", torch.tensor([], dtype=torch.long)),
        ("input_ids", torch.tensor([0, 2])),
        ("input_ids", torch.tensor([-1])),
        ("input_ids", torch.tensor([2.0])),
        ("input_ids", torch.tensor([[2]])),
        ("input_ids", torch.tensor([2], device="meta")),
        ("input_ids", [2]),
        ("label", torch.tensor([0])),
        ("label", torch.tensor(0.0)),
        ("label", torch.tensor(-1)),
        ("sample_id", ""),
        ("original_length", 0),
        ("original_length", True),
        ("truncated", True),
        ("truncated_tokens", 1),
        ("unknown_count", 1),
        ("original_unknown_count", 1),
    ],
)
def test_malformed_examples_are_rejected(field: str, value: object) -> None:
    example = item((2,))
    example[field] = value
    with pytest.raises(CollationError, match="Batch item 0"):
        collate_sentences([example])


def test_empty_batches_and_missing_fields_are_rejected() -> None:
    with pytest.raises(CollationError, match="empty batch"):
        collate_sentences([])
    with pytest.raises(CollationError, match="required sentence fields"):
        collate_sentences([{}])


def test_recipe_is_versioned_and_json_serializable() -> None:
    recipe = collation_recipe()
    assert recipe["version"] == "1"
    assert json.loads(json.dumps(recipe)) == recipe
    recipe["padding_id"] = 99
    assert collation_recipe()["padding_id"] == 0
