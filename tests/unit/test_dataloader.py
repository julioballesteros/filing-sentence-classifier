"""Verify deterministic epoch orders, complete coverage, and independent RNGs."""

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from filing_sentence_classifier.data.collate import SentenceBatch
from filing_sentence_classifier.data.dataloader import create_dataloader
from filing_sentence_classifier.data.dataset import SentenceDataset
from filing_sentence_classifier.data.loading import (
    LoadedSplit,
    SentenceRecord,
    SplitName,
)
from filing_sentence_classifier.text.encoding import TextEncoder
from filing_sentence_classifier.text.vocabulary import Vocabulary


@pytest.fixture
def dataset() -> SentenceDataset:
    names = ("specific", "historical", "generic")
    split = LoadedSplit(
        name=SplitName.TRAIN,
        records=tuple(
            SentenceRecord(
                f"s{index}",
                index,
                f"g{index}",
                "known " * (index % 4 + 1),
                index % 3,
                names[index % 3],
            )
            for index in range(11)
        ),
        label_ids=(0, 1, 2),
        label_names=names,
        manifest_sha256="a" * 64,
        records_sha256="b" * 64,
    )
    return SentenceDataset(
        split, TextEncoder(Vocabulary.fit([("known",)], min_frequency=1))
    )


def epoch_ids(loader: DataLoader[SentenceBatch]) -> list[str]:
    return [sample_id for batch in loader for sample_id in batch["sample_ids"]]


def test_same_seed_reproduces_changing_epochs_with_complete_coverage(
    dataset: SentenceDataset,
) -> None:
    first = create_dataloader(dataset, batch_size=4, shuffle=True, seed=2026)
    second = create_dataloader(dataset, batch_size=4, shuffle=True, seed=2026)
    first_epochs = [epoch_ids(first) for _ in range(2)]
    assert first_epochs == [epoch_ids(second) for _ in range(2)]
    assert first_epochs[0] != first_epochs[1]
    assert first_epochs[0] != epoch_ids(
        create_dataloader(dataset, batch_size=4, shuffle=True, seed=7)
    )
    for order in first_epochs:
        assert sorted(order) == sorted(dataset.split.sample_ids)
    batches = list(create_dataloader(dataset, batch_size=4, shuffle=True))
    assert [len(batch["sample_ids"]) for batch in batches] == [4, 4, 3]


def test_validation_iterations_do_not_affect_training_or_global_rng(
    dataset: SentenceDataset,
) -> None:
    global_state = torch.get_rng_state().clone()
    train = create_dataloader(dataset, batch_size=3, shuffle=True)
    reference = create_dataloader(dataset, batch_size=3, shuffle=True)
    validation = create_dataloader(dataset, batch_size=4)
    assert epoch_ids(train) == epoch_ids(reference)
    for _ in range(3):
        assert epoch_ids(validation) == list(dataset.split.sample_ids)
    assert epoch_ids(train) == epoch_ids(reference)
    assert torch.equal(torch.get_rng_state(), global_state)


def test_subset_and_single_example_batches_are_supported(
    dataset: SentenceDataset,
) -> None:
    loader = create_dataloader(Subset(dataset, [8, 1, 7]), batch_size=2)
    assert epoch_ids(loader) == ["s8", "s1", "s7"]
    batch = next(iter(create_dataloader(Subset(dataset, [0]))))
    assert batch["input_ids"].shape == batch["attention_mask"].shape == (1, 1)
    assert batch["labels"].shape == (1,)


@pytest.mark.parametrize(
    "options",
    [
        {"batch_size": 0},
        {"batch_size": True},
        {"batch_size": 1.5},
        {"num_workers": -1},
        {"num_workers": True},
        {"seed": -1},
        {"seed": 2**64},
        {"seed": True},
        {"shuffle": 1},
        {"pin_memory": 1},
    ],
)
def test_invalid_loader_configuration_is_rejected(
    dataset: SentenceDataset, options: dict
) -> None:
    with pytest.raises(ValueError):
        create_dataloader(dataset, **options)
