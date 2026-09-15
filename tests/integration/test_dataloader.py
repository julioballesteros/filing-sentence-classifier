"""Exercise batches over saved partitions, including ordered spawn workers."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from filing_sentence_classifier.data.collate import SentenceBatch
from filing_sentence_classifier.data.dataloader import create_dataloader
from filing_sentence_classifier.data.dataset import SentenceDataset
from filing_sentence_classifier.data.loading import load_split
from filing_sentence_classifier.text.encoding import TextEncoder
from filing_sentence_classifier.text.tokenization import tokenize
from filing_sentence_classifier.text.vocabulary import Vocabulary


@pytest.fixture
def datasets(development_artifact: Path) -> tuple[SentenceDataset, SentenceDataset]:
    train = load_split(development_artifact, "train")
    val = load_split(
        development_artifact, "val", expected_manifest_sha256=train.manifest_sha256
    )
    encoder = TextEncoder(
        Vocabulary.fit(tokenize(text) for text in train.texts), max_length=4
    )
    return SentenceDataset(train, encoder), SentenceDataset(val, encoder)


def snapshot(loader: DataLoader[SentenceBatch]) -> list[dict]:
    return [
        {
            key: value.tolist() if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        for batch in loader
    ]


def test_batches_keep_ids_labels_and_encoding_metadata_aligned_after_shuffle(
    datasets: tuple[SentenceDataset, SentenceDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, val = datasets

    def forbidden(*args, **kwargs):
        pytest.fail("Batching must not fit or encode again.")

    monkeypatch.setattr(Vocabulary, "fit", forbidden)
    monkeypatch.setattr(TextEncoder, "encode", forbidden)
    for dataset, shuffle in ((train, True), (val, False)):
        by_id = {
            record.sample_id: (record, encoded)
            for record, encoded in zip(
                dataset.split.records, dataset.encodings, strict=True
            )
        }
        order = []
        for batch in create_dataloader(dataset, batch_size=7, shuffle=shuffle):
            assert batch["input_ids"].shape == batch["attention_mask"].shape
            assert batch["input_ids"].shape[1] == int(batch["lengths"].max())
            assert torch.all(batch["attention_mask"].sum(1) > 0)
            for row, sample_id in enumerate(batch["sample_ids"]):
                record, encoded = by_id[sample_id]
                mask = batch["attention_mask"][row]
                assert batch["input_ids"][row][mask].tolist() == list(encoded.input_ids)
                assert torch.all(batch["input_ids"][row][~mask] == 0)
                assert batch["labels"][row].item() == record.label
                assert batch["truncated"][row].item() == encoded.truncated
                assert batch["original_lengths"][row].item() == encoded.original_length
                assert batch["unknown_counts"][row].item() == encoded.unknown_count
                order.append(sample_id)
        assert sorted(order) == sorted(dataset.split.sample_ids)
        if not shuffle:
            assert order == list(dataset.split.sample_ids)


@pytest.mark.parametrize("workers", [1, 2])
def test_worker_count_does_not_change_seeded_epoch_batches(
    datasets: tuple[SentenceDataset, SentenceDataset],
    workers: int,
) -> None:
    train, _ = datasets
    single = create_dataloader(train, batch_size=13, shuffle=True, seed=17)
    parallel = create_dataloader(
        train, batch_size=13, shuffle=True, seed=17, num_workers=workers
    )
    parallel.timeout = 20
    for _ in range(2):
        assert snapshot(parallel) == snapshot(single)
