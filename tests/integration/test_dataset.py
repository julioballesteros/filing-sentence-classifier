"""Verify saved-partition integration and PyTorch worker compatibility."""

import json
from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from filing_sentence_classifier.data.dataset import SentenceDataset
from filing_sentence_classifier.data.loading import load_split
from filing_sentence_classifier.text.encoding import TextEncoder, encoding_statistics
from filing_sentence_classifier.text.tokenization import tokenize
from filing_sentence_classifier.text.vocabulary import Vocabulary


def test_dataset_uses_only_the_loaded_split_and_never_fits_or_reads_files(
    development_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = load_split(development_artifact, "train")
    vocabulary = Vocabulary.fit(tokenize(text) for text in train.texts)
    encoder = TextEncoder.from_dict(
        json.loads(json.dumps(TextEncoder(vocabulary).to_dict()))
    )
    val = load_split(
        development_artifact, "val", expected_manifest_sha256=train.manifest_sha256
    )
    expected = tuple(encoder.encode(text) for text in val.texts)
    before = encoder.to_dict()

    def forbidden(*args, **kwargs):
        pytest.fail("The Dataset must not read files or fit a vocabulary.")

    monkeypatch.setattr(Vocabulary, "fit", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    dataset = SentenceDataset(val, encoder)
    assert dataset.encodings == expected
    assert dataset.split.manifest_sha256 == train.manifest_sha256
    assert dataset.split.records_sha256 == val.records_sha256
    assert [item["sample_id"] for item in dataset] == list(val.sample_ids)
    assert [item["label"].item() for item in dataset] == list(val.targets)
    assert encoding_statistics(dataset.encodings) == encoding_statistics(expected)
    assert encoder.to_dict() == before


@pytest.mark.parametrize("workers", [0, 1])
def test_dataloader_preserves_order_and_values_with_spawn_workers(
    development_artifact: Path,
    workers: int,
) -> None:
    train = load_split(development_artifact, "train")
    encoder = TextEncoder(
        Vocabulary.fit(tokenize(text) for text in train.texts), max_length=4
    )
    dataset = SentenceDataset(train, encoder)
    # Disable automatic batching to test per-example transport in isolation.
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=workers,
        multiprocessing_context="spawn" if workers else None,
        timeout=20 if workers else 0,
    )
    items = list(loader)
    assert [item["sample_id"] for item in items] == list(train.sample_ids)
    assert [item["label"].item() for item in items] == list(train.targets)
    for index, item in enumerate(items):
        assert item["input_ids"].tolist() == list(dataset.encodings[index].input_ids)
        assert item["input_ids"].device.type == "cpu"
        assert item["truncated"] == dataset.encodings[index].truncated
