"""Check the tensor, identity, and cache contracts of sentence examples."""

import pickle
from dataclasses import replace

import pytest
import torch
from torch.utils.data import Dataset, Subset

from filing_sentence_classifier.data.dataset import DatasetError, SentenceDataset
from filing_sentence_classifier.data.loading import (
    LoadedSplit,
    SentenceRecord,
    SplitName,
)
from filing_sentence_classifier.text.encoding import TextEncoder
from filing_sentence_classifier.text.tokenization import tokenize
from filing_sentence_classifier.text.vocabulary import UNK_ID, Vocabulary


@pytest.fixture
def split() -> LoadedSplit:
    return LoadedSplit(
        name=SplitName.TRAIN,
        records=(
            SentenceRecord("first", 0, "g0", "Growth", 0, "specific"),
            SentenceRecord("second", 1, "g1", "We expect growth", 1, "historical"),
            SentenceRecord(
                "third", 2, "g2", "Unseen growth other suffix", 2, "generic"
            ),
        ),
        label_ids=(0, 1, 2),
        label_names=("specific", "historical", "generic"),
        manifest_sha256="a" * 64,
        records_sha256="b" * 64,
    )


@pytest.fixture
def encoder() -> TextEncoder:
    vocabulary = Vocabulary.fit([tokenize("We expect growth")], min_frequency=1)
    return TextEncoder(vocabulary, max_length=2)


def test_items_preserve_ids_labels_and_variable_length_cpu_tensors(
    split: LoadedSplit,
    encoder: TextEncoder,
) -> None:
    dataset = SentenceDataset(split, encoder)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == len(split.records)
    assert dataset.split is split
    assert dataset.encodings == tuple(encoder.encode(text) for text in split.texts)
    assert [len(dataset[index]["input_ids"]) for index in range(len(dataset))] == [
        1,
        2,
        2,
    ]
    for index, record in enumerate(split.records):
        item = dataset[index]
        expected = encoder.encode(record.text)
        assert item["input_ids"].tolist() == list(expected.input_ids)
        assert item["input_ids"].dtype == item["label"].dtype == torch.long
        assert item["input_ids"].device.type == item["label"].device.type == "cpu"
        assert item["input_ids"].ndim == 1
        assert item["label"].shape == torch.Size([])
        assert item["label"].item() == record.label
        assert not item["input_ids"].requires_grad
        assert not item["label"].requires_grad
        assert item["sample_id"] == record.sample_id
        assert item["original_length"] == expected.original_length
        assert item["truncated"] == expected.truncated
        assert item["truncated_tokens"] == expected.truncated_tokens
        assert item["unknown_count"] == expected.unknown_count
        assert item["original_unknown_count"] == expected.original_unknown_count
        assert torch.all(item["input_ids"] > 0)
    assert dataset[2]["input_ids"].tolist() == [UNK_ID, encoder.vocabulary["growth"]]


def test_mutating_returned_tensors_and_dicts_cannot_change_later_reads(
    split: LoadedSplit,
    encoder: TextEncoder,
) -> None:
    dataset = SentenceDataset(split, encoder)
    item = dataset[0]
    item["input_ids"].fill_(999)
    item["label"].fill_(99)
    item["sample_id"] = "changed"
    again = dataset[0]
    assert again["input_ids"].tolist() == [encoder.vocabulary["growth"]]
    assert again["label"].item() == 0
    assert again["sample_id"] == "first"
    assert dataset.encodings[0].input_ids == (encoder.vocabulary["growth"],)


def test_encoding_occurs_once_and_dataset_does_not_keep_the_encoder(
    split: LoadedSplit,
    encoder: TextEncoder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_encode = TextEncoder.encode
    calls = []

    def count_encode(self, text):
        calls.append(text)
        return original_encode(self, text)

    monkeypatch.setattr(TextEncoder, "encode", count_encode)
    dataset = SentenceDataset(split, encoder)
    assert calls == list(split.texts)

    def unexpected_encode(*args, **kwargs):
        pytest.fail("Dataset access must reuse cached encodings.")

    monkeypatch.setattr(TextEncoder, "encode", unexpected_encode)
    for _ in range(2):
        assert [item["sample_id"] for item in dataset] == list(split.sample_ids)
    # A retained encoder would contain a non-picklable read-only vocabulary map.
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.split == split
    assert restored.encodings == dataset.encodings
    assert restored[0]["input_ids"].tolist() == dataset[0]["input_ids"].tolist()


def test_indexing_and_subsets_preserve_sequence_order(
    split: LoadedSplit, encoder: TextEncoder
) -> None:
    dataset = SentenceDataset(split, encoder)
    assert dataset[-1]["sample_id"] == "third"
    assert [item["sample_id"] for item in Subset(dataset, [2, 0])] == ["third", "first"]
    for index in (len(dataset), -len(dataset) - 1):
        with pytest.raises(IndexError):
            dataset[index]
    for index in (True, 0.0, "0", slice(None)):
        with pytest.raises(TypeError):
            dataset[index]


def test_explicit_cpu_tensors_ignore_the_global_default_device(
    split: LoadedSplit,
    encoder: TextEncoder,
) -> None:
    dataset = SentenceDataset(split, encoder)
    with torch.device("meta"):
        item = dataset[0]
    assert item["input_ids"].device.type == item["label"].device.type == "cpu"
    assert item["label"].item() == 0


@pytest.mark.parametrize("label", [-1, 3, True, 1.0])
def test_invalid_labels_are_rejected_before_training(
    split: LoadedSplit, encoder: TextEncoder, label
) -> None:
    invalid = replace(split, records=(replace(split.records[0], label=label),))
    with pytest.raises(DatasetError, match="class label.*first"):
        SentenceDataset(invalid, encoder)


@pytest.mark.parametrize("labels", [(), (0, 2, 4), (0, True, 2), (2, 1, 0)])
def test_class_ids_must_be_model_output_indices(
    split: LoadedSplit, encoder: TextEncoder, labels
) -> None:
    with pytest.raises(DatasetError, match="Class IDs"):
        SentenceDataset(replace(split, label_ids=labels), encoder)


def test_invalid_records_have_actionable_errors(
    split: LoadedSplit, encoder: TextEncoder
) -> None:
    with pytest.raises(DatasetError, match="at least one sentence"):
        SentenceDataset(replace(split, records=()), encoder)
    with pytest.raises(DatasetError, match="unique nonempty ID"):
        SentenceDataset(
            replace(split, records=(split.records[0], split.records[0])), encoder
        )
    with pytest.raises(DatasetError, match="class label"):
        SentenceDataset(
            replace(split, records=(replace(split.records[0], label_text="wrong"),)),
            encoder,
        )
    with pytest.raises(DatasetError, match="first at index 0"):
        SentenceDataset(
            replace(split, records=(replace(split.records[0], text=" \t"),)), encoder
        )


def test_single_sentence_and_duplicate_texts_are_supported(
    split: LoadedSplit, encoder: TextEncoder
) -> None:
    single = SentenceDataset(replace(split, records=(split.records[0],)), encoder)
    assert len(single) == 1
    assert single[0]["input_ids"].shape == torch.Size([1])
    duplicate = replace(split.records[0], sample_id="duplicate", source_row=3)
    dataset = SentenceDataset(
        replace(split, records=(split.records[0], duplicate)), encoder
    )
    assert dataset[0]["input_ids"].tolist() == dataset[1]["input_ids"].tolist()
    assert dataset[0]["sample_id"] != dataset[1]["sample_id"]
