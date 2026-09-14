"""Check the complete prepared-text path and restoration without refitting."""

import json
from pathlib import Path

import pytest

from filing_sentence_classifier.data.cleaning import clean_text
from filing_sentence_classifier.data.loading import load_split
from filing_sentence_classifier.data.vocabulary import build_vocabulary
from filing_sentence_classifier.text.encoding import TextEncoder, encoding_statistics
from filing_sentence_classifier.text.tokenization import tokenize
from filing_sentence_classifier.text.vocabulary import UNK_ID, Vocabulary


def test_cleaning_and_restoration_give_the_same_training_and_inference_ids() -> None:
    raw = "  Cafe\u0301\u0092s\tlong-term outlook: $1,250.50.\n"
    prepared = clean_text(raw).text
    vocabulary = Vocabulary.fit([tokenize(prepared)], min_frequency=1)
    training_encoder = TextEncoder(vocabulary, max_length=5)
    inference_encoder = TextEncoder.from_dict(
        json.loads(json.dumps(training_encoder.to_dict()))
    )
    expected = training_encoder.encode(prepared)
    assert inference_encoder.encode(clean_text(raw).text) == expected
    assert expected.truncated
    assert expected.original_length == 9
    assert len(expected.input_ids) == 5
    assert expected.original_unknown_count == 0


def test_saved_train_vocabulary_encodes_validation_without_refitting(
    development_artifact: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = build_vocabulary(development_artifact, tmp_path / "vocabulary")
    vocabulary_path = artifact / "vocabulary.json"
    vocabulary_bytes = vocabulary_path.read_bytes()
    vocabulary = Vocabulary.from_dict(json.loads(vocabulary_bytes))

    def no_fitting(*args, **kwargs):
        pytest.fail("Encoding or restoration must never fit a vocabulary.")

    monkeypatch.setattr(Vocabulary, "fit", no_fitting)
    encoder = TextEncoder(vocabulary, max_length=3)
    saved = tmp_path / "encoder.json"
    saved.write_text(json.dumps(encoder.to_dict()), encoding="utf-8")
    restored = TextEncoder.from_dict(json.loads(saved.read_text(encoding="utf-8")))
    val = load_split(development_artifact, "val")
    encoded = tuple(restored.encode(text) for text in val.texts)
    assert encoded == tuple(encoder.encode(text) for text in val.texts)
    assert all(len(result.input_ids) == 3 and result.truncated for result in encoded)
    assert encoding_statistics(encoded)["rows"] == len(val.records)
    assert restored.encode("validationonly").input_ids == (UNK_ID,)
    assert "validationonly" not in vocabulary.token_to_id
    assert vocabulary_path.read_bytes() == vocabulary_bytes
    assert restored.to_dict() == encoder.to_dict()
