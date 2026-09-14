"""Check the boundary between source cleaning and model tokenization."""

from filing_sentence_classifier.data.cleaning import clean_text
from filing_sentence_classifier.text.tokenization import tokenize


def test_cleaned_training_and_inference_text_share_the_same_tokens() -> None:
    raw_text = "  Cafe\u0301\u0092s\tlong-term outlook: $1,250.50.\n"
    prepared = clean_text(raw_text)
    assert prepared.text == "Café’s long-term outlook: $1,250.50."
    expected = ("café's", "long", "-", "term", "outlook", ":", "$", "1,250.50", ".")
    assert tokenize(prepared.text) == expected
    assert tokenize(clean_text(prepared.text).text) == expected
    assert prepared.text == "Café’s long-term outlook: $1,250.50."
