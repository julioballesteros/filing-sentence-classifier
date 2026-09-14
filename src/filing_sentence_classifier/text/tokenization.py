"""Deterministic word tokenization for the planned PyTorch classifier."""

import re

TOKENIZATION_VERSION = "1"

_TOKEN_PATTERN = re.compile(
    # Match English thousands groups before words, without accepting part of a
    # malformed comma-separated number such as 12,3456 or 1,234,56.
    r"(?<![\w,])[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?(?![0-9]|,[0-9])"
    r"|[0-9]+\.[0-9]+"
    # Unicode alphanumeric runs, with apostrophes only inside a word.
    r"|[^\W_]+(?:'[^\W_]+)*"
    # Keep every remaining non-whitespace character, including underscores.
    r"|\S"
)


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize prepared text using the fixed versioned rules.

    Lowercase with str.lower and map U+2019 to a straight apostrophe. Preserve
    internal apostrophes, Unicode alphanumeric words, and numeric values. ASCII
    numbers may contain English thousands groups or a decimal point with digits
    on both sides. Other punctuation and symbols each form one token; whitespace
    is discarded. There is no stop-word removal, fitting, or truncation.

    Input cleaning (including Unicode NFC and source encoding repairs) belongs
    upstream. Return an immutable sequence; reject non-string and blank inputs
    with TypeError and ValueError, respectively.
    """
    if not isinstance(text, str):
        raise TypeError("Text must be a string.")
    normalized = text.lower().replace("\u2019", "'")
    tokens = tuple(_TOKEN_PATTERN.findall(normalized))
    if not tokens:
        raise ValueError("Text must contain at least one non-whitespace character.")
    return tokens


def tokenization_recipe() -> dict[str, object]:
    """Return a fresh JSON-serializable description for training artifacts.

    Increment TOKENIZATION_VERSION whenever the tokenization rules change.
    """
    return {
        "version": TOKENIZATION_VERSION,
        "input": "prepared_text",
        "operations_in_order": ["str.lower", "u2019_to_ascii_apostrophe", "re.findall"],
        "pattern": _TOKEN_PATTERN.pattern,
        "regex_engine": "python_re_unicode",
        "numbers": "preserve_values_english_grouping_and_dot_decimals",
        "other_non_whitespace": "one_token_per_character",
        "empty_input": "raise_value_error",
        "stop_words": None,
        "stemming": False,
        "lemmatization": False,
        "truncation": None,
    }
