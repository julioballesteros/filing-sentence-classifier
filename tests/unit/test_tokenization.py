"""Verify token boundaries and preservation of financial sentence information."""

import json

import pytest

from filing_sentence_classifier.text.tokenization import (
    TOKENIZATION_VERSION,
    tokenization_recipe,
    tokenize,
)


def test_forward_looking_sentence() -> None:
    assert tokenize("We don’t expect long-term growth of 12.5%.") == (
        "we",
        "don't",
        "expect",
        "long",
        "-",
        "term",
        "growth",
        "of",
        "12.5",
        "%",
        ".",
    )


def test_apostrophe_variants_share_tokens() -> None:
    expected = ("company's", "plans", "don't", "change", ".")
    assert tokenize("Company’s plans don’t change.") == expected
    assert tokenize("Company's plans don't change.") == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("'growth' shareholders’", ("'", "growth", "'", "shareholders", "'")),
        ("‘growth’ “outlook”", ("‘", "growth", "'", "“", "outlook", "”")),
        ("long-term 10-K", ("long", "-", "term", "10", "-", "k")),
        ("2026–2027 costs—benefits", ("2026", "–", "2027", "costs", "—", "benefits")),
        ("$1,250.50 (+12.5%).", ("$", "1,250.50", "(", "+", "12.5", "%", ")", ".")),
        ("1,000 12,345 123,456,789.01", ("1,000", "12,345", "123,456,789.01")),
        ("-0.50 007 2027.", ("-", "0.50", "007", "2027", ".")),
        (".5 5. 1.25.", (".", "5", "5", ".", "1.25", ".")),
        ("1,25 12,3456", ("1", ",", "25", "12", ",", "3456")),
        ("1,234,56 1,23,456", ("1", ",", "234", ",", "56", "1", ",", "23", ",", "456")),
        ("1,250, 2,500", ("1,250", ",", "2,500")),
        ("CAFÉ Straße Q3 401k", ("café", "straße", "q3", "401k")),
        (
            "S&P per_share U.S.",
            ("s", "&", "p", "per", "_", "share", "u", ".", "s", "."),
        ),
        ("\t We\n  will\u00a0expand.\r\n", ("we", "will", "expand", ".")),
        ("... $ € %", (".", ".", ".", "$", "€", "%")),
    ],
)
def test_boundaries(text: str, expected: tuple[str, ...]) -> None:
    assert tokenize(text) == expected


def test_function_words_and_inflections_are_preserved() -> None:
    assert tokenize("We may NOT grow and could be growing but will grow.") == (
        "we",
        "may",
        "not",
        "grow",
        "and",
        "could",
        "be",
        "growing",
        "but",
        "will",
        "grow",
        ".",
    )


@pytest.mark.parametrize("text", ["", " ", "\t\r\n", "\u00a0\u2003"])
def test_blank_input_is_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="non-whitespace"):
        tokenize(text)


@pytest.mark.parametrize("text", [None, 12, True, b"growth", ["growth"]])
def test_non_string_input_is_not_coerced(text: object) -> None:
    with pytest.raises(TypeError, match="must be a string"):
        tokenize(text)  # type: ignore[arg-type]


def test_long_input_preserves_order_and_all_non_whitespace_characters() -> None:
    sentence = "Café’s Q3 outlook: +$1,250.50, not €12—per_share! 📈 "
    text = sentence * 400
    tokens = tokenize(text)
    assert tokens == tokenize(sentence) * 400
    expected_characters = "".join(text.lower().replace("’", "'").split())
    assert "".join(tokens) == expected_characters
    assert tokenize(text) == tokens


def test_recipe_is_versioned_json_serializable_and_independent() -> None:
    recipe = tokenization_recipe()
    assert recipe["version"] == TOKENIZATION_VERSION == "1"
    restored = json.loads(json.dumps(recipe))
    assert restored == recipe
    recipe["version"] = "changed"
    operations = recipe["operations_in_order"]
    assert isinstance(operations, list)
    operations.clear()
    assert tokenization_recipe() == restored
