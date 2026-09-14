"""Check learned text features, class identity, convergence, and persistence."""

import hashlib
import math
from pathlib import Path

import joblib
import pytest
from scipy.sparse import issparse

from filing_sentence_classifier.baselines.majority import BaselineError
from filing_sentence_classifier.baselines.tfidf import (
    TfidfConfig,
    fit_tfidf,
    load_tfidf_model,
    predict_labels,
    save_tfidf_model,
)

TEXTS = (
    "Profit increased yesterday",
    "Revenue increased yesterday",
    "We expect growth",
    "We expect investment",
    "Results may differ",
    "Outcomes may differ",
)
TARGETS = (10, 10, 20, 20, 30, 30)
CONFIG = b"[tfidf]\nngram_range = [1, 2]\nmin_df = 2\n[logistic_regression]\nc = 1.0\nmax_iter = 1000\ntol = 0.0001\n"


def test_vocabulary_and_idf_are_fitted_only_on_training_rows() -> None:
    model = fit_tfidf(TEXTS, TARGETS, label_ids=(30, 10, 20), config=TfidfConfig())
    vectorizer = model.named_steps["tfidf"]
    before = (dict(vectorizer.vocabulary_), vectorizer.idf_.tolist())
    assert "we expect" in vectorizer.vocabulary_
    assert "may" in vectorizer.vocabulary_
    assert "profit" not in vectorizer.vocabulary_  # min_df=2
    assert vectorizer.idf_[vectorizer.vocabulary_["expect"]] == pytest.approx(
        math.log(7 / 3) + 1
    )
    assert predict_labels(model, TEXTS) == TARGETS
    held_out = ("unseenvalidationtoken", "WE expect unseenvalidationtoken")
    transformed = vectorizer.transform(held_out)
    assert issparse(transformed)
    assert transformed[0].nnz == 0
    assert len(predict_labels(model, held_out)) == 2
    assert (vectorizer.vocabulary_, vectorizer.idf_.tolist()) == before
    assert "unseenvalidationtoken" not in vectorizer.vocabulary_


def test_config_is_strict_and_matches_the_reference_recipe() -> None:
    assert TfidfConfig.from_toml(CONFIG) == TfidfConfig()
    for bad in (
        CONFIG.replace(b"min_df", b"min_dff"),
        CONFIG.replace(b"[1, 2]", b"[2, 1]"),
        CONFIG.replace(b"min_df = 2", b"min_df = true"),
        CONFIG.replace(b"c = 1.0", b"c = nan"),
        CONFIG.replace(b"c = 1.0", b"c = 0"),
        CONFIG.replace(b"max_iter = 1000", b"max_iter = 0"),
        CONFIG.replace(b"tol = 0.0001", b"tol = inf"),
        CONFIG + b"typo = 1\n",
    ):
        with pytest.raises(ValueError):
            TfidfConfig.from_toml(bad)


@pytest.mark.parametrize(
    "case",
    ["empty", "length", "missing_class", "bool_label", "blank", "empty_vocabulary"],
)
def test_invalid_training_data_is_rejected(case: str) -> None:
    texts, targets, labels = list(TEXTS), list(TARGETS), [10, 20, 30]
    if case == "empty":
        texts, targets = [], []
    elif case == "length":
        targets.pop()
    elif case == "missing_class":
        labels.append(40)
    elif case == "bool_label":
        targets[0] = True
    elif case == "blank":
        texts[0] = "  "
    else:
        texts = ["a !"] * len(texts)
    with pytest.raises(ValueError):
        fit_tfidf(texts, targets, label_ids=labels, config=TfidfConfig())


def test_nonconvergence_is_an_error() -> None:
    with pytest.raises(BaselineError, match="did not converge"):
        fit_tfidf(
            TEXTS, TARGETS, label_ids=(10, 20, 30), config=TfidfConfig(max_iter=1)
        )


def test_joblib_round_trip_retains_vocabulary_idf_and_coefficients(
    tmp_path: Path,
) -> None:
    model = fit_tfidf(TEXTS, TARGETS, label_ids=(10, 20, 30), config=TfidfConfig())
    path = tmp_path / "model.joblib"
    save_tfidf_model(model, path)
    original_bytes = path.read_bytes()
    model.named_steps["tfidf"]._stop_words_id = 123
    save_tfidf_model(model, path)
    assert path.read_bytes() == original_bytes
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    restored = load_tfidf_model(path, expected_sha256=checksum)
    assert predict_labels(restored, TEXTS) == TARGETS
    assert (
        restored.named_steps["tfidf"].vocabulary_
        == model.named_steps["tfidf"].vocabulary_
    )
    assert (
        restored.named_steps["tfidf"].idf_.tolist()
        == model.named_steps["tfidf"].idf_.tolist()
    )
    assert (
        restored.named_steps["logreg"].coef_.tolist()
        == model.named_steps["logreg"].coef_.tolist()
    )
    with pytest.raises(BaselineError, match="checksum"):
        load_tfidf_model(path, expected_sha256="0" * 64)
    joblib.dump({"not": "a pipeline"}, path)
    with pytest.raises(BaselineError, match="Expected a TF-IDF"):
        load_tfidf_model(
            path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
