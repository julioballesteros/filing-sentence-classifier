"""TF-IDF and multinomial logistic regression fitted as one scikit-learn pipeline."""

import hashlib
import warnings
from collections.abc import Sequence
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import joblib  # type: ignore[import-untyped]
from sklearn.exceptions import (  # type: ignore[import-untyped]
    ConvergenceWarning,
    InconsistentVersionWarning,
)
from sklearn.feature_extraction.text import (  # type: ignore[import-untyped]
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.utils.validation import check_is_fitted  # type: ignore[import-untyped]
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from filing_sentence_classifier.baselines.config import TfidfConfig as TfidfConfig
from filing_sentence_classifier.baselines.majority import BaselineError


def create_tfidf_pipeline(config: TfidfConfig) -> Pipeline:
    """Construct the shared pipeline recipe without fitting any component."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=config.ngram_range,
                    min_df=config.min_df,
                    sublinear_tf=True,
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    solver="lbfgs",
                    l1_ratio=0.0,
                    C=config.c,
                    max_iter=config.max_iter,
                    tol=config.tol,
                ),
            ),
        ]
    )


def fit_tfidf(
    texts: Sequence[str],
    targets: Sequence[int],
    *,
    label_ids: Sequence[int],
    config: TfidfConfig,
) -> Pipeline:
    """Fit vocabulary, IDF, and classifier using only the supplied training rows."""
    if (
        not texts
        or len(texts) != len(targets)
        or any(not isinstance(text, str) or not text.strip() for text in texts)
        or len(label_ids) < 2
        or len(set(label_ids)) != len(label_ids)
        or any(type(label) is not int for label in (*label_ids, *targets))
        or set(targets) != set(label_ids)
    ):
        raise BaselineError(
            "Expected aligned training rows containing every declared class."
        )
    pipeline = create_tfidf_pipeline(config)
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            pipeline.fit(texts, targets)
        except ConvergenceWarning as exc:
            raise BaselineError(
                "Logistic regression did not converge; no run published."
            ) from exc
    return pipeline


def predict_labels(pipeline: Pipeline, texts: Sequence[str]) -> tuple[int, ...]:
    """Return class IDs, not indices into the estimator's sorted classes_."""
    with threadpool_limits(limits=1):
        return tuple(int(label) for label in pipeline.predict(texts))


def load_tfidf_model(path: Path | BinaryIO, *, expected_sha256: str) -> Pipeline:
    """Load a trusted joblib Pipeline after checking its expected bytes.

    Joblib can execute code: checksums detect corruption, not an untrusted origin.
    Use the training environment; cross-version scikit-learn loads are rejected.
    Binary streams let inference restore previously verified bundle bytes.
    """
    content = path.read_bytes() if isinstance(path, Path) else path.read()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise BaselineError("TF-IDF model checksum mismatch.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        try:
            pipeline = joblib.load(BytesIO(content))
        except InconsistentVersionWarning as exc:
            raise BaselineError(
                "TF-IDF model requires its training scikit-learn version."
            ) from exc
    if (
        not isinstance(pipeline, Pipeline)
        or list(pipeline.named_steps) != ["tfidf", "logreg"]
        or not isinstance(pipeline.named_steps["tfidf"], TfidfVectorizer)
        or not isinstance(pipeline.named_steps["logreg"], LogisticRegression)
    ):
        raise BaselineError("Expected a TF-IDF/logistic regression pipeline.")
    check_is_fitted(pipeline.named_steps["tfidf"])
    check_is_fitted(pipeline.named_steps["logreg"])
    return pipeline


def save_tfidf_model(pipeline: Pipeline, path: Path) -> None:
    """Persist learned state without the vectorizer's process-local validation cache."""
    saved = deepcopy(pipeline)
    # sklearn caches id(stop_words), which is a memory address, not learned state.
    # Removing it triggers the normal consistency check on the next transform.
    saved.named_steps["tfidf"].__dict__.pop("_stop_words_id", None)
    joblib.dump(saved, path, compress=3, protocol=5)
