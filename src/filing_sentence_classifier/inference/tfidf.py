"""Inference from the fitted TF-IDF/logistic regression pipeline in a bundle."""

from dataclasses import dataclass, field
from io import BytesIO
from numbers import Integral
from pathlib import Path
from typing import Self

import numpy as np
from sklearn.feature_extraction.text import (  # type: ignore[import-untyped]
    TfidfTransformer,
)
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from filing_sentence_classifier.artifacts import (
    MAX_MANIFEST_BYTES,
    BundleError,
    BundleManifest,
    read_bundle_file,
)
from filing_sentence_classifier.baselines.config import TfidfConfig
from filing_sentence_classifier.baselines.tfidf import (
    create_tfidf_pipeline,
    load_tfidf_model,
)
from filing_sentence_classifier.inference.compatibility import check_environment
from filing_sentence_classifier.inference.contracts import (
    Prediction,
    PredictionContractError,
)


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> None:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != shape
        or value.dtype != np.float64
        or not np.isfinite(value).all()
    ):
        raise BundleError(f"Expected finite float64 {name} with shape {shape}.")


def _check_pipeline(
    model: Pipeline, config: TfidfConfig, num_classes: int
) -> tuple[int, ...]:
    """Validate recipe and fitted state; return columns in canonical class order."""
    reference = create_tfidf_pipeline(config)
    for name in reference.named_steps:
        if model.named_steps[name].get_params(deep=False) != reference.named_steps[
            name
        ].get_params(deep=False):
            raise BundleError(
                f"Saved {name} parameters do not match the configuration."
            )
    if model.memory is not None or model.transform_input is not None:
        raise BundleError(
            "Expected the plain TF-IDF pipeline without caching or routing."
        )
    vectorizer, classifier = model.named_steps["tfidf"], model.named_steps["logreg"]
    transformer = vectorizer._tfidf
    expected_transformer = TfidfTransformer(
        **{
            key: getattr(vectorizer, key)
            for key in ("norm", "use_idf", "smooth_idf", "sublinear_tf")
        }
    )
    if not isinstance(transformer, TfidfTransformer) or transformer.get_params(
        deep=False
    ) != expected_transformer.get_params(deep=False):
        raise BundleError("Fitted TF-IDF transformer does not match its vectorizer.")
    vocabulary = vectorizer.vocabulary_
    if (
        not isinstance(vocabulary, dict)
        or not vocabulary
        or any(
            not isinstance(token, str)
            or not token
            or isinstance(index, bool)
            or not isinstance(index, Integral)
            for token, index in vocabulary.items()
        )
        or set(vocabulary.values()) != set(range(len(vocabulary)))
    ):
        raise BundleError("Expected a nonempty vocabulary with contiguous feature IDs.")
    features = len(vocabulary)
    for estimator in (transformer, classifier):
        count = estimator.n_features_in_
        if (
            isinstance(count, bool)
            or not isinstance(count, Integral)
            or count != features
        ):
            raise BundleError("Fitted feature dimensions do not match the vocabulary.")
    classes = classifier.classes_
    if (
        not isinstance(classes, np.ndarray)
        or classes.shape != (num_classes,)
        or not np.issubdtype(classes.dtype, np.integer)
        or set(classes.tolist()) != set(range(num_classes))
    ):
        raise BundleError("Pipeline classes do not match the bundle's class IDs.")
    rows = 1 if num_classes == 2 else num_classes
    _finite_array(vectorizer.idf_, (features,), "IDF weights")
    if (vectorizer.idf_ < 1).any():
        raise BundleError("Smoothed IDF weights must be at least one.")
    _finite_array(classifier.coef_, (rows, features), "coefficients")
    _finite_array(classifier.intercept_, (rows,), "intercepts")
    return tuple(classes.tolist().index(label) for label in range(num_classes))


@dataclass(frozen=True)
class TfidfBackend:
    """Reuse a verified fitted pipeline without refitting, files, or neural tools."""

    manifest: BundleManifest
    _model: Pipeline = field(repr=False, compare=False)
    _class_columns: tuple[int, ...]

    @classmethod
    def from_bundle(cls, directory: Path, manifest: BundleManifest) -> Self:
        """Load a trusted joblib bundle using the recorded dependency versions."""
        if manifest.family != "tfidf_logreg":
            raise BundleError("Expected a tfidf_logreg bundle.")
        check_environment(manifest)
        try:
            if manifest.files["config.toml"].size_bytes > MAX_MANIFEST_BYTES:
                raise BundleError("Configuration exceeds the 1 MiB limit.")
            config = TfidfConfig.from_toml(
                read_bundle_file(directory, "config.toml", manifest=manifest)
            )
            content = read_bundle_file(directory, "model.joblib", manifest=manifest)
            model = load_tfidf_model(
                BytesIO(content), expected_sha256=manifest.files["model.joblib"].sha256
            )
            columns = _check_pipeline(model, config, len(manifest.labels.names))
            return cls(manifest, model, columns)
        except Exception as exc:
            # Joblib and malformed estimator state can raise different exception
            # types. Preserve their cause behind the public bundle-loading error.
            if isinstance(exc, BundleError):
                raise
            raise BundleError(f"Could not restore TF-IDF bundle: {exc}") from exc

    def predict(
        self, texts: tuple[str, ...], *, batch_size: int
    ) -> tuple[Prediction, ...]:
        """Transform prepared text and align predict_proba columns with class IDs."""
        results: list[Prediction] = []
        with threadpool_limits(limits=1):
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                probabilities = self._model.predict_proba(batch)
                if not isinstance(probabilities, np.ndarray) or probabilities.shape != (
                    len(batch),
                    len(self._class_columns),
                ):
                    raise PredictionContractError(
                        "Unexpected TF-IDF probability dimensions."
                    )
                results.extend(
                    Prediction(
                        self.manifest.model_id,
                        self.manifest.labels,
                        tuple(scores),
                        truncated=False,
                    )
                    for scores in probabilities[:, self._class_columns].tolist()
                )
        return tuple(results)
