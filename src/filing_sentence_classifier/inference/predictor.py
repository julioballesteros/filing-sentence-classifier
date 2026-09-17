"""Public prediction interface with lazy loading of model-family backends."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Self

from filing_sentence_classifier.artifacts import (
    BundleError,
    BundleManifest,
    verify_bundle,
)
from filing_sentence_classifier.inference.contracts import (
    Prediction,
    PredictionContractError,
    prepare_texts,
)


class _PreparedPredictor(Protocol):
    def predict(
        self, texts: tuple[str, ...], *, batch_size: int
    ) -> tuple[Prediction, ...]: ...


@dataclass(frozen=True)
class Predictor:
    """Load once with from_bundle, then reuse for ordered raw-text requests."""

    manifest: BundleManifest
    _backend: _PreparedPredictor = field(repr=False, compare=False)

    @classmethod
    def from_bundle(
        cls, directory: Path, *, expected_manifest_sha256: str | None = None
    ) -> Self:
        """Verify a bundle and restore its supported model family into memory."""
        manifest = verify_bundle(
            directory, expected_manifest_sha256=expected_manifest_sha256
        )
        if manifest.family == "mean_pool_mlp":
            try:
                from filing_sentence_classifier.inference.pytorch import PyTorchBackend
            except ModuleNotFoundError as exc:
                if exc.name != "torch":
                    raise
                raise BundleError(
                    "PyTorch is missing. Install the package with its [train] extra."
                ) from exc
            return cls(manifest, PyTorchBackend.from_bundle(directory, manifest))
        try:
            from filing_sentence_classifier.inference.tfidf import TfidfBackend
        except ModuleNotFoundError as exc:
            if (exc.name or "").split(".", 1)[0] not in {
                "sklearn",
                "joblib",
                "numpy",
                "scipy",
                "threadpoolctl",
            }:
                raise
            raise BundleError(
                "TF-IDF dependencies are missing. Install the package with its [data] extra."
            ) from exc
        return cls(manifest, TfidfBackend.from_bundle(directory, manifest))

    def predict(
        self, texts: Sequence[str], *, batch_size: int = 32
    ) -> tuple[Prediction, ...]:
        """Validate the entire request before inference; return no partial results.

        batch_size controls computation chunks independently of the bundle's
        maximum request size. Empty requests return (). Preserve order and
        duplicates, and apply the saved cleaning policy before model encoding.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise PredictionContractError("batch_size must be a positive integer.")
        prepared = prepare_texts(texts, limits=self.manifest.limits)
        if not prepared:
            return ()
        return self._backend.predict(prepared, batch_size=batch_size)
