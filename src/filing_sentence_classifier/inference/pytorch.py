"""CPU float32 inference with the saved MeanPoolMLP and text encoder."""

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Self

import torch
from torch.nn.utils.rnn import pad_sequence

from filing_sentence_classifier.artifacts import (
    MAX_MANIFEST_BYTES,
    BundleError,
    BundleManifest,
    _parse_json_object,
    read_bundle_file,
)
from filing_sentence_classifier.inference.compatibility import check_environment
from filing_sentence_classifier.inference.contracts import Prediction
from filing_sentence_classifier.models.mean_pool_mlp import MeanPoolMLP
from filing_sentence_classifier.text.encoding import TextEncoder
from filing_sentence_classifier.text.vocabulary import PAD_ID
from filing_sentence_classifier.training.checkpoints import load_checkpoint
from filing_sentence_classifier.training.config import TrainingConfig


@dataclass(frozen=True)
class PyTorchBackend:
    """A restored model and encoder; predictions require no further file access."""

    manifest: BundleManifest
    _encoder: TextEncoder = field(repr=False)
    _model: MeanPoolMLP = field(repr=False, compare=False)

    @classmethod
    def from_bundle(cls, directory: Path, manifest: BundleManifest) -> Self:
        """Restore verified bytes, checking config, vocabulary and class dimensions."""
        if manifest.family != "mean_pool_mlp":
            raise BundleError("Expected a mean_pool_mlp bundle.")
        check_environment(manifest)
        try:
            for name in ("config.toml", "encoder.json"):
                if manifest.files[name].size_bytes > MAX_MANIFEST_BYTES:
                    raise BundleError(f"Metadata exceeds the 1 MiB limit: {name}.")
            config = TrainingConfig.from_toml(
                read_bundle_file(directory, "config.toml", manifest=manifest)
            )
            encoder = TextEncoder.from_dict(
                _parse_json_object(
                    read_bundle_file(directory, "encoder.json", manifest=manifest)
                )
            )
            checkpoint = read_bundle_file(directory, "model.pt", manifest=manifest)
            # Meta construction avoids random initialization and RNG changes.
            # Allocate CPU float32 storage only for the weights being restored.
            with torch.device("meta"):
                model = MeanPoolMLP(
                    len(encoder.vocabulary),
                    embedding_dim=config.model.embedding_dim,
                    hidden_dim=config.model.hidden_dim,
                    dropout=config.model.dropout,
                    num_classes=len(manifest.labels.names),
                ).to(dtype=torch.float32)
            model.to_empty(device="cpu")
            load_checkpoint(BytesIO(checkpoint), model, expected_config=config)
            model.requires_grad_(False)
            return cls(manifest, encoder, model)
        except (OSError, ValueError, RuntimeError) as exc:
            if isinstance(exc, BundleError):
                raise
            raise BundleError(f"Could not restore PyTorch bundle: {exc}") from exc

    def predict(
        self, texts: tuple[str, ...], *, batch_size: int
    ) -> tuple[Prediction, ...]:
        """Encode prepared text, pad on the right, and apply softmax in class order."""
        encodings = tuple(self._encoder.encode(text) for text in texts)
        results: list[Prediction] = []
        with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
            for start in range(0, len(encodings), batch_size):
                batch = encodings[start : start + batch_size]
                ids = pad_sequence(
                    [
                        torch.tensor(row.input_ids, dtype=torch.long, device="cpu")
                        for row in batch
                    ],
                    batch_first=True,
                    padding_value=PAD_ID,
                    padding_side="right",
                )
                probabilities = self._model(ids, ids != PAD_ID).softmax(dim=1).tolist()
                results.extend(
                    Prediction(
                        self.manifest.model_id,
                        self.manifest.labels,
                        tuple(scores),
                        truncated=row.truncated,
                    )
                    for row, scores in zip(batch, probabilities, strict=True)
                )
        return tuple(results)
