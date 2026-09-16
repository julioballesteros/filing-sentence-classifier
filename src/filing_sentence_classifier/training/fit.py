"""Coordinate epochs, early stopping, and restoration of the selected model."""

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from torch import nn

from filing_sentence_classifier.data.collate import SentenceBatch
from filing_sentence_classifier.evaluation.metrics import ClassificationMetrics
from filing_sentence_classifier.training.checkpoints import (
    load_checkpoint,
    save_checkpoint,
)
from filing_sentence_classifier.training.config import TrainingConfig
from filing_sentence_classifier.training.engine import (
    EpochResult,
    TrainingError,
    ValidationResult,
    train_epoch,
    validate_epoch,
)
from filing_sentence_classifier.training.optimizers import create_optimizer


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    training: EpochResult
    validation_mean_loss: float
    validation_metrics: ClassificationMetrics


@dataclass(frozen=True)
class FitResult:
    """Aggregate history and validation predictions from the restored best model."""

    history: tuple[EpochMetrics, ...]
    best_epoch: int
    best_validation: ValidationResult
    stopped_early: bool
    checkpoint_path: Path


def fit(
    model: nn.Module,
    train_batches: Iterable[SentenceBatch],
    val_batches: Iterable[SentenceBatch],
    *,
    config: TrainingConfig,
    label_ids: Iterable[int],
    validation_sample_ids: Iterable[str],
    checkpoint_path: Path,
) -> FitResult:
    """Fit an initialized model with one AdamW instance and reusable batch sources.

    The caller seeds and constructs a model matching config, places it on the
    configured device, and supplies fixed train/validation data. No data loading,
    preprocessing, reseeding, or test access occurs here. One-shot iterators are
    rejected; DataLoaders and other re-iterable batch sources are supported.

    Select the highest unrounded validation macro-F1; exact ties keep the earliest
    epoch. Independently reset patience only when the score exceeds its reference
    by more than min_delta. Small gains can accumulate from that reference. Stop
    after patience consecutive epochs without such a gain, or at max_epochs.

    The checkpoint path must be new for this run. Each best epoch is saved before
    checking patience. On success restore its weights and re-evaluate validation;
    the model is left in eval mode with cleared gradients. Numerical or I/O errors
    propagate and leave any previously saved best checkpoint available.
    """
    if isinstance(train_batches, Iterator) or isinstance(val_batches, Iterator):
        raise TrainingError(
            "fit requires reusable batch sources, not one-shot iterators."
        )
    if checkpoint_path.exists() or checkpoint_path.is_symlink():
        raise TrainingError("Use a new checkpoint path for each training run.")
    classes, sample_ids = tuple(label_ids), tuple(validation_sample_ids)
    optimizer = create_optimizer(model, config)
    history = []
    best_score = patience_reference = -math.inf
    best_epoch = bad_epochs = 0
    stopped_early = False

    for epoch in range(1, config.max_epochs + 1):
        training = train_epoch(
            model, train_batches, optimizer, device=config.runtime.device
        )
        validation = validate_epoch(
            model,
            val_batches,
            label_ids=classes,
            expected_sample_ids=sample_ids,
            device=config.runtime.device,
        )
        score = validation.metrics.macro_f1
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise TrainingError("Validation macro-F1 must be finite and in [0, 1].")
        history.append(
            EpochMetrics(epoch, training, validation.mean_loss, validation.metrics)
        )
        if score > best_score:
            save_checkpoint(
                checkpoint_path,
                model,
                config=config,
                epoch=epoch,
                macro_f1=score,
                overwrite=best_epoch > 0,
            )
            best_score, best_epoch = score, epoch
        if score > patience_reference + config.min_delta:
            patience_reference, bad_epochs = score, 0
        else:
            bad_epochs += 1
        if bad_epochs >= config.patience:
            stopped_early = epoch < config.max_epochs
            break

    load_checkpoint(checkpoint_path, model, expected_config=config)
    restored = validate_epoch(
        model,
        val_batches,
        label_ids=classes,
        expected_sample_ids=sample_ids,
        device=config.runtime.device,
    )
    return FitResult(
        tuple(history), best_epoch, restored, stopped_early, checkpoint_path
    )
