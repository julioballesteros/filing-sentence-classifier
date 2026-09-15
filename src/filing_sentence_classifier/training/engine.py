"""Explicit training and validation epochs, without run orchestration."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from filing_sentence_classifier.data.collate import SentenceBatch

if TYPE_CHECKING:
    from filing_sentence_classifier.evaluation.metrics import ClassificationMetrics


class TrainingError(RuntimeError):
    """An epoch cannot complete because its inputs or numerical state are invalid."""


@dataclass(frozen=True)
class EpochResult:
    """Detached training statistics; loss is averaged over processed examples."""

    mean_loss: float
    num_examples: int
    num_batches: int


@dataclass(frozen=True)
class ValidationResult(EpochResult):
    """Whole-partition metrics and aligned Python tuples in batch traversal order."""

    metrics: ClassificationMetrics
    sample_ids: tuple[str, ...]
    targets: tuple[int, ...]
    predicted_labels: tuple[int, ...]


def _check_model(model: nn.Module, device: torch.device, phase: str) -> None:
    for name, value in chain(model.named_parameters(), model.named_buffers()):
        if value.device != device:
            raise TrainingError(
                f"Place the model on the {phase} device before running the epoch."
            )
        if not bool(torch.isfinite(value).all()):
            raise TrainingError(
                f"Non-finite parameter or buffer before {phase}: {name}."
            )


def _training_parameters(
    model: nn.Module, optimizer: Optimizer, device: torch.device
) -> list[tuple[str, nn.Parameter]]:
    parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    optimized = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    if (
        not parameters
        or len(optimized) != len(parameters)
        or {id(parameter) for parameter in optimized}
        != {id(parameter) for _, parameter in parameters}
    ):
        raise TrainingError(
            "Optimizer must own exactly the model's trainable parameters."
        )
    _check_model(model, device, "training")
    return parameters


def _batch_tensors(
    batch: SentenceBatch, device: torch.device, batch_number: int
) -> tuple[Tensor, Tensor, Tensor]:
    ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    if (
        ids.ndim != 2
        or mask.shape != ids.shape
        or labels.dtype != torch.long
        or labels.ndim != 1
        or labels.numel() == 0
        or labels.shape[0] != ids.shape[0]
        or len(batch["sample_ids"]) != labels.shape[0]
    ):
        raise TrainingError(
            f"Batch {batch_number}: expected aligned inputs, long labels, and sample IDs."
        )
    # Identity and diagnostic metadata stay on CPU; only model inputs/targets move.
    return ids.to(device), mask.to(device), labels.to(device)


def _checked_loss(
    logits: Tensor,
    labels: Tensor,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    batch_number: int,
    *,
    num_classes: int | None = None,
) -> Tensor:
    if (
        logits.ndim != 2
        or logits.shape[0] != labels.shape[0]
        or logits.shape[1] < 2
        or not logits.is_floating_point()
        or logits.device != device
    ):
        raise TrainingError(
            f"Batch {batch_number}: expected floating logits [B, C] on the epoch device."
        )
    if num_classes is not None and logits.shape[1] != num_classes:
        raise TrainingError(
            f"Batch {batch_number}: logit columns must match declared classes."
        )
    if not bool(torch.isfinite(logits).all()):
        raise TrainingError(f"Batch {batch_number}: non-finite logits.")
    if bool(((labels < 0) | (labels >= logits.shape[1])).any()):
        raise TrainingError(
            f"Batch {batch_number}: labels must be class indices in [0, C)."
        )
    loss = cast(Tensor, criterion(logits, labels))
    if not bool(torch.isfinite(loss)):
        raise TrainingError(f"Batch {batch_number}: non-finite loss.")
    return loss


def train_epoch(
    model: nn.Module,
    batches: Iterable[SentenceBatch],
    optimizer: Optimizer,
    *,
    device: str | torch.device = "cpu",
) -> EpochResult:
    """Update the model once per batch and return example-weighted mean loss.

    Sets train mode and enables autograd, but does not construct or reset the model,
    optimizer, loaders, or RNGs. Inputs follow the sentence collator's contract.
    Targets must be class indices; no class weighting, smoothing, or ignored rows
    are used. Losses are observed before each update, not on the final model.

    Numerical errors abort the epoch instead of skipping batches. Earlier updates
    are not rolled back, including a step that produced non-finite parameters.
    No files, validation metrics, or checkpoints are produced here.
    """
    if torch.is_inference_mode_enabled():
        raise TrainingError("Training cannot run inside torch.inference_mode().")
    target_device = torch.device(device)
    parameters = _training_parameters(model, optimizer, target_device)
    criterion = nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.0)
    total_loss = 0.0
    num_examples = num_batches = 0
    model.train()

    with torch.enable_grad():
        for batch_number, batch in enumerate(batches, start=1):
            input_ids, attention_mask, labels = _batch_tensors(
                batch, target_device, batch_number
            )
            optimizer.zero_grad(set_to_none=True)
            logits = cast(Tensor, model(input_ids, attention_mask))
            loss = _checked_loss(logits, labels, criterion, target_device, batch_number)
            if not loss.requires_grad:
                raise TrainingError(
                    f"Batch {batch_number}: loss is disconnected from autograd."
                )
            loss.backward()  # type: ignore[no-untyped-call]
            for name, parameter in parameters:
                if parameter.grad is None or not bool(
                    torch.isfinite(parameter.grad).all()
                ):
                    raise TrainingError(
                        f"Batch {batch_number}: missing or non-finite gradient for {name}."
                    )
            optimizer.step()
            for name, parameter in parameters:
                if not bool(torch.isfinite(parameter).all()):
                    raise TrainingError(
                        f"Batch {batch_number}: non-finite parameter after optimizer step: {name}."
                    )

            batch_size = labels.shape[0]
            total_loss += loss.detach().item() * batch_size
            num_examples += batch_size
            num_batches += 1

    if num_examples == 0:
        raise TrainingError("Cannot train on an empty batch iterable.")
    mean_loss = total_loss / num_examples
    if not math.isfinite(mean_loss):
        raise TrainingError("Non-finite epoch loss.")
    return EpochResult(mean_loss, num_examples, num_batches)


def validate_epoch(
    model: nn.Module,
    batches: Iterable[SentenceBatch],
    *,
    label_ids: Iterable[int],
    expected_sample_ids: Iterable[str],
    device: str | torch.device = "cpu",
) -> ValidationResult:
    """Evaluate each expected sample once, with eval mode and no autograd.

    Class IDs must be the zero-based logit column order. Predictions use argmax;
    ties select the lowest class ID. Compute shared metrics once on the complete
    partition and weight mean cross-entropy by examples, including partial batches.
    Returned IDs, targets, and predictions follow batch order and contain no tensors.

    Leaves the model in eval mode; train_epoch explicitly re-enters train mode.
    Existing gradients are neither cleared nor changed. No optimizer, fitting,
    file access, or reseeding is involved. The data extra supplies shared metrics;
    importing this module and running train_epoch still require only the train extra.
    """
    # Keep scikit-learn optional for callers that only use the training epoch.
    from filing_sentence_classifier.evaluation.metrics import classification_metrics

    classes = tuple(label_ids)
    if (
        len(classes) < 2
        or any(type(label) is not int for label in classes)
        or classes != tuple(range(len(classes)))
    ):
        raise TrainingError(
            "Validation class IDs must be contiguous from zero in logit order."
        )
    expected_ids = tuple(expected_sample_ids)
    if (
        not expected_ids
        or any(
            not isinstance(sample_id, str) or not sample_id
            for sample_id in expected_ids
        )
        or len(set(expected_ids)) != len(expected_ids)
    ):
        raise TrainingError(
            "Expected validation sample IDs must be nonempty strings and unique."
        )
    expected = set(expected_ids)
    target_device = torch.device(device)
    _check_model(model, target_device, "validation")
    criterion = nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.0)
    sample_ids: list[str] = []
    targets: list[int] = []
    predicted_labels: list[int] = []
    seen: set[str] = set()
    total_loss = 0.0
    num_batches = 0
    model.eval()

    with torch.inference_mode():
        for batch_number, batch in enumerate(batches, start=1):
            input_ids, attention_mask, labels = _batch_tensors(
                batch, target_device, batch_number
            )
            for sample_id in batch["sample_ids"]:
                if not isinstance(sample_id, str) or not sample_id:
                    raise TrainingError(
                        f"Batch {batch_number}: invalid validation sample ID."
                    )
                if sample_id in seen:
                    raise TrainingError(
                        f"Batch {batch_number}: duplicate validation sample ID: {sample_id}."
                    )
                if sample_id not in expected:
                    raise TrainingError(
                        f"Batch {batch_number}: unexpected validation sample ID: {sample_id}."
                    )
                seen.add(sample_id)
            logits = cast(Tensor, model(input_ids, attention_mask))
            loss = _checked_loss(
                logits,
                labels,
                criterion,
                target_device,
                batch_number,
                num_classes=len(classes),
            )
            total_loss += loss.item() * labels.shape[0]
            num_batches += 1
            sample_ids.extend(batch["sample_ids"])
            targets.extend(labels.tolist())
            predicted_labels.extend(logits.argmax(dim=1).tolist())

    if not sample_ids:
        raise TrainingError("Cannot validate on an empty batch iterable.")
    if seen != expected:
        raise TrainingError(
            f"Validation is missing {len(expected - seen)} expected sample IDs."
        )
    mean_loss = total_loss / len(sample_ids)
    if not math.isfinite(mean_loss):
        raise TrainingError("Non-finite validation epoch loss.")
    metrics = classification_metrics(targets, predicted_labels, label_ids=classes)
    return ValidationResult(
        mean_loss,
        len(sample_ids),
        num_batches,
        metrics,
        tuple(sample_ids),
        tuple(targets),
        tuple(predicted_labels),
    )
