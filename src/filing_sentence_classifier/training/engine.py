"""Explicit supervised training over sentence batches, without run orchestration."""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from filing_sentence_classifier.data.collate import SentenceBatch


class TrainingError(RuntimeError):
    """An epoch cannot complete because its inputs or numerical state are invalid."""


@dataclass(frozen=True)
class EpochResult:
    """Detached training statistics; loss is averaged over processed examples."""

    mean_loss: float
    num_examples: int
    num_batches: int


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
    if any(parameter.device != device for parameter in model.parameters()):
        raise TrainingError(
            "Place the model on the training device before creating the optimizer."
        )
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise TrainingError(f"Non-finite parameter before training: {name}.")
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
            if (
                logits.ndim != 2
                or logits.shape[0] != labels.shape[0]
                or logits.shape[1] < 2
                or not logits.is_floating_point()
                or logits.device != target_device
            ):
                raise TrainingError(
                    f"Batch {batch_number}: expected floating logits [B, C] on the training device."
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
