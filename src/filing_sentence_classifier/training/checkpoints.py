"""Atomic model checkpoints with validated metadata and strict state restoration."""

import math
import os
import pickle
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

import torch
from torch import Tensor, nn

from filing_sentence_classifier.training.config import TrainingConfig


class CheckpointError(RuntimeError):
    """A checkpoint cannot be saved or restored under the model contract."""


@dataclass(frozen=True)
class CheckpointMetadata:
    """Selection evidence and complete settings for the saved model weights."""

    config: TrainingConfig
    epoch: int
    macro_f1: float


def _model_type(model: nn.Module) -> str:
    return f"{type(model).__module__}.{type(model).__qualname__}"


def _check_selection(config: TrainingConfig, epoch: int, macro_f1: float) -> None:
    if type(epoch) is not int or not 1 <= epoch <= config.max_epochs:
        raise CheckpointError("Checkpoint epoch must be in [1, max_epochs].")
    if (
        type(macro_f1) not in (float, int)
        or not math.isfinite(macro_f1)
        or not 0 <= macro_f1 <= 1
    ):
        raise CheckpointError("Checkpoint macro-F1 must be finite and in [0, 1].")


def _tensor_state(value: object) -> dict[str, Tensor]:
    if not isinstance(value, dict) or not value:
        raise CheckpointError("Expected a nonempty model state dictionary.")
    for name, tensor in value.items():
        if (
            not isinstance(name, str)
            or not isinstance(tensor, Tensor)
            or tensor.layout != torch.strided
            or not bool(torch.isfinite(tensor).all())
        ):
            raise CheckpointError(
                "Model state must contain named, dense, finite tensors."
            )
    return cast(dict[str, Tensor], value)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    config: TrainingConfig,
    epoch: int,
    macro_f1: float,
    overwrite: bool = False,
) -> None:
    """Save detached CPU weights and settings, without optimizer or RNG state.

    A new file is published without replacing an existing path. Set overwrite
    only for later improvements within the same run. Write to a sibling temporary
    file first, so a failed save leaves the previous checkpoint intact.
    """
    _check_selection(config, epoch, macro_f1)
    # Preserve PyTorch's per-module state version metadata as well as the tensors.
    state = deepcopy(_tensor_state(model.state_dict()))
    for name, tensor in state.items():
        state[name] = tensor.detach().cpu()
    payload = {
        "schema_version": 1,
        "model_type": _model_type(model),
        "config": config.to_dict(),
        "epoch": epoch,
        "macro_f1": float(macro_f1),
        "model_state_dict": state,
    }
    temporary: Path | None = None
    try:
        if path.is_symlink():
            raise CheckpointError("Checkpoint path must not be a symbolic link.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Hard-link publication fails atomically if another run owns the path.
            os.link(temporary, path)
    except (OSError, RuntimeError) as exc:
        raise CheckpointError(f"Could not save checkpoint {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: Path | BinaryIO,
    model: nn.Module,
    *,
    expected_config: TrainingConfig | None = None,
) -> CheckpointMetadata:
    """Restore into a matching model and leave it in eval mode with no gradients.

    Load tensors on CPU with weights_only=True; only primitive metadata and a
    state_dict are accepted. Check type, keys, shapes, and dtypes before copying
    any parameter. The caller supplies the matching architecture and encoder.
    A binary stream allows inference to restore previously verified bytes.
    This restores a selected model, not a resumable optimizer/loader/RNG state.
    """
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise CheckpointError(f"Could not load checkpoint {path}: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "schema_version",
            "model_type",
            "config",
            "epoch",
            "macro_f1",
            "model_state_dict",
        }
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["model_type"] != _model_type(model)
    ):
        raise CheckpointError("Unsupported checkpoint schema or model type.")
    try:
        config = TrainingConfig.from_dict(payload["config"])
    except ValueError as exc:
        raise CheckpointError(f"Invalid checkpoint configuration: {exc}") from exc
    if expected_config is not None and config != expected_config:
        raise CheckpointError("Checkpoint configuration does not match this run.")
    _check_selection(config, payload["epoch"], payload["macro_f1"])
    state = _tensor_state(payload["model_state_dict"])
    current = model.state_dict()
    if set(current) != set(state) or any(
        current[name].shape != tensor.shape or current[name].dtype != tensor.dtype
        for name, tensor in state.items()
    ):
        raise CheckpointError(
            "Checkpoint keys, shapes, or dtypes do not match the model."
        )
    model.load_state_dict(state, strict=True)
    model.zero_grad(set_to_none=True)
    model.eval()
    return CheckpointMetadata(config, payload["epoch"], payload["macro_f1"])
