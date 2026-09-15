"""Explicit process-wide initialization for reproducible CPU experiments."""

import random
from dataclasses import asdict
from importlib import import_module
from types import ModuleType

import torch
import torch.utils.deterministic

from filing_sentence_classifier.training.config import RuntimeConfig


def configure_runtime(config: RuntimeConfig) -> dict[str, object]:
    """Apply the CPU/float32 profile and seed before constructing model/loaders.

    Call once at the start of each run, not each epoch. This changes global Python,
    NumPy (if installed), and PyTorch RNGs, plus PyTorch device, dtype, thread, and
    determinism settings. Pass config.seed separately to each create_dataloader;
    their independent generators are not reset by this function.

    NumPy Generator instances and random.Random instances need their own explicit
    seeds. Python hash randomization is decided at process startup and is not
    modified here. Exact repetition requires the same environment and execution
    schedule; this is not a checkpoint-resumption API.
    """
    if not isinstance(config, RuntimeConfig):
        raise TypeError("Expected a validated RuntimeConfig.")
    numpy: ModuleType | None
    try:
        numpy = import_module("numpy")
    except ModuleNotFoundError as exc:
        if exc.name != "numpy":
            raise
        numpy = None

    torch.set_num_threads(config.num_threads)
    torch.set_default_device(config.device)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True, warn_only=False)
    # PyTorch exposes this setting through a dynamic module property.
    torch.utils.deterministic.fill_uninitialized_memory = True  # type: ignore[attr-defined]
    random.seed(config.seed)
    if numpy is not None:
        numpy.random.seed(config.seed)
    torch.manual_seed(config.seed)

    return {
        **asdict(config),
        "dtype": "float32",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,  # type: ignore[attr-defined]
        "num_interop_threads": torch.get_num_interop_threads(),
        "numpy_seeded": numpy is not None,
    }
