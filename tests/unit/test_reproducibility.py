"""Verify explicit RNG initialization and observable CPU runtime settings."""

import json
import os
import random

import numpy as np
import pytest
import torch

from filing_sentence_classifier.training import reproducibility
from filing_sentence_classifier.training.config import RuntimeConfig, TrainingConfig
from filing_sentence_classifier.training.reproducibility import configure_runtime


@pytest.fixture(autouse=True)
def restore_process_settings():
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    device, dtype = torch.get_default_device(), torch.get_default_dtype()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    yield
    random.setstate(python_rng)
    np.random.set_state(numpy_rng)
    torch.set_rng_state(torch_rng)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.utils.deterministic.fill_uninitialized_memory = fill


def draws() -> tuple:
    return random.random(), np.random.random(4).tolist(), torch.rand(4).tolist()


def test_same_seed_resets_global_rngs_but_different_seed_changes_draws() -> None:
    config = RuntimeConfig()
    configure_runtime(config)
    first = draws()
    assert draws() != first
    configure_runtime(config)
    assert draws() == first
    configure_runtime(RuntimeConfig(seed=29))
    assert draws() != first


def test_runtime_applies_cpu_float32_threads_and_strict_determinism() -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("meta")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    metadata = configure_runtime(RuntimeConfig(num_threads=2))
    assert torch.empty(1).device.type == "cpu"
    assert torch.empty(1).dtype == torch.float32
    assert torch.get_num_threads() == 2
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.utils.deterministic.fill_uninitialized_memory
    assert metadata == {
        "seed": 17,
        "device": "cpu",
        "num_workers": 0,
        "num_threads": 2,
        "dtype": "float32",
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "fill_uninitialized_memory": True,
        "num_interop_threads": torch.get_num_interop_threads(),
        "numpy_seeded": True,
    }
    assert json.loads(json.dumps(metadata)) == metadata


def test_parsing_configuration_has_no_runtime_side_effects() -> None:
    configure_runtime(RuntimeConfig())
    state = torch.get_rng_state().clone()
    python_state = random.getstate()
    TrainingConfig.from_dict(TrainingConfig().to_dict())
    assert torch.equal(torch.get_rng_state(), state)
    assert random.getstate() == python_state


def test_optional_numpy_and_startup_hash_seed(monkeypatch) -> None:
    def missing_numpy(name):
        assert name == "numpy"
        raise ModuleNotFoundError("No NumPy", name="numpy")

    monkeypatch.setattr(reproducibility, "import_module", missing_numpy)
    monkeypatch.setenv("PYTHONHASHSEED", "123")
    metadata = configure_runtime(RuntimeConfig(seed=2**32 - 1))
    assert not metadata["numpy_seeded"]
    assert torch.initial_seed() == 2**32 - 1
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_broken_numpy_installations_are_not_silently_ignored(monkeypatch) -> None:
    def broken_numpy(name):
        raise ModuleNotFoundError("Missing internal dependency", name="numpy.internal")

    monkeypatch.setattr(reproducibility, "import_module", broken_numpy)
    state = torch.get_rng_state().clone()
    with pytest.raises(ModuleNotFoundError, match="internal dependency"):
        configure_runtime(RuntimeConfig())
    assert torch.equal(torch.get_rng_state(), state)
