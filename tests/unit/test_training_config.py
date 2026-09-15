"""Validate the complete neural configuration and its portable representation."""

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from filing_sentence_classifier.training.config import (
    ConfigurationError,
    ModelConfig,
    RuntimeConfig,
    TrainingConfig,
)

REFERENCE = Path(__file__).parents[2] / "configs/experiments/mean-pool-mlp-v1.toml"


def test_reference_config_is_complete_and_round_trips_through_json() -> None:
    config = TrainingConfig.from_toml(REFERENCE.read_bytes())
    assert config == TrainingConfig()
    assert config.runtime.seed == 17
    assert TrainingConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    changed = replace(config, model=ModelConfig(32, 16, 0.2), learning_rate=0.01)
    assert TrainingConfig.from_dict(changed.to_dict()) == changed
    assert config.model == ModelConfig()
    state = config.to_dict()
    state["runtime"]["seed"] = 43
    assert config.runtime.seed == 17
    with pytest.raises(FrozenInstanceError):
        config.runtime.seed = 43


@pytest.mark.parametrize("section", ["model", "training", "runtime"])
@pytest.mark.parametrize("change", ["missing", "extra", "not_a_table"])
def test_sections_reject_missing_unknown_and_invalid_fields(section, change) -> None:
    state = TrainingConfig().to_dict()
    if change == "missing":
        state[section].pop(next(iter(state[section])))
    elif change == "extra":
        state[section]["typo"] = 1
    else:
        state[section] = []
    with pytest.raises(ConfigurationError, match=section):
        TrainingConfig.from_dict(state)


@pytest.mark.parametrize("version", [2, True, "1"])
def test_incompatible_schema_versions_are_rejected(version) -> None:
    state = TrainingConfig().to_dict()
    state["schema_version"] = version
    with pytest.raises(ConfigurationError, match="schema_version"):
        TrainingConfig.from_dict(state)


def test_root_fields_are_strict() -> None:
    for state in ({}, None, TrainingConfig().to_dict() | {"unknown": 1}):
        with pytest.raises(ConfigurationError):
            TrainingConfig.from_dict(state)


@pytest.mark.parametrize("content", [b"\xff", b"[model", b"seed = 1\nseed = 2"])
def test_invalid_toml_has_a_configuration_error(content: bytes) -> None:
    with pytest.raises(ConfigurationError, match="Invalid training TOML"):
        TrainingConfig.from_toml(content)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("model", "embedding_dim", 0),
        ("model", "hidden_dim", 2.5),
        ("model", "hidden_dim", True),
        ("model", "dropout", 1.0),
        ("model", "dropout", -0.1),
        ("model", "dropout", float("nan")),
        ("training", "batch_size", 0),
        ("training", "batch_size", True),
        ("training", "max_epochs", -1),
        ("training", "learning_rate", 0),
        ("training", "learning_rate", True),
        ("training", "learning_rate", "0.001"),
        ("training", "learning_rate", float("inf")),
        ("training", "weight_decay", -0.1),
        ("training", "weight_decay", float("nan")),
        ("runtime", "seed", -1),
        ("runtime", "seed", True),
        ("runtime", "seed", 2**32),
        ("runtime", "device", "auto"),
        ("runtime", "device", "cuda"),
        ("runtime", "device", "mps"),
        ("runtime", "num_workers", -1),
        ("runtime", "num_workers", True),
        ("runtime", "num_threads", 0),
    ],
)
def test_invalid_settings_are_rejected(section, key, value) -> None:
    state = TrainingConfig().to_dict()
    state[section][key] = value
    with pytest.raises(ConfigurationError):
        TrainingConfig.from_dict(state)


def test_direct_construction_validates_nested_types_and_seed_boundaries() -> None:
    for values in ({"model": {}}, {"runtime": {}}):
        with pytest.raises(ConfigurationError):
            TrainingConfig(**values)
    assert RuntimeConfig(seed=0).seed == 0
    assert RuntimeConfig(seed=2**32 - 1).seed == 2**32 - 1
