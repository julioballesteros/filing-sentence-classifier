"""Validated, serializable training settings without importing ML dependencies."""

import math
import tomllib
from dataclasses import asdict, dataclass, field
from typing import Self


class ConfigurationError(ValueError):
    """Training settings are incomplete, unsupported, or inconsistent."""


def _integer(name: str, value: int, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}.")


def _nonnegative_number(name: str, value: float) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ConfigurationError(f"{name} must be a finite nonnegative number.")


@dataclass(frozen=True)
class ModelConfig:
    """Architecture choices; vocabulary size and class count come from artifacts."""

    embedding_dim: int = 128
    hidden_dim: int = 64
    dropout: float = 0.0

    def __post_init__(self) -> None:
        _integer("model.embedding_dim", self.embedding_dim, 1)
        _integer("model.hidden_dim", self.hidden_dim, 1)
        _nonnegative_number("model.dropout", self.dropout)
        if self.dropout >= 1:
            raise ConfigurationError("model.dropout must be in [0, 1).")


@dataclass(frozen=True)
class RuntimeConfig:
    """CPU reference profile; the seed is independent of the saved split seed."""

    seed: int = 17
    device: str = "cpu"
    num_workers: int = 0
    num_threads: int = 1

    def __post_init__(self) -> None:
        _integer("runtime.seed", self.seed, 0)
        if self.seed >= 2**32:
            raise ConfigurationError("runtime.seed must be in [0, 2**32).")
        if self.device != "cpu":
            raise ConfigurationError("Only the explicit CPU runtime is supported.")
        _integer("runtime.num_workers", self.num_workers, 0)
        _integer("runtime.num_threads", self.num_threads, 1)


@dataclass(frozen=True)
class TrainingConfig:
    """Initial MeanPoolMLP/AdamW settings, separate from data and saved encoding."""

    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    batch_size: int = 32
    max_epochs: int = 30
    patience: int = 5
    min_delta: float = 0.0
    learning_rate: float = 0.001
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.model, ModelConfig) or not isinstance(
            self.runtime, RuntimeConfig
        ):
            raise ConfigurationError("Expected ModelConfig and RuntimeConfig sections.")
        _integer("training.batch_size", self.batch_size, 1)
        _integer("training.max_epochs", self.max_epochs, 1)
        _integer("training.patience", self.patience, 1)
        _nonnegative_number("training.min_delta", self.min_delta)
        if self.min_delta >= 1:
            raise ConfigurationError("training.min_delta must be in [0, 1).")
        _nonnegative_number("training.learning_rate", self.learning_rate)
        if self.learning_rate == 0:
            raise ConfigurationError("training.learning_rate must be positive.")
        _nonnegative_number("training.weight_decay", self.weight_decay)

    @classmethod
    def from_toml(cls, content: bytes) -> Self:
        """Parse TOML bytes; require every field and reject unknown settings."""
        try:
            state = tomllib.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ConfigurationError(f"Invalid training TOML: {exc}") from exc
        return cls.from_dict(state)

    @classmethod
    def from_dict(cls, state: object) -> Self:
        """Restore the same schema from JSON metadata without touching RNG state."""
        if (
            not isinstance(state, dict)
            or set(state) != {"schema_version", "model", "training", "runtime"}
            or type(state["schema_version"]) is not int
            or state["schema_version"] != 2
        ):
            raise ConfigurationError(
                "Expected training configuration schema_version=2."
            )
        for name, keys in (
            ("model", {"embedding_dim", "hidden_dim", "dropout"}),
            (
                "training",
                {
                    "batch_size",
                    "max_epochs",
                    "patience",
                    "min_delta",
                    "learning_rate",
                    "weight_decay",
                },
            ),
            ("runtime", {"seed", "device", "num_workers", "num_threads"}),
        ):
            if not isinstance(state[name], dict) or set(state[name]) != keys:
                raise ConfigurationError(f"Unexpected or missing fields in [{name}].")
        return cls(
            model=ModelConfig(**state["model"]),
            runtime=RuntimeConfig(**state["runtime"]),
            **state["training"],
        )

    def to_dict(self) -> dict[str, object]:
        """Return complete, JSON-serializable settings for future run artifacts."""
        return {
            "schema_version": 2,
            "model": {
                "embedding_dim": self.model.embedding_dim,
                "hidden_dim": self.model.hidden_dim,
                "dropout": float(self.model.dropout),
            },
            "training": {
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "min_delta": float(self.min_delta),
                "learning_rate": float(self.learning_rate),
                "weight_decay": float(self.weight_decay),
            },
            "runtime": asdict(self.runtime),
        }
