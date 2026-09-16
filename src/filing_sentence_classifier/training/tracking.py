"""Optional tracking boundary and an explicit local MLflow adapter."""

from __future__ import annotations

import os
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Literal, Protocol

from filing_sentence_classifier.training.artifacts import json_bytes, sha256

if TYPE_CHECKING:
    from mlflow import MlflowClient

    from filing_sentence_classifier.training.fit import EpochMetrics

RunStatus = Literal["FINISHED", "FAILED", "KILLED"]


class RunTracker(Protocol):
    """Observe a local run without owning training, selection, or serialization."""

    def start(self, run_dir: Path) -> dict[str, str]: ...

    def log_metadata(self, manifest: Mapping[str, object]) -> None: ...

    def log_epoch(self, row: EpochMetrics) -> None: ...

    def log_summary(self, summary: Mapping[str, object]) -> None: ...

    def finish(self, run_dir: Path, status: RunStatus) -> None: ...


def _parameters(values: Mapping[str, object], prefix: str = "") -> dict[str, str]:
    result = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_parameters(value, name))
        else:
            result[name] = str(value)
    return result


class MLflowTracker:
    """Mirror one run to SQLite and local artifacts, without fluent global state.

    Construction imports no MLflow and creates no files. start disables MLflow
    telemetry process-wide before importing the optional SDK. All tracking calls
    are synchronous; errors propagate to the run owner. Use one instance per run.
    """

    def __init__(
        self, directory: Path, experiment_name: str = "filing-sentence-classifier"
    ) -> None:
        if not experiment_name.strip():
            raise ValueError("MLflow experiment name must not be blank.")
        self.directory = directory.expanduser().resolve()
        self.experiment_name = experiment_name
        self._client: MlflowClient | None = None
        self._run_id: str | None = None

    def start(self, run_dir: Path) -> dict[str, str]:
        if self._run_id is not None:
            raise ValueError("Create a new tracker for each training run.")
        run_dir = run_dir.resolve()
        if self.directory.is_relative_to(run_dir) or run_dir.is_relative_to(
            self.directory
        ):
            raise ValueError("MLflow storage and the local run must not overlap.")
        os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
        try:
            from mlflow import MlflowClient
            from mlflow.exceptions import MlflowException
        except ModuleNotFoundError as exc:
            if exc.name != "mlflow":
                raise
            raise RuntimeError(
                "MLflow is missing. Install the package with its [tracking] extra."
            ) from exc

        self.directory.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{(self.directory / 'mlflow.db').as_posix()}"
        artifact_uri = (self.directory / "artifacts").as_uri()
        client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        self._client = client
        experiment = client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            try:
                client.create_experiment(
                    self.experiment_name, artifact_location=artifact_uri
                )
            except MlflowException as exc:
                if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                    raise
            experiment = client.get_experiment_by_name(self.experiment_name)
        if experiment is None or experiment.lifecycle_stage != "active":
            raise ValueError("MLflow experiment must exist and be active.")
        if experiment.artifact_location != artifact_uri:
            raise ValueError("MLflow experiment uses a different artifact location.")
        run = client.create_run(
            experiment.experiment_id,
            run_name=run_dir.name,
            tags={"fsc.local_run": str(run_dir), "fsc.family": "mean_pool_mlp"},
        )
        self._run_id = run.info.run_id
        return {
            "backend": "mlflow",
            "version": version("mlflow"),
            "tracking_uri": tracking_uri,
            "experiment_id": experiment.experiment_id,
            "experiment_name": self.experiment_name,
            "run_id": self._run_id,
            "artifact_uri": run.info.artifact_uri,
        }

    def _active(self) -> tuple[MlflowClient, str]:
        if self._client is None or self._run_id is None:
            raise RuntimeError("Start the tracker before recording run data.")
        return self._client, self._run_id

    def log_metadata(self, manifest: Mapping[str, object]) -> None:
        from mlflow.entities import Param, RunTag

        client, run_id = self._active()
        selected = {key: manifest[key] for key in ("config", "recipe", "data", "model")}
        preprocessing = manifest["preprocessing"]
        assert isinstance(preprocessing, dict)
        selected["preprocessing"] = {
            key: preprocessing[key] for key in ("max_length", "vocabulary_size")
        }
        source = manifest["source"]
        assert isinstance(source, dict)
        tags = [
            RunTag("fsc.code_sha256", sha256(json_bytes(source["code_sha256"]))),
        ]
        git = source.get("git")
        if isinstance(git, dict):
            tags.extend(
                [
                    RunTag("mlflow.source.git.commit", str(git["commit"])),
                    RunTag(
                        "fsc.reproducible_from_commit",
                        str(git["reproducible_from_commit"]).lower(),
                    ),
                ]
            )
        params = [Param(key, value) for key, value in _parameters(selected).items()]
        client.log_batch(run_id, params=params, tags=tags, synchronous=True)

    def _log_metrics(self, values: Mapping[str, float], step: int) -> None:
        from mlflow.entities import Metric

        client, run_id = self._active()
        timestamp = int(time() * 1000)
        client.log_batch(
            run_id,
            metrics=[
                Metric(key, value, timestamp, step) for key, value in values.items()
            ],
            synchronous=True,
        )

    def log_epoch(self, row: EpochMetrics) -> None:
        values = {
            "train/loss": row.training.mean_loss,
            "val/loss": row.validation_mean_loss,
            "val/macro_f1": row.validation_metrics.macro_f1,
            "val/accuracy": row.validation_metrics.accuracy,
        }
        for scores in row.validation_metrics.per_class:
            for name in ("precision", "recall", "f1"):
                values[f"val/class_{scores.label}/{name}"] = getattr(scores, name)
        self._log_metrics(values, step=row.epoch)

    def log_summary(self, summary: Mapping[str, object]) -> None:
        self._log_metrics(
            {
                f"summary/{name}": float(value)
                for name, value in summary.items()
                if isinstance(value, (int, float))
            },
            step=0,
        )

    def finish(self, run_dir: Path, status: RunStatus) -> None:
        if self._run_id is None:
            return  # Initialization may have failed before a tracked run existed.
        client, run_id = self._active()
        terminal_status: RunStatus = "FAILED"
        try:
            client.log_artifacts(run_id, str(run_dir))
            terminal_status = status
        finally:
            # An artifact-copy failure must not leave the run marked successful.
            client.set_terminated(run_id, status=terminal_status)
