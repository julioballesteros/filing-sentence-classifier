"""Compose frozen data, encoding, training, and persistent local run artifacts."""

import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import cast

import torch

from filing_sentence_classifier.data.dataloader import create_dataloader
from filing_sentence_classifier.data.dataset import SentenceDataset
from filing_sentence_classifier.data.loading import LoadedSplit, load_split
from filing_sentence_classifier.evaluation.evaluate import evaluate_prediction_file
from filing_sentence_classifier.models.mean_pool_mlp import MeanPoolMLP
from filing_sentence_classifier.text.encoding import TextEncoder, encoding_statistics
from filing_sentence_classifier.text.tokenization import tokenization_recipe
from filing_sentence_classifier.text.vocabulary import Vocabulary
from filing_sentence_classifier.training.artifacts import (
    capture_source,
    environment,
    file_inventory,
    json_bytes,
    sha256,
    write_json,
)
from filing_sentence_classifier.training.config import TrainingConfig
from filing_sentence_classifier.training.fit import EpochMetrics, fit
from filing_sentence_classifier.training.plots import save_learning_curves
from filing_sentence_classifier.training.reproducibility import configure_runtime


class TrainingRunError(RuntimeError):
    """A training run failed or its destination already belongs to another run."""


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Expected an object in the vocabulary manifest.")
    return cast(dict[str, object], value)


def _load_vocabulary(directory: Path, train: LoadedSplit) -> tuple[Vocabulary, bytes]:
    content = (directory / "manifest.json").read_bytes()
    manifest = _object(json.loads(content))
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("stage") != "vocabulary_build"
        or manifest.get("fit_split") != "train"
        or manifest.get("tokenization") != tokenization_recipe()
        or manifest.get("data")
        != {
            "manifest_sha256": train.manifest_sha256,
            "train_sha256": train.records_sha256,
            "train_rows": len(train.records),
        }
    ):
        raise ValueError(
            "Vocabulary provenance does not match the frozen training partition."
        )
    saved = _object(_object(manifest.get("files")).get("vocabulary.json"))
    vocabulary_bytes = (directory / "vocabulary.json").read_bytes()
    if saved.get("sha256") != sha256(vocabulary_bytes) or saved.get(
        "size_bytes"
    ) != len(vocabulary_bytes):
        raise ValueError("Vocabulary checksum or size does not match its manifest.")
    vocabulary = Vocabulary.from_dict(json.loads(vocabulary_bytes))
    if manifest.get("vocabulary") != vocabulary.recipe():
        raise ValueError("Vocabulary recipe does not match its manifest.")
    return vocabulary, content


def run_training(
    data_dir: Path,
    output_dir: Path,
    config_path: Path,
    vocabulary_dir: Path,
    *,
    max_length: int = 128,
    expected_manifest_sha256: str | None = None,
    project_dir: Path | None = None,
    on_epoch: Callable[[EpochMetrics], None] | None = None,
) -> Path:
    """Run one configured CPU experiment and preserve success or failure evidence.

    Require a new destination, verified train/val, and a vocabulary fitted on that
    train artifact. Construct the encoder without fitting and save it with the
    selected weights. History is flushed per epoch; failed runs retain partial
    history and any best checkpoint. Test and raw source data are never loaded.

    Runtime initialization changes process-wide RNG and numerical settings. Source
    files are snapshotted from the imported package; project_dir (default cwd) adds
    checkout/lockfile provenance when available. No automatic retry or overwrite.
    """
    destination = output_dir.expanduser().absolute()
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise TrainingRunError(
            f"Choose a new --output-dir; cannot create {destination}: {exc}"
        ) from exc
    started = perf_counter()
    manifest: dict[str, object] = {
        "schema_version": 1,
        "stage": "neural_training_run",
        "status": "running",
        "run_id": destination.name,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "recipe": {
            "family": "mean_pool_mlp",
            "fit_split": "train",
            "evaluation_split": "val",
            "selection_metric": "macro_f1",
            "checkpoint_tie_break": "earliest_epoch",
            "optimizer": "AdamW",
            "loss": "unweighted_cross_entropy",
        },
    }
    try:
        write_json(destination / "manifest.json", manifest)
        config_bytes = config_path.expanduser().read_bytes()
        (destination / "config.toml").write_bytes(config_bytes)
        config = TrainingConfig.from_toml(config_bytes)
        manifest["config"] = config.to_dict()
        manifest["source"] = capture_source(destination, project_dir or Path.cwd())
        manifest["environment"] = environment()
        train = load_split(
            data_dir, "train", expected_manifest_sha256=expected_manifest_sha256
        )
        vocabulary, vocabulary_manifest = _load_vocabulary(
            vocabulary_dir.expanduser(), train
        )
        encoder = TextEncoder(vocabulary, max_length=max_length)
        val = load_split(
            data_dir, "val", expected_manifest_sha256=train.manifest_sha256
        )
        if (train.label_ids, train.label_names) != (val.label_ids, val.label_names):
            raise ValueError("Train and validation must have the same class mapping.")
        data_manifest = (data_dir.expanduser() / "manifest.json").read_bytes()
        if sha256(data_manifest) != train.manifest_sha256:
            raise ValueError("Data manifest changed while loading the run inputs.")
        (destination / "data-manifest.json").write_bytes(data_manifest)
        (destination / "vocabulary-manifest.json").write_bytes(vocabulary_manifest)
        (destination / "encoder.json").write_bytes(json_bytes(encoder.to_dict()))
        manifest["data"] = {
            "manifest_sha256": train.manifest_sha256,
            "train_sha256": train.records_sha256,
            "val_sha256": val.records_sha256,
            "train_rows": len(train.records),
            "val_rows": len(val.records),
            "label_names": dict(
                zip(map(str, train.label_ids), train.label_names, strict=True)
            ),
        }
        runtime = configure_runtime(config.runtime)
        runtime["torch_build"] = torch.__config__.show()
        manifest["runtime"] = runtime
        train_dataset, val_dataset = (
            SentenceDataset(train, encoder),
            SentenceDataset(val, encoder),
        )
        manifest["preprocessing"] = {
            "max_length": max_length,
            "vocabulary_size": len(vocabulary),
            "train": encoding_statistics(train_dataset.encodings),
            "val": encoding_statistics(val_dataset.encodings),
        }
        train_loader = create_dataloader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.runtime.seed,
            num_workers=config.runtime.num_workers,
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config.batch_size,
            seed=config.runtime.seed,
            num_workers=config.runtime.num_workers,
        )
        model = MeanPoolMLP(
            len(vocabulary),
            embedding_dim=config.model.embedding_dim,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
            num_classes=len(train.label_ids),
        ).to(config.runtime.device)
        manifest["model"] = {
            "num_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "num_classes": len(train.label_ids),
        }
        write_json(destination / "manifest.json", manifest)
        fitting_started = perf_counter()
        with (destination / "history.jsonl").open(
            "x", encoding="utf-8"
        ) as history_file:

            def record_epoch(row: EpochMetrics) -> None:
                history_file.write(
                    json.dumps(asdict(row), sort_keys=True, allow_nan=False) + "\n"
                )
                history_file.flush()
                if on_epoch is not None:
                    on_epoch(row)

            result = fit(
                model,
                train_loader,
                val_loader,
                config=config,
                label_ids=val.label_ids,
                validation_sample_ids=val.sample_ids,
                checkpoint_path=destination / "checkpoints" / "best.pt",
                on_epoch=record_epoch,
            )
        fitting_seconds = perf_counter() - fitting_started
        prediction_path = destination / "predictions.val.jsonl"
        prediction_path.write_text(
            "".join(
                json.dumps(
                    {"sample_id": sample_id, "predicted_label": label}, sort_keys=True
                )
                + "\n"
                for sample_id, label in zip(
                    result.best_validation.sample_ids,
                    result.best_validation.predicted_labels,
                    strict=True,
                )
            ),
            encoding="utf-8",
        )
        metrics = evaluate_prediction_file(
            data_dir,
            "val",
            prediction_path,
            expected_manifest_sha256=train.manifest_sha256,
        )
        if metrics["metrics"] != asdict(result.best_validation.metrics):
            raise ValueError(
                "Saved predictions disagree with restored-model validation."
            )
        write_json(destination / "metrics.val.json", metrics)
        write_json(
            destination / "summary.json",
            {
                "best_epoch": result.best_epoch,
                "epochs_completed": len(result.history),
                "stopped_early": result.stopped_early,
                "training_seconds": fitting_seconds,
                "validation_mean_loss": result.best_validation.mean_loss,
                "validation_macro_f1": result.best_validation.metrics.macro_f1,
                "validation_accuracy": result.best_validation.metrics.accuracy,
            },
        )
        save_learning_curves(
            result.history, result.best_epoch, destination / "learning-curves.png"
        )
        manifest.update(
            status="completed",
            finished_at_utc=datetime.now(UTC).isoformat(),
            duration_seconds=perf_counter() - started,
        )
        manifest["files"] = file_inventory(destination)
        write_json(destination / "manifest.json", manifest)
        return destination
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(
            status="failed",
            finished_at_utc=datetime.now(UTC).isoformat(),
            duration_seconds=perf_counter() - started,
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        try:
            manifest["files"] = file_inventory(destination)
            write_json(destination / "manifest.json", manifest)
        except OSError as recording_error:
            raise TrainingRunError(
                f"Training failed ({exc}); failure recording also failed: {recording_error}"
            ) from exc
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise TrainingRunError(
            f"Training run failed; evidence retained in {destination}: {exc}"
        ) from exc
