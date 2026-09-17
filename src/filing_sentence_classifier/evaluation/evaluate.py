"""Align predictions by sample identity and evaluate one frozen development split."""

import hashlib
import json
import platform
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Protocol

from filing_sentence_classifier.data.loading import SplitName, load_split
from filing_sentence_classifier.evaluation.metrics import (
    ClassificationMetrics,
    EvaluationError,
    classification_metrics,
)


@dataclass(frozen=True)
class Prediction:
    sample_id: str
    predicted_label: int


class EvaluationDataset(Protocol):
    """Aligned targets and identities; loading and access policy belong to callers."""

    @property
    def name(self) -> str: ...

    @property
    def sample_ids(self) -> tuple[str, ...]: ...

    @property
    def targets(self) -> tuple[int, ...]: ...

    @property
    def label_ids(self) -> tuple[int, ...]: ...


def evaluate_predictions(
    split: EvaluationDataset, predictions: Iterable[Prediction]
) -> ClassificationMetrics:
    """Require exactly one prediction per target ID, allowing any prediction order."""
    by_id: dict[str, int] = {}
    for prediction in predictions:
        if not isinstance(prediction.sample_id, str) or not prediction.sample_id:
            raise EvaluationError("Predictions require nonempty string sample IDs.")
        if prediction.sample_id in by_id:
            raise EvaluationError("Duplicate sample ID in predictions.")
        by_id[prediction.sample_id] = prediction.predicted_label
    expected = set(split.sample_ids)
    if set(by_id) != expected:
        raise EvaluationError(
            f"Prediction IDs do not match {split.name}: "
            f"missing={len(expected - set(by_id))}, unexpected={len(set(by_id) - expected)}."
        )
    return classification_metrics(
        split.targets,
        (by_id[sample_id] for sample_id in split.sample_ids),
        label_ids=split.label_ids,
    )


def evaluate_prediction_file(
    directory: Path,
    split: SplitName | str,
    predictions_path: Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Return JSON-ready metrics plus the exact data/prediction provenance.

    Prediction JSONL rows contain only sample_id and predicted_label. Ground truth
    always comes from the verified split, never from a prediction file. No model
    fitting, cleaning, splitting, or test access is performed by evaluation.
    """
    dataset = load_split(
        directory, split, expected_manifest_sha256=expected_manifest_sha256
    )
    predictions: list[Prediction] = []
    try:
        content = predictions_path.expanduser().read_bytes()
        for line in content.splitlines():
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {
                "sample_id",
                "predicted_label",
            }:
                raise EvaluationError(
                    "Prediction rows require exactly sample_id and predicted_label."
                )
            if (
                not isinstance(row["sample_id"], str)
                or type(row["predicted_label"]) is not int
            ):
                raise EvaluationError(
                    "Prediction rows require a string sample_id and integer predicted_label."
                )
            predictions.append(Prediction(row["sample_id"], row["predicted_label"]))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Could not read prediction JSONL: {exc}") from exc
    metrics = evaluate_predictions(dataset, predictions)
    return {
        "schema_version": 1,
        "split": dataset.name.value,
        "data_manifest_sha256": dataset.manifest_sha256,
        "records_sha256": dataset.records_sha256,
        "predictions_sha256": hashlib.sha256(content).hexdigest(),
        "label_names": dict(
            zip(
                (str(label) for label in dataset.label_ids),
                dataset.label_names,
                strict=True,
            )
        ),
        "primary_metric": "macro_f1",
        "zero_division": 0,
        "confusion_matrix_axes": {"rows": "true", "columns": "predicted"},
        "metrics": asdict(metrics),
        "environment": {
            "python": platform.python_version(),
            "scikit-learn": version("scikit-learn"),
            "filing-sentence-classifier": version("filing-sentence-classifier"),
        },
        "code_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("evaluate.py", "metrics.py")
        },
    }
