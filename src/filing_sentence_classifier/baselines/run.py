"""Fit the majority reference on frozen train and evaluate frozen validation."""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from filing_sentence_classifier.baselines.majority import (
    BaselineError,
    MajorityClassifier,
)
from filing_sentence_classifier.data.loading import load_split
from filing_sentence_classifier.evaluation.evaluate import evaluate_prediction_file


def _json(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _publish(staging: Path, destination: Path) -> Path:
    if destination.is_symlink():
        raise BaselineError("The run destination must not be a symlink.")
    if destination.exists():
        if (
            not destination.is_dir()
            or {path.name for path in destination.iterdir()}
            != {path.name for path in staging.iterdir()}
            or any(
                (destination / path.name).is_symlink()
                or not (destination / path.name).is_file()
                or (destination / path.name).read_bytes() != path.read_bytes()
                for path in staging.iterdir()
            )
        ):
            raise BaselineError(
                "Existing run differs; left unchanged. Choose another --output-dir."
            )
    else:
        staging.rename(destination)
    return destination


def run_majority_baseline(
    data_dir: Path,
    output_dir: Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> Path:
    """Save model, validation predictions, common metrics, and a hashed run manifest.

    Fit before loading validation, using only training targets. No test, source
    data, cleaning, or splitting is accessed. Identical runs verify existing bytes;
    changed runs fail without replacing results. No randomness is involved.
    """
    try:
        train = load_split(
            data_dir, "train", expected_manifest_sha256=expected_manifest_sha256
        )
        model = MajorityClassifier.fit(train.targets, label_ids=train.label_ids)
        val = load_split(
            data_dir, "val", expected_manifest_sha256=train.manifest_sha256
        )
        predicted = model.predict(val.texts)
        destination = output_dir.expanduser().absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix=".majority-", dir=destination.parent
        ) as temporary:
            staging = Path(temporary) / "run"
            staging.mkdir()
            (staging / "model.json").write_bytes(_json(model.to_dict()))
            prediction_path = staging / "predictions.val.jsonl"
            prediction_path.write_text(
                "".join(
                    json.dumps(
                        {"sample_id": sample_id, "predicted_label": label},
                        sort_keys=True,
                    )
                    + "\n"
                    for sample_id, label in zip(val.sample_ids, predicted, strict=True)
                ),
                encoding="utf-8",
            )
            metrics = evaluate_prediction_file(
                data_dir,
                "val",
                prediction_path,
                expected_manifest_sha256=train.manifest_sha256,
            )
            (staging / "metrics.val.json").write_bytes(_json(metrics))
            package_root = Path(__file__).parents[1]
            manifest = {
                "schema_version": 1,
                "stage": "baseline_run",
                "recipe": {
                    "family": "majority",
                    "fit_split": "train",
                    "evaluation_split": "val",
                    "tie_break": "smallest_label_id",
                    "weighting": "one_vote_per_training_row",
                    "text_features": None,
                    "seed": None,
                },
                "data": {
                    "manifest_sha256": train.manifest_sha256,
                    "train_sha256": train.records_sha256,
                    "val_sha256": val.records_sha256,
                    "train_rows": len(train.records),
                    "val_rows": len(val.records),
                    "label_names": dict(
                        zip(
                            (str(label) for label in train.label_ids),
                            train.label_names,
                            strict=True,
                        )
                    ),
                },
                "environment": {
                    "python": platform.python_version(),
                    **{
                        name: version(name)
                        for name in (
                            "filing-sentence-classifier",
                            "scikit-learn",
                            "numpy",
                            "scipy",
                        )
                    },
                },
                "code_sha256": {
                    name: hashlib.sha256((package_root / name).read_bytes()).hexdigest()
                    for name in (
                        "baselines/majority.py",
                        "baselines/run.py",
                        "data/loading.py",
                        "evaluation/evaluate.py",
                        "evaluation/metrics.py",
                    )
                },
                "files": {
                    path.name: {
                        "size_bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in sorted(staging.iterdir())
                },
            }
            (staging / "manifest.json").write_bytes(_json(manifest))
            return _publish(staging, destination)
    except (OSError, ValueError) as exc:
        raise BaselineError(f"Could not complete majority baseline: {exc}") from exc
