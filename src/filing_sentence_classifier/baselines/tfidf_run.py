"""Run the configured TF-IDF reference on frozen development partitions."""

import hashlib
import json
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from filing_sentence_classifier.baselines.majority import BaselineError

# Reuse the existing byte verification and publication policy for baseline runs.
from filing_sentence_classifier.baselines.run import _json, _publish
from filing_sentence_classifier.baselines.tfidf import (
    TfidfConfig,
    fit_tfidf,
    load_tfidf_model,
    predict_labels,
    save_tfidf_model,
)
from filing_sentence_classifier.data.loading import load_split
from filing_sentence_classifier.evaluation.evaluate import evaluate_prediction_file


def run_tfidf_baseline(
    data_dir: Path,
    output_dir: Path,
    config_path: Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> Path:
    """Fit before loading val, save the whole pipeline, and use common evaluation."""
    try:
        config_bytes = config_path.expanduser().read_bytes()
        config = TfidfConfig.from_toml(config_bytes)
        train = load_split(
            data_dir, "train", expected_manifest_sha256=expected_manifest_sha256
        )
        model = fit_tfidf(
            train.texts, train.targets, label_ids=train.label_ids, config=config
        )
        val = load_split(
            data_dir, "val", expected_manifest_sha256=train.manifest_sha256
        )
        predicted = predict_labels(model, val.texts)
        destination = output_dir.expanduser().absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".tfidf-", dir=destination.parent) as temporary:
            staging = Path(temporary) / "run"
            staging.mkdir()
            (staging / "config.toml").write_bytes(config_bytes)
            model_path = staging / "model.joblib"
            save_tfidf_model(model, model_path)
            restored = load_tfidf_model(
                model_path,
                expected_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
            )
            if predict_labels(restored, val.texts) != predicted:
                raise BaselineError(
                    "Restored pipeline predictions differ; no run published."
                )
            class_counts = Counter(train.targets)
            (staging / "model.json").write_bytes(
                _json(
                    {
                        "schema_version": 1,
                        "family": "tfidf_logreg",
                        "serialization": "joblib",
                        "model_file": "model.joblib",
                        "class_ids": [int(label) for label in model.classes_],
                        "training_class_counts": {
                            str(label): class_counts[label] for label in train.label_ids
                        },
                        "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
                        "n_iter": [
                            int(value) for value in model.named_steps["logreg"].n_iter_
                        ],
                        "converged": True,
                    }
                )
            )
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
                "recipe": config.recipe(),
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
                            "joblib",
                            "threadpoolctl",
                        )
                    },
                },
                "code_sha256": {
                    name: hashlib.sha256((package_root / name).read_bytes()).hexdigest()
                    for name in (
                        "baselines/tfidf.py",
                        "baselines/tfidf_run.py",
                        "baselines/run.py",
                        "baselines/majority.py",
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
        raise BaselineError(f"Could not complete TF-IDF baseline: {exc}") from exc
