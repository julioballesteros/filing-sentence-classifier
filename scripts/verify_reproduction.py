"""Verify freshly rebuilt fixed recipes against the recorded validation/test results.

This is technical reproduction of an already exposed test campaign, not a new
selection study. Run after the commands in reports/reproduction.md. It requires
the original configurations and data identities, never tunes or fits anything,
and preserves the historical selection and evaluation reports.
"""

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from filing_sentence_classifier.baselines.majority import MajorityClassifier
from filing_sentence_classifier.data.fls import FLS_SOURCE, FLS_TEST_FILE, LABEL_NAMES
from filing_sentence_classifier.data.heldout import (
    jsonl_bytes,
    prepare_published_test,
    test_preparation_recipe,
)
from filing_sentence_classifier.data.loading import load_split
from filing_sentence_classifier.evaluation.evaluate import (
    Prediction,
    evaluate_predictions,
)
from filing_sentence_classifier.inference.predictor import Predictor
from filing_sentence_classifier.training.artifacts import (
    capture_source,
    environment,
    file_inventory,
    sha256,
    write_json,
)


def verify_reproduction(root: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("Choose a new output directory for verification.")
    inputs = {}

    def read(path, expected=None):
        path = root / path
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected a regular input file: {path}")
        content = path.read_bytes()
        actual = {"sha256": sha256(content), "size_bytes": len(content)}
        if expected is not None and actual != {key: expected[key] for key in actual}:
            raise ValueError(f"Frozen input differs: {path}")
        inputs[path.relative_to(root).as_posix()] = actual
        return content

    def load(path, expected=None):
        return json.loads(read(path, expected))

    results = load("reports/test-v1/results.json")
    selection_bytes = read("reports/test-v1/selection.json")
    if sha256(selection_bytes) != results["selection_sha256"]:
        raise ValueError("The recorded selection does not match the test campaign.")
    selection = json.loads(selection_bytes)
    if selection["preparation"] != test_preparation_recipe():
        raise ValueError("Test preparation differs from the registered policy.")
    freeze = load(selection["freeze"]["path"], selection["freeze"])
    seeds_ref = freeze["evidence"]["seeds.json"]
    seeds = load(seeds_ref["path"], seeds_ref)
    data = freeze["data"]
    data_dir = root / Path(selection["development_manifest"]["path"]).parent
    groups = set()
    for split in ("train", "val"):
        part = load_split(
            data_dir, split, expected_manifest_sha256=data["manifest_sha256"]
        )
        if part.records_sha256 != data[f"{split}_sha256"]:
            raise ValueError("Rebuilt development data differs from the frozen split.")
        groups.update(row.group_id for row in part.records)
    read(
        Path(selection["development_manifest"]["path"]),
        selection["development_manifest"],
    )
    originals = {row["run_id"]: row["provenance"]["files"] for row in seeds["runs"]}
    majority = load("reports/majority-v1/manifest.json")
    originals["majority-v1"] = majority["files"]
    classical_id = freeze["classical_reference"]["run_id"]
    originals[classical_id] = load(
        f"reports/tfidf-selection-v1/{classical_id}/manifest.json"
    )["files"]
    expected_ids = {
        "majority-v1",
        classical_id,
        *(row["run_id"] for row in seeds["runs"]),
    }
    if (
        len(expected_ids) != 5
        or {row["model_id"] for row in results["models"]} != expected_ids
    ):
        raise ValueError("Expected the five originally selected models.")

    restored = []
    for result in results["models"]:
        name = result["model_id"]
        directory = Path("artifacts/runs") / name
        run = load(directory / "manifest.json")
        if (
            run["data"] != data
            or run["recipe"]["fit_split"] != "train"
            or run["recipe"]["evaluation_split"] != "val"
        ):
            raise ValueError(f"Rebuilt run uses different development data: {name}")
        original = originals[name]
        # Compare the actual bytes, not only the claims in the new run manifest.
        read(directory / "predictions.val.jsonl", original["predictions.val.jsonl"])
        if result["family"] == "majority":
            model_bytes = read(directory / "model.json", original["model.json"])
            model = MajorityClassifier.from_dict(json.loads(model_bytes))
            identical_payload = True
        else:
            read(directory / "config.toml", original["config.toml"])
            if result["family"] == "mean_pool_mlp":
                read(directory / "encoder.json", original["encoder.json"])
                if (
                    run.get("status") != "completed"
                    or run["config"]["runtime"]["seed"] != result["seed"]
                ):
                    raise ValueError(f"Incomplete or incorrect neural seed: {name}")
                payload = "checkpoints/best.pt"
            else:
                payload = "model.joblib"
            payload_bytes = read(directory / payload, run["files"][payload])
            identical_payload = sha256(payload_bytes) == original[payload]["sha256"]
            bundle = Path("artifacts/bundles") / name
            manifest_bytes = read(bundle / "manifest.json")
            model = Predictor.from_bundle(
                root / bundle, expected_manifest_sha256=sha256(manifest_bytes)
            )
            if (
                model.manifest.model_id != name
                or model.manifest.source_manifest_sha256
                != sha256((root / directory / "manifest.json").read_bytes())
            ):
                raise ValueError(f"Bundle does not belong to the rebuilt run: {name}")
            for file, metadata in model.manifest.files.items():
                read(bundle / file, metadata.to_dict())
            bundle_payload = (
                "model.pt" if result["family"] == "mean_pool_mlp" else "model.joblib"
            )
            if model.manifest.files[bundle_payload].sha256 != sha256(payload_bytes):
                raise ValueError(f"Export changed the rebuilt model payload: {name}")
        restored.append((result, model, identical_payload))

    output.mkdir(parents=True)
    state = {
        "schema_version": 1,
        "stage": "fixed_recipe_reproduction",
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "prior_test_exposure": "The recorded final campaign and qualitative error analysis are complete. This run only verifies the same fixed recipes.",
        "new_hyperparameter_configurations": 0,
        "test_driven_tuning": False,
        "selection_sha256": results["selection_sha256"],
        "test_access_started": False,
        "inputs": inputs,
        "environment": environment(),
        "source": capture_source(output, root),
        "verification_script_sha256": sha256(Path(__file__).read_bytes()),
        "models": [],
    }
    write_json(output / "manifest.json", state)
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        state["test_access_started"] = True
        write_json(output / "manifest.json", state)
        prepared = prepare_published_test(
            read(
                Path("data/raw") / FLS_SOURCE.revision / FLS_TEST_FILE.path,
                {
                    "sha256": FLS_TEST_FILE.sha256,
                    "size_bytes": FLS_TEST_FILE.size_bytes,
                },
            ),
            FLS_SOURCE,
            FLS_TEST_FILE,
            LABEL_NAMES,
            expected_rows=selection["expected_rows"],
        )
        if groups & {row.group_id for row in prepared.records}:
            raise ValueError("Unexpected development/test overlap.")
        prepared_bytes = jsonl_bytes([asdict(row) for row in prepared.records])
        if any(
            sha256(prepared_bytes) != row["records_sha256"] for row in results["models"]
        ):
            raise ValueError("Prepared test differs from the recorded campaign.")
        state["test_records_sha256"] = sha256(prepared_bytes)
        for expected, model, identical_payload in restored:
            name = expected["model_id"]
            scores = []
            if isinstance(model, MajorityClassifier):
                predictions = [
                    Prediction(sample_id, label)
                    for sample_id, label in zip(
                        prepared.sample_ids, model.predict(prepared.texts), strict=True
                    )
                ]
            else:
                predictions = []
                for start in range(0, len(prepared.records), 32):
                    rows = prepared.records[start : start + 32]
                    for row, prediction in zip(
                        rows,
                        model.predict(tuple(row.text for row in rows), batch_size=32),
                        strict=True,
                    ):
                        predictions.append(
                            Prediction(row.sample_id, prediction.label_id)
                        )
                        scores.append(
                            {"sample_id": row.sample_id, **prediction.to_dict()}
                        )
            prediction_bytes = jsonl_bytes([asdict(row) for row in predictions])
            metrics = asdict(evaluate_predictions(prepared, predictions))
            directory = output / name
            directory.mkdir()
            (directory / "predictions.test.jsonl").write_bytes(prediction_bytes)
            write_json(directory / "metrics.test.json", metrics)
            if scores:
                (directory / "scores.test.jsonl").write_bytes(jsonl_bytes(scores))
            equal_labels = sha256(prediction_bytes) == expected["predictions_sha256"]
            equal_metrics = json.loads(json.dumps(metrics)) == expected["metrics"]
            state["models"].append(
                {
                    "model_id": name,
                    "validation_predictions_identical": True,
                    "test_predictions_identical": equal_labels,
                    "test_metrics_identical": equal_metrics,
                    "model_payload_identical": identical_payload,
                    "score_bytes_identical": sha256(jsonl_bytes(scores))
                    == expected["artifacts"]["scores"]["sha256"]
                    if scores
                    else None,
                }
            )
            if not equal_labels or not equal_metrics:
                raise ValueError(
                    f"Reproduction differs from recorded predictions/metrics: {name}. Retain and investigate; do not tune to test."
                )
        state["status"] = "completed"
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        state["finished_at_utc"] = datetime.now(UTC).isoformat()
        state["files"] = file_inventory(output)
        write_json(output / "manifest.json", state)
    print(
        f"All five rebuilt models reproduce validation and test predictions: {output}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/reproduction-check")
    )
    args = parser.parse_args()
    verify_reproduction(args.project_dir.resolve(), args.output_dir.resolve())
