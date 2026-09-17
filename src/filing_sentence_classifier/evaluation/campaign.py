"""Run the predeclared five-model test campaign without fitting or selection."""

import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any

from filing_sentence_classifier.artifacts import BundleFile, verify_bundle
from filing_sentence_classifier.baselines.majority import MajorityClassifier
from filing_sentence_classifier.data.heldout import (
    jsonl_bytes,
    prepare_published_test,
    test_preparation_recipe,
)
from filing_sentence_classifier.data.loading import load_split
from filing_sentence_classifier.data.spec import DatasetSource, SourceFile
from filing_sentence_classifier.evaluation.evaluate import (
    Prediction,
    evaluate_predictions,
)
from filing_sentence_classifier.evaluation.metrics import EvaluationError
from filing_sentence_classifier.inference.contracts import validate_model_id
from filing_sentence_classifier.inference.predictor import Predictor
from filing_sentence_classifier.training.artifacts import (
    capture_source,
    environment,
    file_inventory,
    sha256,
    write_json,
)


@dataclass(frozen=True)
class SelectedModel:
    model_id: str
    family: str
    seed: int | None
    model: Predictor | MajorityClassifier


def _object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise EvaluationError("Expected an object in the test selection evidence.")
    return value


def _read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise EvaluationError(f"Expected a regular evaluation input: {path}.")
    return path.read_bytes()


def _reference(root: Path, value: object) -> tuple[Path, bytes]:
    ref = _object(value)
    name = ref.get("path")
    if not isinstance(name, str):
        raise EvaluationError("Evidence requires a project-relative path.")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise EvaluationError("Evidence paths must stay inside the project root.")
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise EvaluationError("Evidence paths must stay inside the project root.")
    expected = BundleFile.from_dict(
        {key: ref.get(key) for key in ("sha256", "size_bytes")}
    )
    content = _read(path)
    if len(content) != expected.size_bytes or sha256(content) != expected.sha256:
        raise EvaluationError(f"Test selection integrity check failed: {name}.")
    return path, content


def _preflight(
    selection: dict[str, Any],
    root: Path,
    dataset: DatasetSource,
    test_file: SourceFile,
    label_names: dict[int, str],
) -> tuple[tuple[SelectedModel, ...], set[str], dict[str, bytes]]:
    """Validate all selections and restore models before any test labels are read."""
    if (
        type(selection.get("schema_version")) is not int
        or selection["schema_version"] != 1
        or selection.get("stage") != "final_test_selection"
        or selection.get("preparation") != test_preparation_recipe()
        or selection.get("source")
        != {
            "repo_id": dataset.repo_id,
            "revision": dataset.revision,
            "split": "test",
            "file": asdict(test_file),
        }
        or type(selection.get("expected_rows")) is not int
        or selection["expected_rows"] < 1
        or selection.get("policy")
        != {
            "fit": False,
            "test_driven_selection": False,
            "delivery_seed": 17,
            "primary_metric": "macro_f1",
            "zero_division": 0,
            "batch_size": 32,
            "device": "cpu",
            "num_threads": 1,
        }
    ):
        raise EvaluationError("Unsupported final test selection or preparation policy.")
    _, freeze_bytes = _reference(root, selection.get("freeze"))
    freeze = _object(json.loads(freeze_bytes))
    if (
        freeze.get("schema_version") != 1
        or freeze.get("stage") != "delivery_freeze"
        or freeze.get("status") != "frozen"
        or freeze.get("refit_train_and_validation") is not False
        or freeze.get("reserved_test_evaluation") is not False
        or _object(freeze.get("seed_summary")).get("seeds") != [17, 29, 43]
    ):
        raise EvaluationError("Expected the completed pre-test delivery freeze.")
    data = _object(freeze.get("data"))
    if data.get("label_names") != {str(k): v for k, v in label_names.items()}:
        raise EvaluationError("Frozen labels differ from the published test labels.")
    # Follow the existing freeze's seed evidence, not an independently supplied list.
    _, seeds_bytes = _reference(root, _object(freeze.get("evidence"))["seeds.json"])
    seeds = _object(json.loads(seeds_bytes))
    if seeds.get("seeds") != [17, 29, 43] or seeds.get("data") != data:
        raise EvaluationError("Seed evidence differs from the frozen development data.")
    neural = _object(freeze.get("neural_model"))
    classical = _object(freeze.get("classical_reference"))
    expected: dict[str, tuple[str, int | None, str, dict[str, Any]]] = {}
    seed_runs = seeds.get("runs")
    if not isinstance(seed_runs, list) or len(seed_runs) != 3:
        raise EvaluationError("Expected exactly three frozen neural seed runs.")
    if [_object(row).get("seed") for row in seed_runs] != [17, 29, 43]:
        raise EvaluationError("Expected the predeclared neural seeds 17, 29, 43.")
    for row in seed_runs:
        run = _object(row)
        provenance = _object(run.get("provenance"))
        expected[run["run_id"]] = (
            "mean_pool_mlp",
            run["seed"],
            provenance["manifest_sha256"],
            provenance["files"],
        )
    if neural.get("seed") != 17 or neural.get("run_id") != seed_runs[0]["run_id"]:
        raise EvaluationError("The delivery seed must remain the frozen seed 17.")
    classical_files = _object(classical.get("artifacts"))
    expected[classical["run_id"]] = (
        "tfidf_logreg",
        None,
        classical_files["manifest.json"]["sha256"],
        classical_files,
    )
    entries = selection.get("models")
    if (
        not isinstance(entries, list)
        or len(entries) != 4
        or any(not isinstance(entry, dict) for entry in entries)
        or {entry.get("run_id") for entry in entries} != set(expected)
    ):
        raise EvaluationError(
            "Test campaign must include exactly the four frozen model runs."
        )
    development_path, development_bytes = _reference(
        root, selection.get("development_manifest")
    )
    if sha256(development_bytes) != data["manifest_sha256"]:
        raise EvaluationError("Development manifest differs from the delivery freeze.")
    groups: set[str] = set()
    train_counts: Counter[int] = Counter()
    for split in ("train", "val"):
        loaded = load_split(
            development_path.parent,
            split,
            expected_manifest_sha256=data["manifest_sha256"],
        )
        if (
            loaded.records_sha256 != data[f"{split}_sha256"]
            or len(loaded.records) != data[f"{split}_rows"]
        ):
            raise EvaluationError(
                "Development partition differs from the delivery freeze."
            )
        groups.update(row.group_id for row in loaded.records)
        if split == "train":
            train_counts.update(loaded.targets)
    majority_path, majority_bytes = _reference(root, selection.get("majority_manifest"))
    majority_run = _object(json.loads(majority_bytes))
    if (
        majority_run.get("schema_version") != 1
        or majority_run.get("stage") != "baseline_run"
        or majority_run.get("data") != data
        or _object(majority_run.get("recipe")).get("family") != "majority"
        or majority_run["recipe"].get("fit_split") != "train"
        or majority_run["recipe"].get("evaluation_split") != "val"
    ):
        raise EvaluationError(
            "Majority baseline must use the same frozen development data."
        )
    _, majority_model_bytes = _reference(
        root,
        {
            "path": (majority_path.parent / "model.json").relative_to(root).as_posix(),
            **_object(majority_run["files"]["model.json"]),
        },
    )
    majority = MajorityClassifier.from_dict(json.loads(majority_model_bytes))
    if majority.label_ids != tuple(
        sorted(label_names)
    ) or majority.class_counts != tuple(
        train_counts[label] for label in majority.label_ids
    ):
        raise EvaluationError(
            "Majority fitted state differs from the training contract."
        )
    validate_model_id(majority_path.parent.name)
    models = [SelectedModel(majority_path.parent.name, "majority", None, majority)]
    evidence = {
        "freeze.json": freeze_bytes,
        "seeds.json": seeds_bytes,
        "development-manifest.json": development_bytes,
        "majority-manifest.json": majority_bytes,
        "majority-model.json": majority_model_bytes,
    }
    for entry in entries:
        run_id = entry["run_id"]
        validate_model_id(run_id)
        family, seed, run_hash, inventory = expected[run_id]
        path, content = _reference(root, entry.get("bundle_manifest"))
        manifest = verify_bundle(path.parent, expected_manifest_sha256=sha256(content))
        if (
            manifest.model_id != run_id
            or manifest.source_run_id != run_id
            or manifest.family != family
            or manifest.source_manifest_sha256 != run_hash
            or manifest.labels.to_dict() != data["label_names"]
        ):
            raise EvaluationError("Bundle is not one of the frozen selected runs.")
        for name, payload in manifest.files.items():
            original = "checkpoints/best.pt" if name == "model.pt" else name
            ref = inventory[original]
            if payload.to_dict() != {key: ref[key] for key in ("sha256", "size_bytes")}:
                raise EvaluationError(
                    "Bundle payload differs from the frozen model bytes."
                )
        predictor = Predictor.from_bundle(
            path.parent, expected_manifest_sha256=sha256(content)
        )
        models.append(SelectedModel(run_id, family, seed, predictor))
        evidence[f"{run_id}.bundle.json"] = content
    if len({model.model_id for model in models}) != 5:
        raise EvaluationError("All five evaluation model IDs must be distinct.")
    return tuple(models), groups, evidence


def run_test_campaign(
    selection_path: Path,
    output_dir: Path,
    *,
    expected_selection_sha256: str,
    project_dir: Path,
    raw_dir: Path,
    dataset: DatasetSource,
    test_file: SourceFile,
    label_names: dict[int, str],
) -> Path:
    """Evaluate all registered models on all rows; keep failed exposure records.

    Require a new output directory even for technical reproduction. Preflight
    failures never open test data. After preflight, the manifest records that
    test access may have occurred, including failures. Existing results and the
    source selection are never overwritten. No training or tracking is invoked.
    """
    root = project_dir.expanduser().resolve()
    destination = output_dir.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise EvaluationError(
            "Evaluation destination exists; choose a new --output-dir."
        )
    try:
        selection_bytes = _read(selection_path.expanduser())
        if sha256(selection_bytes) != expected_selection_sha256:
            raise EvaluationError(
                "Selection checksum does not match the requested campaign."
            )
        selection = _object(json.loads(selection_bytes))
        models, groups, evidence = _preflight(
            selection, root, dataset, test_file, label_names
        )
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise EvaluationError(f"Test campaign preflight failed: {exc}") from exc

    import torch

    destination.mkdir(parents=True, exist_ok=False)
    state: dict[str, object] = {
        "schema_version": 1,
        "stage": "final_test_evaluation",
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "selection_sha256": expected_selection_sha256,
        "test_access_started": False,
        "new_training_runs": 0,
        "refit_train_and_validation": False,
        "test_driven_selection": False,
        "delivery_run_id": json.loads(evidence["freeze.json"])["neural_model"][
            "run_id"
        ],
        "runtime": selection["policy"],
    }
    manifest_path = destination / "manifest.json"
    write_json(manifest_path, state)
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        (destination / "selection.json").write_bytes(selection_bytes)
        frozen = destination / "evidence"
        frozen.mkdir()
        for name, content in evidence.items():
            (frozen / name).write_bytes(content)
        state["source"] = capture_source(destination, root)
        state["environment"] = {
            **environment(),
            "evaluation_packages": {
                name: version(name) for name in ("pyarrow", "joblib", "threadpoolctl")
            },
        }
        state["test_access_started"] = True
        write_json(manifest_path, state)
        prepared = prepare_published_test(
            _read(raw_dir.expanduser() / dataset.revision / test_file.path),
            dataset,
            test_file,
            label_names,
            expected_rows=selection["expected_rows"],
        )
        overlap = len(groups & {row.group_id for row in prepared.records})
        if overlap:
            raise EvaluationError(
                "Test adapter introduces development overlap; campaign stopped."
            )
        data_dir = destination / "data"
        data_dir.mkdir()
        (data_dir / "test.jsonl").write_bytes(
            jsonl_bytes([asdict(row) for row in prepared.records])
        )
        (data_dir / "changes.jsonl").write_bytes(jsonl_bytes(prepared.changes))
        write_json(
            data_dir / "manifest.json",
            {
                **prepared.manifest,
                "development_overlap_groups": overlap,
                "files": file_inventory(data_dir),
            },
        )
        data_hash = sha256((data_dir / "manifest.json").read_bytes())
        records_hash = sha256((data_dir / "test.jsonl").read_bytes())
        for selected in models:
            run = destination / selected.model_id
            run.mkdir()
            predictions = []
            scores = []
            if isinstance(selected.model, MajorityClassifier):
                predictions = [
                    Prediction(sample_id, label)
                    for sample_id, label in zip(
                        prepared.sample_ids,
                        selected.model.predict(prepared.texts),
                        strict=True,
                    )
                ]
            else:
                for start in range(0, len(prepared.records), 32):
                    rows = prepared.records[start : start + 32]
                    output = selected.model.predict(
                        tuple(row.text for row in rows), batch_size=32
                    )
                    for row, result in zip(rows, output, strict=True):
                        predictions.append(Prediction(row.sample_id, result.label_id))
                        scores.append({"sample_id": row.sample_id, **result.to_dict()})
            prediction_bytes = jsonl_bytes(
                [asdict(prediction) for prediction in predictions]
            )
            (run / "predictions.test.jsonl").write_bytes(prediction_bytes)
            if scores:
                (run / "scores.test.jsonl").write_bytes(jsonl_bytes(scores))
            metrics = evaluate_predictions(prepared, predictions)
            write_json(
                run / "metrics.test.json",
                {
                    "schema_version": 1,
                    "split": "test",
                    "model_id": selected.model_id,
                    "family": selected.family,
                    "seed": selected.seed,
                    "selection_sha256": expected_selection_sha256,
                    "data_manifest_sha256": data_hash,
                    "records_sha256": records_hash,
                    "predictions_sha256": sha256(prediction_bytes),
                    "label_names": {str(k): v for k, v in label_names.items()},
                    "primary_metric": "macro_f1",
                    "zero_division": 0,
                    "confusion_matrix_axes": {"rows": "true", "columns": "predicted"},
                    "metrics": asdict(metrics),
                    "truncated_rows": sum(
                        score["truncated"] is True for score in scores
                    ),
                },
            )
        state["status"] = "completed"
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        state["finished_at_utc"] = datetime.now(UTC).isoformat()
        state["files"] = file_inventory(destination)
        write_json(manifest_path, state)
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(
            previous_determinism, warn_only=previous_warn_only
        )
    return destination
