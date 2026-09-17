"""Export verified run artifacts without fitting or deserializing model payloads."""

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from filing_sentence_classifier.artifacts import (
    MAX_MANIFEST_BYTES,
    BundleError,
    BundleFile,
    BundleManifest,
    _check_hash,
    _parse_json_object,
    verify_bundle,
)
from filing_sentence_classifier.baselines.config import TfidfConfig
from filing_sentence_classifier.data.cleaning import cleaning_recipe
from filing_sentence_classifier.inference.contracts import (
    DEFAULT_INPUT_LIMITS,
    InputLimits,
    LabelSet,
    validate_model_id,
)
from filing_sentence_classifier.text.encoding import TextEncoder
from filing_sentence_classifier.training.config import TrainingConfig


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise BundleError(f"Expected an object for {name}.")
    return cast(dict[str, object], value)


def _regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise BundleError(f"Expected a regular source file: {path.name}.")


def _read_metadata(path: Path, *, expected_hash: str | None = None) -> bytes:
    _regular_file(path)
    with path.open("rb") as stream:
        content = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(content) > MAX_MANIFEST_BYTES:
        raise BundleError(f"Metadata exceeds the 1 MiB limit: {path.name}.")
    if expected_hash is not None:
        _check_hash(expected_hash)
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise BundleError(f"Source manifest checksum mismatch: {path.name}.")
    return content


def _copy_payload(source: Path, target: Path, expected: BundleFile) -> None:
    """Verify the exact bytes written to staging, including concurrent changes."""
    _regular_file(source)
    if source.stat().st_size != expected.size_bytes:
        raise BundleError(f"Source payload size mismatch: {source.name}.")
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as reader, target.open("xb") as writer:
        while chunk := reader.read(1_048_576):
            size += len(chunk)
            if size > expected.size_bytes:
                raise BundleError(f"Source payload grew during export: {source.name}.")
            digest.update(chunk)
            writer.write(chunk)
    if size != expected.size_bytes or digest.hexdigest() != expected.sha256:
        raise BundleError(f"Source payload checksum mismatch: {source.name}.")


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise BundleError(f"Expected a positive integer for {name}.")
    return value


def _check_data_metadata(content: bytes, labels: LabelSet) -> None:
    data = _parse_json_object(content)
    if (
        type(data.get("schema_version")) is not int
        or data["schema_version"] != 1
        or data.get("stage") != "development_splits"
        or data.get("cleaning_recipe") != cleaning_recipe()
        or LabelSet.from_dict(data.get("label_names")) != labels
    ):
        raise BundleError("Dataset metadata has incompatible cleaning or labels.")


def _check_neural_metadata(
    run: dict[str, object], staging: Path, labels: LabelSet
) -> None:
    config = TrainingConfig.from_toml(_read_metadata(staging / "config.toml"))
    if config != TrainingConfig.from_dict(run.get("config")):
        raise BundleError("Saved configuration does not match the run manifest.")
    encoder = TextEncoder.from_dict(
        _parse_json_object(_read_metadata(staging / "encoder.json"))
    )
    preprocessing = _object(run.get("preprocessing"), "preprocessing")
    model = _object(run.get("model"), "model")
    if (
        _positive_integer(preprocessing.get("vocabulary_size"), "vocabulary_size")
        != len(encoder.vocabulary)
        or _positive_integer(preprocessing.get("max_length"), "max_length")
        != encoder.max_length
        or _positive_integer(model.get("num_classes"), "num_classes")
        != len(labels.names)
    ):
        raise BundleError("Saved encoder or class count does not match the run.")


def _check_tfidf_metadata(
    run: dict[str, object], staging: Path, metadata: bytes, labels: LabelSet
) -> None:
    config = TfidfConfig.from_toml(_read_metadata(staging / "config.toml"))
    if config.recipe() != run.get("recipe"):
        raise BundleError("Saved configuration does not match the run recipe.")
    model = _parse_json_object(metadata)
    classes = model.get("class_ids")
    if (
        type(model.get("schema_version")) is not int
        or model["schema_version"] != 1
        or model.get("family") != "tfidf_logreg"
        or model.get("serialization") != "joblib"
        or model.get("model_file") != "model.joblib"
        or model.get("converged") is not True
        or not isinstance(classes, list)
        or any(type(label) is not int for label in classes)
        or classes != list(range(len(labels.names)))
    ):
        raise BundleError("TF-IDF model metadata has incompatible classes or status.")
    _positive_integer(model.get("vocabulary_size"), "vocabulary_size")


@contextmanager
def _publication_lock(destination: Path) -> Iterator[None]:
    """Serialize exporters targeting the same path without locking model loading."""
    lock = destination.with_name(f".{destination.name}.export.lock")
    try:
        stream = lock.open("x")
    except FileExistsError as exc:
        raise BundleError(f"An export lock already exists: {lock}.") from exc
    try:
        with stream:
            yield
    finally:
        lock.unlink()


def export_bundle(
    run_dir: Path,
    output_dir: Path,
    *,
    model_id: str,
    data_manifest: Path | None = None,
    limits: InputLimits = DEFAULT_INPUT_LIMITS,
    expected_manifest_sha256: str | None = None,
) -> Path:
    """Publish a deterministic, self-contained bundle from an existing v1 run.

    Verify consumed files against the source manifest, then copy unchanged bytes.
    Dataset metadata supplies verified cleaning and labels; neural runs already
    contain this JSON, while legacy baseline runs require data_manifest explicitly.
    No examples, predictions, ML runtimes, training, tracking, or network are used.
    Binary payload semantics are checked later by the loading adapters.

    Identical exports verify existing bytes without rewriting them. Different or
    incomplete destinations fail. Cooperating exporters use a sibling lock, and
    new bundles are published by renaming a completed sibling staging directory.
    """
    try:
        validate_model_id(model_id)
        if not isinstance(limits, InputLimits):
            raise BundleError("Expected validated input limits.")
        source = run_dir.expanduser().absolute()
        destination = output_dir.expanduser().absolute()
        if source.is_symlink() or not source.is_dir():
            raise BundleError("Expected a run directory, not a symbolic link.")
        if destination.is_symlink():
            raise BundleError("Bundle destination must not be a symbolic link.")
        if source.resolve().is_relative_to(
            destination.resolve()
        ) or destination.resolve().is_relative_to(source.resolve()):
            raise BundleError("Run and bundle directories must not overlap.")
        content = _read_metadata(
            source / "manifest.json", expected_hash=expected_manifest_sha256
        )
        source_hash = hashlib.sha256(content).hexdigest()
        run = _parse_json_object(content)
        recipe = _object(run.get("recipe"), "recipe")
        family = recipe.get("family")
        if (
            type(run.get("schema_version")) is not int
            or run["schema_version"] != 1
            or family not in ("mean_pool_mlp", "tfidf_logreg")
            or recipe.get("fit_split") != "train"
            or recipe.get("evaluation_split") != "val"
        ):
            raise BundleError("Unsupported source run schema or model family.")
        if family == "mean_pool_mlp":
            if (
                run.get("stage") != "neural_training_run"
                or run.get("status") != "completed"
                or recipe.get("selection_metric") != "macro_f1"
                or recipe.get("checkpoint_tie_break") != "earliest_epoch"
            ):
                raise BundleError("Expected a completed neural training run.")
            payloads = {
                "config.toml": "config.toml",
                "encoder.json": "encoder.json",
                "model.pt": "checkpoints/best.pt",
            }
            if (source / "checkpoints").is_symlink():
                raise BundleError("Checkpoint directory must not be a symbolic link.")
            environment = _object(run.get("environment"), "environment")
            versions = {
                **_object(environment.get("packages"), "environment.packages"),
                "python": environment.get("python"),
            }
            run_id = run.get("run_id")
        else:
            if run.get("stage") != "baseline_run":
                raise BundleError("Expected a TF-IDF baseline run.")
            payloads = {"config.toml": "config.toml", "model.joblib": "model.joblib"}
            versions = _object(run.get("environment"), "environment")
            # Legacy baseline manifests predate explicit run IDs.
            run_id = run.get("run_id", source.name)

        inventory = _object(run.get("files"), "files")
        files = {
            target: BundleFile.from_dict(inventory.get(name))
            for target, name in payloads.items()
        }
        data = _object(run.get("data"), "data")
        labels = LabelSet.from_dict(data.get("label_names"))
        if data_manifest is None:
            if "data-manifest.json" not in inventory:
                raise BundleError(
                    "This run needs --data-manifest pointing to its development "
                    "manifest.json; no dataset rows are required."
                )
            data_manifest = source / "data-manifest.json"
        data_path = data_manifest.expanduser().absolute()
        if data_path.resolve().is_relative_to(destination.resolve()):
            raise BundleError(
                "Dataset metadata must be outside the bundle destination."
            )
        data_hash = data.get("manifest_sha256")
        _check_hash(cast(str, data_hash))
        data_bytes = _read_metadata(data_path, expected_hash=cast(str, data_hash))
        if data_path == source / "data-manifest.json":
            if BundleFile.from_dict(inventory.get("data-manifest.json")) != BundleFile(
                cast(str, data_hash), len(data_bytes)
            ):
                raise BundleError("Dataset metadata inventory does not match the run.")
        _check_data_metadata(data_bytes, labels)
        manifest = BundleManifest(
            model_id=model_id,
            family=family,
            labels=labels,
            limits=limits,
            environment=cast(dict[str, str], versions),
            source_run_id=cast(str, run_id),
            source_manifest_sha256=source_hash,
            files=files,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (
            _publication_lock(destination),
            TemporaryDirectory(
                prefix=f".{destination.name}.export-", dir=destination.parent
            ) as temporary,
        ):
            staging = Path(temporary) / "bundle"
            staging.mkdir()
            for target, name in payloads.items():
                _copy_payload(source / name, staging / target, files[target])
            if family == "mean_pool_mlp":
                _check_neural_metadata(run, staging, labels)
            else:
                metadata_path = Path(temporary) / "model.json"
                expected = BundleFile.from_dict(inventory.get("model.json"))
                if expected.size_bytes > MAX_MANIFEST_BYTES:
                    raise BundleError("Model metadata exceeds the 1 MiB limit.")
                _copy_payload(source / "model.json", metadata_path, expected)
                _check_tfidf_metadata(
                    run, staging, _read_metadata(metadata_path), labels
                )
            manifest_bytes = (
                json.dumps(
                    manifest.to_dict(), indent=2, sort_keys=True, allow_nan=False
                )
                + "\n"
            ).encode("utf-8")
            (staging / "manifest.json").write_bytes(manifest_bytes)
            expected_hash = hashlib.sha256(manifest_bytes).hexdigest()
            verify_bundle(staging, expected_manifest_sha256=expected_hash)
            if destination.exists() or destination.is_symlink():
                try:
                    verify_bundle(destination, expected_manifest_sha256=expected_hash)
                except BundleError as exc:
                    raise BundleError(
                        "Existing bundle differs; left unchanged. Choose another "
                        "--output-dir and model ID for changed contents."
                    ) from exc
            else:
                staging.rename(destination)
        return destination
    except (OSError, ValueError) as exc:
        if isinstance(exc, BundleError):
            raise
        raise BundleError(f"Could not export bundle: {exc}") from exc
