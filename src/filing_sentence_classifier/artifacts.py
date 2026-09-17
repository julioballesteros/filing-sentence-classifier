"""Versioned inference-bundle metadata and integrity checks, without model loading.

Bundle v1 reuses the existing checkpoint/encoder and fitted joblib formats.
Payload interpretation and runtime dependency compatibility belong to the
family adapters. Integrity checks do not authenticate an untrusted producer.
"""

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Self, cast

from filing_sentence_classifier.data.cleaning import CLEANING_VERSION
from filing_sentence_classifier.inference.contracts import (
    InputLimits,
    LabelSet,
    PredictionContractError,
    validate_model_id,
)

BUNDLE_SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 1_048_576
ModelFamily = Literal["mean_pool_mlp", "tfidf_logreg"]
_PAYLOADS = {
    "mean_pool_mlp": frozenset({"config.toml", "model.pt", "encoder.json"}),
    "tfidf_logreg": frozenset({"config.toml", "model.joblib"}),
}
_DEPENDENCIES = {
    "mean_pool_mlp": {"python", "filing-sentence-classifier", "torch"},
    "tfidf_logreg": {
        "python",
        "filing-sentence-classifier",
        "scikit-learn",
        "joblib",
        "numpy",
        "scipy",
        "threadpoolctl",
    },
}


class BundleError(ValueError):
    """Bundle metadata, format version, or local file integrity is invalid."""


def _object(value: object, fields: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise BundleError(f"Unexpected or missing {name} fields.")
    return cast(dict[str, object], value)


def _check_hash(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise BundleError("Expected a lowercase SHA-256 hex digest.")


@dataclass(frozen=True)
class BundleFile:
    """Expected bytes of one fixed-name payload, independent of its location."""

    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _check_hash(self.sha256)
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise BundleError("Payload size_bytes must be a positive integer.")

    def to_dict(self) -> dict[str, object]:
        return {"sha256": self.sha256, "size_bytes": self.size_bytes}

    @classmethod
    def from_dict(cls, state: object) -> Self:
        value = _object(state, {"sha256", "size_bytes"}, "file inventory")
        return cls(cast(str, value["sha256"]), cast(int, value["size_bytes"]))


@dataclass(frozen=True)
class BundleManifest:
    """Portable metadata; payload filenames are fixed by the model family.

    Class IDs follow the labels' explicit order. Environment versions describe
    creation; parsing them does not assert compatibility with the current host.
    Source provenance is an identity, never a path required at inference time.
    model_id identifies a model version: changing any bundle bytes needs a new ID.
    """

    model_id: str
    family: ModelFamily
    labels: LabelSet
    limits: InputLimits
    environment: Mapping[str, str]
    source_run_id: str
    source_manifest_sha256: str
    files: Mapping[str, BundleFile]

    def __post_init__(self) -> None:
        try:
            validate_model_id(self.model_id)
            validate_model_id(self.source_run_id)
        except PredictionContractError as exc:
            raise BundleError(str(exc)) from exc
        _check_hash(self.source_manifest_sha256)
        if not isinstance(self.family, str) or self.family not in _PAYLOADS:
            raise BundleError("Unsupported model family.")
        if not isinstance(self.labels, LabelSet) or not isinstance(
            self.limits, InputLimits
        ):
            raise BundleError("Expected validated labels and input limits.")
        if (
            not isinstance(self.environment, Mapping)
            or not _DEPENDENCIES[self.family] <= self.environment.keys()
            or any(
                not isinstance(key, str)
                or not key.strip()
                or not isinstance(value, str)
                or not value.strip()
                for key, value in self.environment.items()
            )
        ):
            raise BundleError("Expected recorded Python and model dependency versions.")
        if (
            not isinstance(self.files, Mapping)
            or set(self.files) != _PAYLOADS[self.family]
            or any(not isinstance(value, BundleFile) for value in self.files.values())
        ):
            raise BundleError(
                f"Expected exactly these payloads: {sorted(_PAYLOADS[self.family])}."
            )
        object.__setattr__(
            self, "environment", MappingProxyType(dict(self.environment))
        )
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "model_id": self.model_id,
            "family": self.family,
            "labels": self.labels.to_dict(),
            "preprocessing": {
                "input": "raw_text",
                "cleaning_version": CLEANING_VERSION,
            },
            "limits": self.limits.to_dict(),
            "environment": dict(self.environment),
            "source": {
                "run_id": self.source_run_id,
                "manifest_sha256": self.source_manifest_sha256,
            },
            "files": {name: value.to_dict() for name, value in self.files.items()},
        }

    @classmethod
    def from_dict(cls, state: object) -> Self:
        value = _object(
            state,
            {
                "schema_version",
                "model_id",
                "family",
                "labels",
                "preprocessing",
                "limits",
                "environment",
                "source",
                "files",
            },
            "manifest",
        )
        if (
            type(value["schema_version"]) is not int
            or value["schema_version"] != BUNDLE_SCHEMA_VERSION
        ):
            raise BundleError("Unsupported bundle schema_version.")
        if value["preprocessing"] != {
            "input": "raw_text",
            "cleaning_version": CLEANING_VERSION,
        }:
            raise BundleError("Unsupported preprocessing recipe.")
        source = _object(value["source"], {"run_id", "manifest_sha256"}, "source")
        if not isinstance(value["environment"], dict) or not isinstance(
            value["files"], dict
        ):
            raise BundleError("Expected environment and files objects.")
        try:
            return cls(
                model_id=cast(str, value["model_id"]),
                family=cast(ModelFamily, value["family"]),
                labels=LabelSet.from_dict(value["labels"]),
                limits=InputLimits.from_dict(value["limits"]),
                environment=value["environment"],
                source_run_id=cast(str, source["run_id"]),
                source_manifest_sha256=cast(str, source["manifest_sha256"]),
                files={
                    name: BundleFile.from_dict(item)
                    for name, item in value["files"].items()
                },
            )
        except PredictionContractError as exc:
            raise BundleError(str(exc)) from exc


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"Duplicate JSON key: {key}.")
        result[key] = value
    return result


def _invalid_constant(value: str) -> object:
    raise BundleError(f"Non-finite JSON number: {value}.")


def _parse_json_object(content: bytes) -> dict[str, object]:
    value = json.loads(
        content.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant,
    )
    if not isinstance(value, dict):
        raise BundleError("Expected a JSON object.")
    return cast(dict[str, object], value)


def verify_bundle(
    directory: Path, *, expected_manifest_sha256: str | None = None
) -> BundleManifest:
    """Check a flat bundle's manifest and complete payload inventory on disk.

    Fixed basenames reject absolute paths and traversal. Reject symlinks,
    directories and other non-regular entries. Hash payloads in bounded chunks;
    never import torch/sklearn/joblib or deserialize model payloads here. Checks
    describe these files at verification time; adapters must load verified bytes.
    """
    if expected_manifest_sha256 is not None:
        _check_hash(expected_manifest_sha256)
    directory = directory.expanduser()
    try:
        if directory.is_symlink() or not directory.is_dir():
            raise BundleError("Expected a bundle directory, not a symbolic link.")
        manifest_path = directory / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise BundleError("Expected a regular manifest.json file.")
        with manifest_path.open("rb") as stream:
            content = stream.read(MAX_MANIFEST_BYTES + 1)
        if len(content) > MAX_MANIFEST_BYTES:
            raise BundleError("Bundle manifest exceeds the 1 MiB limit.")
        if (
            expected_manifest_sha256 is not None
            and hashlib.sha256(content).hexdigest() != expected_manifest_sha256
        ):
            raise BundleError("Bundle manifest checksum mismatch.")
        state = _parse_json_object(content)
        manifest = BundleManifest.from_dict(state)
        if {path.name for path in directory.iterdir()} != {
            "manifest.json",
            *manifest.files,
        }:
            raise BundleError("Bundle has missing or unexpected files.")
        for name, expected in manifest.files.items():
            path = directory / name
            if path.is_symlink() or not path.is_file():
                raise BundleError(f"Expected a regular payload file: {name}.")
            if path.stat().st_size != expected.size_bytes:
                raise BundleError(f"Payload size mismatch: {name}.")
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as stream:
                while chunk := stream.read(1_048_576):
                    digest.update(chunk)
                    size += len(chunk)
            if size != expected.size_bytes or digest.hexdigest() != expected.sha256:
                raise BundleError(f"Payload checksum or size mismatch: {name}.")
        return manifest
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, BundleError):
            raise
        raise BundleError(f"Could not verify bundle: {exc}") from exc
