"""Portable bundle schemas, family-specific inventory, and immutable metadata."""

import json
from dataclasses import FrozenInstanceError

import pytest

from filing_sentence_classifier.artifacts import BundleError, BundleFile, BundleManifest
from filing_sentence_classifier.inference.contracts import InputLimits, LabelSet


@pytest.fixture
def manifest():
    return BundleManifest(
        model_id="mlp-v1",
        family="mean_pool_mlp",
        labels=LabelSet(("a", "b", "c")),
        limits=InputLimits(),
        environment={
            "python": "3.13.9",
            "filing-sentence-classifier": "0.1.0",
            "torch": "2.14.0",
        },
        source_run_id="training-v1",
        source_manifest_sha256="a" * 64,
        files={
            name: BundleFile("b" * 64, 10)
            for name in ("config.toml", "model.pt", "encoder.json")
        },
    )


def test_manifest_round_trip_and_defensive_copies(manifest):
    assert (
        BundleManifest.from_dict(json.loads(json.dumps(manifest.to_dict()))) == manifest
    )
    state = manifest.to_dict()
    restored = BundleManifest.from_dict(state)
    state["labels"]["0"] = "changed"
    state["environment"]["torch"] = "different"
    state["files"]["model.pt"]["size_bytes"] = 90
    assert restored == manifest
    with pytest.raises(TypeError):
        manifest.files["model.pt"] = BundleFile("c" * 64, 12)
    with pytest.raises(FrozenInstanceError):
        manifest.model_id = "different"


def test_direct_construction_copies_caller_mappings(manifest):
    files = dict(manifest.files)
    environment = dict(manifest.environment)
    other = BundleManifest(
        manifest.model_id,
        manifest.family,
        manifest.labels,
        manifest.limits,
        environment,
        manifest.source_run_id,
        manifest.source_manifest_sha256,
        files,
    )
    files.clear()
    environment.clear()
    assert other == manifest


@pytest.mark.parametrize("version", [True, False, 0, 2, 1.0, "1", None])
def test_unknown_or_noninteger_bundle_versions_are_rejected(manifest, version):
    state = manifest.to_dict() | {"schema_version": version}
    with pytest.raises(BundleError, match="schema_version"):
        BundleManifest.from_dict(state)


@pytest.mark.parametrize(
    "key",
    [
        "model_id",
        "family",
        "labels",
        "preprocessing",
        "limits",
        "environment",
        "source",
        "files",
    ],
)
def test_missing_root_fields_are_rejected(manifest, key):
    state = manifest.to_dict()
    state.pop(key)
    with pytest.raises(BundleError):
        BundleManifest.from_dict(state)


@pytest.mark.parametrize("family", ["transformer", "majority", [], None])
def test_undeclared_model_families_are_rejected(manifest, family):
    with pytest.raises(BundleError, match="family"):
        BundleManifest.from_dict(manifest.to_dict() | {"family": family})


@pytest.mark.parametrize(
    "field,value",
    [
        ("extra", 1),
        ("model_id", "../mlp"),
        ("labels", {"0": "a", "2": "b"}),
        ("limits", {"max_batch_size": 1}),
        ("environment", []),
        ("files", []),
        ("source", {"run_id": "/absolute/run", "manifest_sha256": "a" * 64}),
        ("source", {"run_id": "run", "manifest_sha256": "invalid"}),
        (
            "source",
            {"run_id": "run", "manifest_sha256": "a" * 64, "path": "/training/data"},
        ),
        ("preprocessing", {"input": "prepared_text", "cleaning_version": "1"}),
        ("preprocessing", {"input": "raw_text", "cleaning_version": "2"}),
        ("preprocessing", {"input": "raw_text", "cleaning_version": True}),
    ],
)
def test_invalid_or_incompatible_metadata_is_rejected(manifest, field, value):
    with pytest.raises(BundleError):
        BundleManifest.from_dict(manifest.to_dict() | {field: value})


@pytest.mark.parametrize(
    "filename",
    [
        "../model.pt",
        "/model.pt",
        "weights/model.pt",
        "C:\\model.pt",
        "model.joblib",
        "manifest.json",
    ],
)
def test_fixed_payload_names_reject_traversal_and_cross_family_files(
    manifest, filename
):
    state = manifest.to_dict()
    state["files"][filename] = state["files"].pop("model.pt")
    with pytest.raises(BundleError, match="payloads"):
        BundleManifest.from_dict(state)


@pytest.mark.parametrize(
    "change",
    ["missing", "extra", "missing_version", "empty_version", "numeric_version"],
)
def test_inventory_and_environment_are_complete(manifest, change):
    state = manifest.to_dict()
    if change == "missing":
        state["files"].pop("encoder.json")
    elif change == "extra":
        state["files"]["notes.txt"] = {"sha256": "a" * 64, "size_bytes": 1}
    elif change == "missing_version":
        state["environment"].pop("torch")
    else:
        state["environment"]["torch"] = "" if change == "empty_version" else 2.14
    with pytest.raises(BundleError):
        BundleManifest.from_dict(state)


@pytest.mark.parametrize(
    "state",
    [
        {"sha256": "z" * 64, "size_bytes": 1},
        {"sha256": "A" * 64, "size_bytes": 1},
        {"sha256": "a" * 64, "size_bytes": True},
        {"sha256": "a" * 64, "size_bytes": 0},
        {"sha256": "a" * 64, "size_bytes": 1.0},
        {"sha256": "a" * 64},
        {"sha256": "a" * 64, "size_bytes": 1, "path": "elsewhere"},
    ],
)
def test_invalid_inventory_entries_fail(state):
    with pytest.raises(BundleError):
        BundleFile.from_dict(state)
