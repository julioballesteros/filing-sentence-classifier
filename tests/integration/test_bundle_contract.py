"""Verify portable bundle bytes without importing runtimes or loading models."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import filing_sentence_classifier
from filing_sentence_classifier.artifacts import (
    MAX_MANIFEST_BYTES,
    BundleError,
    verify_bundle,
)


@pytest.fixture(params=["mean_pool_mlp", "tfidf_logreg"])
def bundle(tmp_path, request):
    directory = tmp_path / "bundle"
    directory.mkdir()
    family = request.param
    # Deliberately opaque fixtures: the verifier must not deserialize these bytes.
    payloads = {"config.toml": b"opaque configuration"}
    if family == "mean_pool_mlp":
        payloads |= {
            "model.pt": b"opaque checkpoint",
            "encoder.json": b"opaque encoder",
        }
        dependencies = ["torch"]
    else:
        payloads["model.joblib"] = b"opaque fitted pipeline"
        dependencies = ["scikit-learn", "joblib", "numpy", "scipy", "threadpoolctl"]
    for name, content in payloads.items():
        (directory / name).write_bytes(content)
    state = {
        "schema_version": 1,
        "model_id": "synthetic-v1",
        "family": family,
        "labels": {"0": "outlook", "1": "history"},
        "limits": {"max_characters": 10000, "max_batch_size": 256},
        "preprocessing": {"input": "raw_text", "cleaning_version": "1"},
        "environment": {
            "python": "3.13.9",
            "filing-sentence-classifier": "0.1.0",
            **dict.fromkeys(dependencies, "1.0"),
        },
        "source": {"run_id": "synthetic-training", "manifest_sha256": "a" * 64},
        "files": {
            name: {
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
            for name, content in payloads.items()
        },
    }
    (directory / "manifest.json").write_text(json.dumps(state), encoding="utf-8")
    return directory


def test_bundle_can_move_and_verify_from_an_unrelated_working_directory(
    bundle, tmp_path
):
    expected = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    original = verify_bundle(bundle, expected_manifest_sha256=expected)
    relocated = tmp_path / "other-name"
    shutil.move(str(bundle), relocated)
    assert verify_bundle(relocated, expected_manifest_sha256=expected) == original
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    package_parent = Path(filing_sentence_classifier.__file__).resolve().parent.parent
    script = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from filing_sentence_classifier.artifacts import verify_bundle
from filing_sentence_classifier.inference.contracts import prepare_texts
assert verify_bundle(Path(sys.argv[2]), expected_manifest_sha256=sys.argv[3]).model_id == 'synthetic-v1'
assert prepare_texts(['  We expect growth. ']) == ('We expect growth.',)
assert not {'torch', 'sklearn', 'joblib', 'mlflow', 'numpy', 'pyarrow'} & sys.modules.keys()
"""
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            script,
            str(package_parent),
            str(relocated),
            expected,
        ],
        cwd=elsewhere,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "change",
    ["same_size_corruption", "different_size", "missing", "extra", "directory"],
)
def test_changed_or_incomplete_payloads_fail(bundle, change):
    path = bundle / "config.toml"
    if change == "same_size_corruption":
        path.write_bytes(b"x" * path.stat().st_size)
    elif change == "different_size":
        path.write_bytes(b"short")
    elif change == "extra":
        (bundle / "notes.txt").write_text("undeclared")
    else:
        path.unlink()
        if change == "directory":
            path.mkdir()
    with pytest.raises(BundleError):
        verify_bundle(bundle)


@pytest.mark.parametrize("target", ["root", "manifest.json", "config.toml"])
def test_symlinks_cannot_reference_other_files(bundle, tmp_path, target):
    if target == "root":
        link = tmp_path / "link"
        link.symlink_to(bundle, target_is_directory=True)
        directory = link
    else:
        path = bundle / target
        moved = tmp_path / "external"
        path.rename(moved)
        path.symlink_to(moved)
        directory = bundle
    with pytest.raises(BundleError, match="regular|symbolic"):
        verify_bundle(directory)


def test_manifest_checksum_can_pin_metadata_independently_of_payload_hashes(bundle):
    path = bundle / "manifest.json"
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    state = json.loads(path.read_text())
    state["labels"]["0"] = "changed label"
    path.write_text(json.dumps(state))
    with pytest.raises(BundleError, match="manifest checksum"):
        verify_bundle(bundle, expected_manifest_sha256=expected)
    with pytest.raises(BundleError, match="SHA-256"):
        verify_bundle(bundle, expected_manifest_sha256="invalid")


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"\xff",
        b"[]",
        b'{"schema_version":1,"schema_version":1}',
        b'{"nested":{"same":1,"same":2}}',
        b'{"bad": NaN}',
        b'{"bad": Infinity}',
    ],
)
def test_invalid_json_and_duplicate_keys_are_rejected(bundle, content):
    (bundle / "manifest.json").write_bytes(content)
    with pytest.raises(BundleError):
        verify_bundle(bundle)


def test_oversized_manifest_and_missing_bundle_have_clear_errors(bundle, tmp_path):
    (bundle / "manifest.json").write_bytes(b" " * (MAX_MANIFEST_BYTES + 1))
    with pytest.raises(BundleError, match="1 MiB"):
        verify_bundle(bundle)
    with pytest.raises(BundleError, match="directory"):
        verify_bundle(tmp_path / "absent")
