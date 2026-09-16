"""Check atomic metadata writes and honest source provenance."""

import shutil
import subprocess

import pytest

from filing_sentence_classifier.training import artifacts


def test_failed_json_replacement_preserves_previous_manifest(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    artifacts.write_json(path, {"status": "running"})
    original = path.read_bytes()

    def fail_replace(*args):
        raise OSError("replacement failed")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        artifacts.write_json(path, {"status": "completed"})
    assert path.read_bytes() == original
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize("changed_checkout", [False, True])
def test_source_snapshot_checks_imported_code_against_the_claimed_checkout(
    tmp_path, monkeypatch, changed_checkout
):
    initial = tmp_path / "initial"
    artifacts.capture_source(initial, tmp_path)
    project = tmp_path / "project"
    shutil.copytree(initial / "source" / "src", project / "src")
    (project / "pyproject.toml").write_text(
        '[project]\nname = "filing-sentence-classifier"\n'
    )
    (project / "uv.lock").write_text("version = 1\n")
    (project / ".python-version").write_text("3.13\n")
    if changed_checkout:
        (project / "src/filing_sentence_classifier/training/config.py").write_text(
            "# A different installed revision.\n"
        )

    def git(args, **kwargs):
        # Even a clean checkout is insufficient when the imported package differs.
        output = "a" * 40 if "rev-parse" in args else ""
        return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")

    monkeypatch.setattr(artifacts.subprocess, "run", git)
    destination = tmp_path / "snapshot"
    provenance = artifacts.capture_source(destination, project)
    assert provenance["git"]["matches_imported_code"] is not changed_checkout
    assert provenance["git"]["reproducible_from_commit"] is not changed_checkout
    for name, expected in provenance["code_sha256"].items():
        assert (
            artifacts.sha256(
                (
                    destination / "source/src/filing_sentence_classifier" / name
                ).read_bytes()
            )
            == expected
        )
    for name in ("pyproject.toml", "uv.lock", ".python-version"):
        assert (destination / "source" / name).read_bytes() == (
            project / name
        ).read_bytes()
