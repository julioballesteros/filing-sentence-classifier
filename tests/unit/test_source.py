"""Exercise acquisition without downloading or inspecting the real dataset."""

import hashlib
import json
from pathlib import Path

import pytest

from filing_sentence_classifier.data import source
from filing_sentence_classifier.data.spec import DatasetSource


def test_download_publishes_verified_originals_and_manifest(
    tmp_path: Path, dataset_source: DatasetSource, hub: list[str]
) -> None:
    snapshot = source.download_dataset(dataset_source, tmp_path)

    assert snapshot == tmp_path / dataset_source.revision
    assert hub == [file.path for file in dataset_source.files]
    assert list(tmp_path.iterdir()) == [snapshot]
    assert not (snapshot / ".cache").exists()
    manifest = json.loads((snapshot / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["source"] == {
        "repo_id": dataset_source.repo_id,
        "repo_type": "dataset",
        "revision": dataset_source.revision,
    }
    assert set(manifest["files"]) == set(hub)
    for name, metadata in manifest["files"].items():
        content = (snapshot / name).read_bytes()
        assert metadata == {
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }


def test_repeat_download_is_offline_and_preserves_files(
    tmp_path: Path,
    dataset_source: DatasetSource,
    hub: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = source.download_dataset(dataset_source, tmp_path)
    before = {p: p.stat().st_mtime_ns for p in snapshot.rglob("*") if p.is_file()}
    monkeypatch.setattr(
        source, "hf_hub_download", lambda **_: pytest.fail("Unexpected network access")
    )

    assert source.download_dataset(dataset_source, tmp_path) == snapshot
    assert {p: p.stat().st_mtime_ns for p in before} == before


@pytest.mark.parametrize("same_size", [False, True])
def test_modified_snapshot_is_rejected_without_overwriting(
    tmp_path: Path, dataset_source: DatasetSource, hub: list[str], same_size: bool
) -> None:
    snapshot = source.download_dataset(dataset_source, tmp_path)
    path = snapshot / dataset_source.files[1].path
    content = path.read_bytes()
    changed = b"x" * len(content) if same_size else b"short"
    path.write_bytes(changed)
    hub.clear()

    with pytest.raises(source.DownloadError, match="left unchanged"):
        source.download_dataset(dataset_source, tmp_path)

    assert path.read_bytes() == changed
    assert hub == []


@pytest.mark.parametrize("manifest_content", [None, "{", "{}"])
def test_missing_or_invalid_manifest_does_not_trigger_redownload(
    tmp_path: Path,
    dataset_source: DatasetSource,
    hub: list[str],
    manifest_content: str | None,
) -> None:
    snapshot = source.download_dataset(dataset_source, tmp_path)
    manifest = snapshot / "manifest.json"
    if manifest_content is None:
        manifest.unlink()
    else:
        manifest.write_text(manifest_content)
    hub.clear()

    with pytest.raises(source.DownloadError, match="left unchanged"):
        source.download_dataset(dataset_source, tmp_path)

    assert hub == []


@pytest.mark.parametrize("failure", ["network", "checksum"])
def test_failed_download_is_not_published_and_can_be_retried(
    tmp_path: Path,
    dataset_source: DatasetSource,
    hub: list[str],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    download = source.hf_hub_download

    def failing_download(**kwargs: object) -> str:
        if failure == "network" and len(hub) == 1:
            raise ConnectionError("Connection interrupted")
        path = download(**kwargs)
        if failure == "checksum":
            file = Path(path)
            file.write_bytes(b"x" * file.stat().st_size)
        return path

    monkeypatch.setattr(source, "hf_hub_download", failing_download)
    with pytest.raises(source.DownloadError):
        source.download_dataset(dataset_source, tmp_path)
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setattr(source, "hf_hub_download", download)
    assert source.download_dataset(dataset_source, tmp_path).is_dir()
