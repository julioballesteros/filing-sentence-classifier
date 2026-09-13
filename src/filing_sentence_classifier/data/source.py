"""Download and verify snapshots from pinned dataset specifications."""

import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import hf_hub_download

from filing_sentence_classifier.data.spec import DatasetSource


class DownloadError(RuntimeError):
    """The source snapshot could not be downloaded or verified."""


def _manifest(dataset: DatasetSource) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source": {
            "repo_id": dataset.repo_id,
            "repo_type": "dataset",
            "revision": dataset.revision,
        },
        "files": {
            file.path: {"size_bytes": file.size_bytes, "sha256": file.sha256}
            for file in dataset.files
        },
    }


def _verify_files(directory: Path, dataset: DatasetSource) -> None:
    for file in dataset.files:
        path = directory / file.path
        if path.is_symlink() or not path.is_file():
            raise DownloadError(f"Missing regular source file: {path}")
        if path.stat().st_size != file.size_bytes:
            raise DownloadError(f"Size mismatch for source file: {path}")
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if checksum != file.sha256:
            raise DownloadError(f"SHA-256 mismatch for source file: {path}")


def _verify_snapshot(directory: Path, dataset: DatasetSource) -> None:
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DownloadError(f"Missing or invalid manifest: {manifest_path}") from exc
    if manifest != _manifest(dataset):
        raise DownloadError(
            f"Manifest does not match the pinned source: {manifest_path}"
        )
    _verify_files(directory, dataset)


def download_dataset(
    dataset: DatasetSource, output_dir: Path = Path("data/raw")
) -> Path:
    """Return a verified snapshot under output_dir/<revision>.

    Existing snapshots are verified without network access and never overwritten.
    New snapshots are published only after all source files pass verification.
    File integrity is checked here; row-level data validation is a separate step.
    """
    try:
        output_dir = output_dir.expanduser().resolve()
        destination = output_dir / dataset.revision
        if destination.exists():
            try:
                _verify_snapshot(destination, dataset)
            except DownloadError as exc:
                raise DownloadError(
                    f"{exc}. Existing snapshot was left unchanged. "
                    "Move it aside or choose another --output-dir before retrying."
                ) from exc
            return destination

        output_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix=f".{dataset.revision}-", dir=output_dir
        ) as temporary:
            staging = Path(temporary) / "snapshot"
            for file in dataset.files:
                try:
                    hf_hub_download(
                        repo_id=dataset.repo_id,
                        repo_type="dataset",
                        revision=dataset.revision,
                        filename=file.path,
                        local_dir=staging,
                        token=False,
                    )
                except Exception as exc:
                    raise DownloadError(
                        f"Could not download {file.path}: {exc}"
                    ) from exc

            _verify_files(staging, dataset)
            (staging / "manifest.json").write_text(
                json.dumps(_manifest(dataset), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            # Hub download metadata is temporary; the snapshot contains originals
            # and our manifest only, independently of the Hub cache.
            cache = staging / ".cache"
            if cache.exists():
                shutil.rmtree(cache)
            staging.rename(destination)
        return destination
    except OSError as exc:
        raise DownloadError(f"Could not store the source snapshot: {exc}") from exc
