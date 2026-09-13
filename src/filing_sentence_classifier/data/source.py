"""Download and verify the reviewed FLS source snapshot."""

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import hf_hub_download

REPO_ID = "FinanceMTEB/FLS"
REVISION = "39b6719f1d7197df4498fea9fce20d4ad782a083"


@dataclass(frozen=True)
class SourceFile:
    path: str
    size_bytes: int
    sha256: str


# Parquet hashes come from the pinned revision's Hub LFS metadata. The README
# hash was computed from its raw bytes at that same revision.
SOURCE_FILES = (
    SourceFile(
        "README.md",
        441,
        "37c28808134de90e28d1fbd2c8d305a6264e914807fc40be5bf31926b80b54c6",
    ),
    SourceFile(
        "data/train-00000-of-00001.parquet",
        292230,
        "b8eace45f565497fb57740a15d8b72eeee00cf03d878f2caccef526bb0ddc44c",
    ),
    SourceFile(
        "data/test-00000-of-00001.parquet",
        109784,
        "9b1c472161fb8a0bbcb2f65f84afe552e33cd7bba5abad0beeb658b42dbb32ec",
    ),
)


class DownloadError(RuntimeError):
    """The source snapshot could not be downloaded or verified."""


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "source": {
            "repo_id": REPO_ID,
            "repo_type": "dataset",
            "revision": REVISION,
        },
        "files": {
            file.path: {"size_bytes": file.size_bytes, "sha256": file.sha256}
            for file in SOURCE_FILES
        },
    }


def _verify_files(directory: Path) -> None:
    for file in SOURCE_FILES:
        path = directory / file.path
        if path.is_symlink() or not path.is_file():
            raise DownloadError(f"Missing regular source file: {path}")
        if path.stat().st_size != file.size_bytes:
            raise DownloadError(f"Size mismatch for source file: {path}")
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if checksum != file.sha256:
            raise DownloadError(f"SHA-256 mismatch for source file: {path}")


def _verify_snapshot(directory: Path) -> None:
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DownloadError(f"Missing or invalid manifest: {manifest_path}") from exc
    if manifest != _manifest():
        raise DownloadError(
            f"Manifest does not match the pinned source: {manifest_path}"
        )
    _verify_files(directory)


def download_fls(output_dir: Path = Path("data/raw")) -> Path:
    """Return a verified snapshot under output_dir/<revision>.

    Existing snapshots are verified without network access and never overwritten.
    New snapshots are published only after all source files pass verification.
    File integrity is checked here; row-level data validation is a separate step.
    """
    try:
        output_dir = output_dir.expanduser().resolve()
        destination = output_dir / REVISION
        if destination.exists():
            try:
                _verify_snapshot(destination)
            except DownloadError as exc:
                raise DownloadError(
                    f"{exc}. Existing snapshot was left unchanged. "
                    "Move it aside or choose another --output-dir before retrying."
                ) from exc
            return destination

        output_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f".{REVISION}-", dir=output_dir) as temporary:
            staging = Path(temporary) / "snapshot"
            for file in SOURCE_FILES:
                try:
                    hf_hub_download(
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        revision=REVISION,
                        filename=file.path,
                        local_dir=staging,
                        token=False,
                    )
                except Exception as exc:
                    raise DownloadError(
                        f"Could not download {file.path}: {exc}"
                    ) from exc

            _verify_files(staging)
            (staging / "manifest.json").write_text(
                json.dumps(_manifest(), indent=2, sort_keys=True) + "\n",
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
