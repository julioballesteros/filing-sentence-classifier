"""Shared fixtures for source acquisition tests."""

import hashlib
from pathlib import Path

import pytest

from filing_sentence_classifier.data import source
from filing_sentence_classifier.data.spec import DatasetSource, SourceFile


@pytest.fixture
def file_contents() -> dict[str, bytes]:
    # Synthetic bytes: these tests validate transport, not Parquet contents.
    return {
        "README.md": b"Synthetic dataset metadata\n",
        "data/train.parquet": b"Synthetic training data\n",
        "data/test.parquet": b"Synthetic test data\n",
    }


@pytest.fixture
def dataset_source(file_contents: dict[str, bytes]) -> DatasetSource:
    return DatasetSource(
        repo_id="tests/synthetic-dataset",
        revision="a" * 40,
        files=tuple(
            SourceFile(path, len(content), hashlib.sha256(content).hexdigest())
            for path, content in file_contents.items()
        ),
    )


@pytest.fixture
def hub(
    monkeypatch: pytest.MonkeyPatch,
    dataset_source: DatasetSource,
    file_contents: dict[str, bytes],
) -> list[str]:
    calls: list[str] = []

    def download(
        *,
        repo_id: str,
        repo_type: str,
        revision: str,
        filename: str,
        local_dir: Path,
        token: bool,
    ) -> str:
        assert (repo_id, repo_type, revision) == (
            dataset_source.repo_id,
            "dataset",
            dataset_source.revision,
        )
        assert token is False
        calls.append(filename)
        path = local_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(file_contents[filename])
        cache = local_dir / ".cache" / "huggingface"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "metadata").write_text("temporary metadata")
        return str(path)

    monkeypatch.setattr(source, "hf_hub_download", download)
    return calls
