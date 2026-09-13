"""Shared fixtures for source acquisition tests."""

import hashlib
from pathlib import Path

import pytest

from filing_sentence_classifier.data import source


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    # Synthetic bytes: these tests validate transport, not Parquet contents.
    payloads = {
        file.path: f"synthetic content for {file.path}\n".encode()
        for file in source.SOURCE_FILES
    }
    monkeypatch.setattr(
        source,
        "SOURCE_FILES",
        tuple(
            source.SourceFile(path, len(content), hashlib.sha256(content).hexdigest())
            for path, content in payloads.items()
        ),
    )
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
            "FinanceMTEB/FLS",
            "dataset",
            "39b6719f1d7197df4498fea9fce20d4ad782a083",
        )
        assert token is False
        calls.append(filename)
        path = local_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[filename])
        cache = local_dir / ".cache" / "huggingface"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "metadata").write_text("temporary metadata")
        return str(path)

    monkeypatch.setattr(source, "hf_hub_download", download)
    return calls
