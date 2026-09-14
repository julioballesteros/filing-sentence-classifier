"""Offline fixtures for source acquisition and frozen development artifacts."""

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


@pytest.fixture
def development_artifact(tmp_path: Path) -> Path:
    """Build a small artifact through the real producers, using synthetic texts."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from filing_sentence_classifier.data.partition import create_development_split
    from filing_sentence_classifier.data.prepare import prepare_training_data

    labels = {0: "specific", 1: "historical", 2: "generic"}
    revision = "e" * 40
    raw = tmp_path / "raw"
    snapshot = raw / revision
    snapshot.mkdir(parents=True)
    rows = [
        {"text": f"Plan {label}-{index}.", "label": label, "label_text": name}
        for label, name in labels.items()
        for index in range(30)
    ]
    rows.extend([dict(rows[0]), dict(rows[30])])
    rows.extend(
        {"text": "Conflicting annotation.", "label": label, "label_text": labels[label]}
        for label in (0, 2)
    )
    pq.write_table(pa.Table.from_pylist(rows), snapshot / "train.parquet")
    pq.write_table(
        pa.table({"text": ["Plan 0-0.", "Reserved example."]}),
        snapshot / "test.parquet",
    )
    files = tuple(
        SourceFile(
            name,
            (snapshot / name).stat().st_size,
            hashlib.sha256((snapshot / name).read_bytes()).hexdigest(),
        )
        for name in ("train.parquet", "test.parquet")
    )
    dataset = DatasetSource("tests/loading", revision, files)
    prepared = prepare_training_data(
        dataset, files[0], labels, raw, tmp_path / "interim"
    )
    return create_development_split(
        dataset, files[0], files[1], labels, prepared, raw, tmp_path / "processed"
    )
