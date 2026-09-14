"""Exercise train-only vocabulary builds and reproducible artifact publication."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from filing_sentence_classifier.cli import app
from filing_sentence_classifier.data.loading import DataLoadError, load_split
from filing_sentence_classifier.data.vocabulary import build_vocabulary
from filing_sentence_classifier.text.tokenization import tokenization_recipe, tokenize
from filing_sentence_classifier.text.vocabulary import (
    UNK_ID,
    Vocabulary,
    VocabularyError,
)


def artifact_bytes(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in directory.iterdir()}


def test_cli_builds_a_portable_reproducible_vocabulary(
    development_artifact: Path, tmp_path: Path
) -> None:
    train = load_split(development_artifact, "train")
    before = artifact_bytes(development_artifact)
    output = tmp_path / "vocabulary"
    args = [
        "build-vocabulary",
        "--data-dir",
        str(development_artifact),
        "--output-dir",
        str(output),
        "--min-frequency",
        "2",
        "--max-size",
        "10",
        "--manifest-sha256",
        train.manifest_sha256,
    ]
    runner = CliRunner()
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    payloads = artifact_bytes(output)
    assert set(payloads) == {"vocabulary.json", "manifest.json"}
    vocabulary = Vocabulary.from_dict(json.loads(payloads["vocabulary.json"]))
    documents = tuple(tokenize(text) for text in train.texts)
    assert vocabulary == Vocabulary.fit(documents, max_size=10)
    manifest = json.loads(payloads["manifest.json"])
    assert manifest["fit_split"] == "train"
    assert manifest["data"] == {
        "manifest_sha256": train.manifest_sha256,
        "train_sha256": train.records_sha256,
        "train_rows": len(train.records),
    }
    assert manifest["tokenization"] == tokenization_recipe()
    assert manifest["vocabulary"] == vocabulary.recipe()
    stats = manifest["statistics"]
    total = sum(len(document) for document in documents)
    unknown = sum(vocabulary[token] == UNK_ID for row in documents for token in row)
    assert stats["train_tokens"] == total
    assert stats["train_unknown_tokens"] == unknown
    assert stats["train_unknown_rate"] == pytest.approx(unknown / total)
    assert stats["vocabulary_size"] == len(vocabulary) == 10
    metadata = manifest["files"]["vocabulary.json"]
    assert metadata["sha256"] == hashlib.sha256(payloads["vocabulary.json"]).hexdigest()
    assert metadata["size_bytes"] == len(payloads["vocabulary.json"])
    mtimes = {path.name: path.stat().st_mtime_ns for path in output.iterdir()}
    assert runner.invoke(app, args).exit_code == 0
    assert artifact_bytes(output) == payloads
    assert mtimes == {path.name: path.stat().st_mtime_ns for path in output.iterdir()}
    # Hash randomization and a different working directory must not change IDs.
    for seed in ("1", "2026"):
        rebuilt = tmp_path / f"rebuilt-{seed}"
        rebuilt_args = list(args)
        rebuilt_args[4] = str(rebuilt)
        subprocess.run(
            [sys.executable, "-m", "filing_sentence_classifier", *rebuilt_args],
            cwd=tmp_path,
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        )
        assert artifact_bytes(rebuilt) == payloads
    assert artifact_bytes(development_artifact) == before


def test_build_needs_only_train_and_never_opens_validation_or_source_data(
    development_artifact: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (development_artifact / "val.jsonl").unlink()
    original_read = Path.read_bytes
    reads = []

    def bounded_read(path: Path) -> bytes:
        assert path.name not in {"val.jsonl", "test.jsonl"}
        assert path.suffix != ".parquet"
        assert "raw" not in path.parts and "interim" not in path.parts
        reads.append(path)
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", bounded_read)
    artifact = build_vocabulary(development_artifact, tmp_path / "vocabulary")
    assert development_artifact / "train.jsonl" in reads
    restored = Vocabulary.from_dict(
        json.loads((artifact / "vocabulary.json").read_text())
    )
    assert restored["validationonly"] == UNK_ID
    assert "validationonly" not in restored.token_to_id


@pytest.mark.parametrize("filename", ["vocabulary.json", "manifest.json"])
def test_existing_changed_artifacts_are_not_overwritten(
    development_artifact: Path, tmp_path: Path, filename: str
) -> None:
    output = build_vocabulary(development_artifact, tmp_path / "vocabulary")
    (output / filename).write_text("changed")
    before = artifact_bytes(output)
    with pytest.raises(VocabularyError, match="Existing vocabulary differs"):
        build_vocabulary(development_artifact, output)
    assert artifact_bytes(output) == before
    assert not list(tmp_path.glob(".vocabulary-*"))


def test_changed_parameters_require_a_new_destination(
    development_artifact: Path, tmp_path: Path
) -> None:
    output = build_vocabulary(development_artifact, tmp_path / "vocabulary")
    before = artifact_bytes(output)
    with pytest.raises(VocabularyError, match="Existing vocabulary differs"):
        build_vocabulary(development_artifact, output, max_size=3)
    assert artifact_bytes(output) == before


def test_invalid_training_data_or_limits_do_not_publish_an_artifact(
    development_artifact: Path, tmp_path: Path
) -> None:
    output = tmp_path / "vocabulary"
    with pytest.raises(DataLoadError, match="Manifest checksum"):
        build_vocabulary(
            development_artifact, output, expected_manifest_sha256="0" * 64
        )
    with pytest.raises(VocabularyError, match="No training tokens"):
        build_vocabulary(development_artifact, output, min_frequency=10000)
    (development_artifact / "train.jsonl").write_text("corrupted")
    result = CliRunner().invoke(
        app,
        [
            "build-vocabulary",
            "--data-dir",
            str(development_artifact),
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 1
    assert "Integrity check failed" in result.output
    assert not output.exists()


def test_write_failure_leaves_no_partial_artifact(
    development_artifact: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_write = Path.write_bytes

    def fail_manifest(path: Path, content: bytes) -> int:
        if path.name == "manifest.json":
            raise OSError("synthetic write failure")
        return original_write(path, content)

    monkeypatch.setattr(Path, "write_bytes", fail_manifest)
    output = tmp_path / "vocabulary"
    with pytest.raises(VocabularyError, match="synthetic write failure"):
        build_vocabulary(development_artifact, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".vocabulary-*"))
