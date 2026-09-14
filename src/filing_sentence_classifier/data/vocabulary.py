"""Build a vocabulary artifact from the verified development training partition."""

import hashlib
import json
import platform
import unicodedata
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from filing_sentence_classifier.data.loading import load_split
from filing_sentence_classifier.text.tokenization import tokenization_recipe, tokenize
from filing_sentence_classifier.text.vocabulary import Vocabulary, VocabularyError


def _json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _publish(staging: Path, destination: Path) -> Path:
    if destination.is_symlink():
        raise VocabularyError("The vocabulary destination must not be a symlink.")
    if destination.exists():
        if (
            not destination.is_dir()
            or {path.name for path in destination.iterdir()}
            != {path.name for path in staging.iterdir()}
            or any(
                (destination / path.name).is_symlink()
                or not (destination / path.name).is_file()
                or (destination / path.name).read_bytes() != path.read_bytes()
                for path in staging.iterdir()
            )
        ):
            raise VocabularyError(
                "Existing vocabulary differs; left unchanged. Choose another --output-dir."
            )
    else:
        staging.rename(destination)
    return destination


def build_vocabulary(
    data_dir: Path,
    output_dir: Path,
    *,
    min_frequency: int = 2,
    max_size: int | None = None,
    expected_manifest_sha256: str | None = None,
) -> Path:
    """Fit only train and publish vocabulary.json plus a provenance manifest.

    No validation/test text, labels for fitting, or source files are used.
    Repeating the same build verifies bytes and preserves existing artifacts.
    """
    try:
        train = load_split(
            data_dir, "train", expected_manifest_sha256=expected_manifest_sha256
        )
        documents = tuple(tokenize(text) for text in train.texts)
        vocabulary = Vocabulary.fit(
            documents, min_frequency=min_frequency, max_size=max_size
        )
        total_tokens = sum(len(document) for document in documents)
        unknown_tokens = total_tokens - sum(vocabulary.counts)
        content = _json(vocabulary.to_dict())
        destination = output_dir.expanduser().absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        package_root = Path(__file__).parents[1]
        manifest = {
            "schema_version": 1,
            "stage": "vocabulary_build",
            "fit_split": "train",
            "tokenization": tokenization_recipe(),
            "vocabulary": vocabulary.recipe(),
            "data": {
                "manifest_sha256": train.manifest_sha256,
                "train_sha256": train.records_sha256,
                "train_rows": len(train.records),
            },
            "statistics": {
                "train_tokens": total_tokens,
                "train_token_types": len({token for row in documents for token in row}),
                "vocabulary_size": len(vocabulary),
                "retained_token_types": len(vocabulary) - 2,
                "train_unknown_tokens": unknown_tokens,
                "train_unknown_rate": unknown_tokens / total_tokens,
            },
            "environment": {
                "python": platform.python_version(),
                "unicode": unicodedata.unidata_version,
                "filing-sentence-classifier": version("filing-sentence-classifier"),
            },
            "code_sha256": {
                name: hashlib.sha256((package_root / name).read_bytes()).hexdigest()
                for name in (
                    "data/loading.py",
                    "data/vocabulary.py",
                    "text/tokenization.py",
                    "text/vocabulary.py",
                )
            },
            "files": {
                "vocabulary.json": {
                    "size_bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            },
        }
        with TemporaryDirectory(prefix=".vocabulary-", dir=destination.parent) as temp:
            staging = Path(temp) / "artifact"
            staging.mkdir()
            (staging / "vocabulary.json").write_bytes(content)
            (staging / "manifest.json").write_bytes(_json(manifest))
            return _publish(staging, destination)
    except (OSError, ValueError) as exc:
        raise VocabularyError(f"Could not build vocabulary: {exc}") from exc
