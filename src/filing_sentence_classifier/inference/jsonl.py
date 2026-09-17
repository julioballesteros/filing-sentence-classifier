"""Bounded JSONL input and prediction output, independent of the CLI framework."""

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, BinaryIO

from filing_sentence_classifier.inference.contracts import PredictionContractError
from filing_sentence_classifier.inference.predictor import Predictor

MAX_JSONL_LINE_BYTES = 1_048_576
MAX_SAMPLE_ID_CHARACTERS = 256


class PredictionIOError(ValueError):
    """Prediction input or output cannot be processed."""


@dataclass(frozen=True)
class _Row:
    line: int
    text: str
    sample_id: str | None


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key.")
        result[key] = value
    return result


def _read_rows(source: BinaryIO) -> Iterator[_Row]:
    line_number = 0
    while raw := source.readline(MAX_JSONL_LINE_BYTES + 1):
        line_number += 1
        if len(raw) > MAX_JSONL_LINE_BYTES:
            raise PredictionIOError(
                f"Line {line_number}: exceeds the 1 MiB JSONL line limit."
            )
        try:
            row = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
        except (ValueError, RecursionError) as exc:
            raise PredictionIOError(
                f"Line {line_number}: expected valid UTF-8 JSON with unique keys."
            ) from exc
        if (
            not isinstance(row, dict)
            or set(row) not in ({"text"}, {"text", "sample_id"})
            or not isinstance(row["text"], str)
        ):
            raise PredictionIOError(
                f"Line {line_number}: expected text and an optional sample_id."
            )
        sample_id = row.get("sample_id")
        if "sample_id" in row:
            if (
                not isinstance(sample_id, str)
                or not sample_id.strip()
                or len(sample_id) > MAX_SAMPLE_ID_CHARACTERS
            ):
                raise PredictionIOError(
                    f"Line {line_number}: sample_id must be a nonblank string "
                    f"of at most {MAX_SAMPLE_ID_CHARACTERS} characters."
                )
            try:
                sample_id.encode("utf-8")
            except UnicodeError as exc:
                raise PredictionIOError(
                    f"Line {line_number}: sample_id contains unsupported Unicode."
                ) from exc
        yield _Row(line_number, row["text"], sample_id)


def predict_jsonl(
    predictor: Predictor,
    source: BinaryIO,
    destination: IO[str],
    *,
    batch_size: int = 32,
) -> int:
    """Stream ordered predictions using one loaded model and bounded requests.

    Each row contains text and optionally sample_id, which is echoed unchanged.
    Empty input succeeds with no output; blank lines are invalid. A late error
    can leave earlier batches in destination. Use new_prediction_file for atomic
    file publication. Diagnostics identify input lines without echoing text.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise PredictionIOError("batch_size must be a positive integer.")
    request_size = min(batch_size, predictor.manifest.limits.max_batch_size)
    pending: list[_Row] = []
    count = 0

    def write_batch() -> None:
        try:
            results = predictor.predict(
                [row.text for row in pending], batch_size=batch_size
            )
        except PredictionContractError as exc:
            raise PredictionIOError(
                f"Lines {pending[0].line}-{pending[-1].line}: {exc}"
            ) from exc
        for row, prediction in zip(pending, results, strict=True):
            output = prediction.to_dict()
            if row.sample_id is not None:
                output["sample_id"] = row.sample_id
            destination.write(json.dumps(output, allow_nan=False) + "\n")

    for row in _read_rows(source):
        pending.append(row)
        if len(pending) == request_size:
            write_batch()
            count += len(pending)
            pending.clear()
    if pending:
        write_batch()
        count += len(pending)
    return count


@contextmanager
def new_prediction_file(path: Path) -> Iterator[IO[str]]:
    """Publish a complete UTF-8 file without replacing an existing destination."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Output already exists: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            yield stream.file
            stream.flush()
            os.fsync(stream.fileno())
        # Linking a sibling temporary file publishes atomically and fails if a
        # concurrent writer has created the destination in the meantime.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
