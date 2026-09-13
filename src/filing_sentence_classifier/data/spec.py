"""Typed specifications for pinned dataset sources."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceFile:
    """An original file and its expected integrity metadata."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class DatasetSource:
    """A pinned Hugging Face dataset and the original files to acquire."""

    repo_id: str
    revision: str
    files: tuple[SourceFile, ...]
