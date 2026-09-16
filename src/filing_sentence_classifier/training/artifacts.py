"""Local run artifact writes and snapshots of the code actually imported."""

import hashlib
import json
import os
import platform
import subprocess
import tempfile
import tomllib
from importlib.metadata import version
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Atomically replace one JSON file owned by the current run."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def file_inventory(directory: Path) -> dict[str, dict[str, str | int]]:
    """Hash regular run files, excluding the manifest that contains the inventory."""
    return {
        path.relative_to(directory).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path.read_bytes()),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != directory / "manifest.json"
    }


def capture_source(output_dir: Path, project_dir: Path) -> dict[str, object]:
    """Snapshot installed Python sources; optionally attach matching checkout state.

    Sources also cover untracked files and non-editable/wheel installations.
    Git describes only the supplied project directory; it is never assumed to be
    the imported code. The snapshot remains authoritative when they differ.
    """
    snapshot = output_dir / "source" / "src" / "filing_sentence_classifier"
    code_hashes = {}

    def copy_package(node: Traversable, relative: Path) -> None:
        for item in sorted(node.iterdir(), key=lambda entry: entry.name):
            if item.name == "__pycache__":
                continue
            child = relative / item.name
            if item.is_dir():
                copy_package(item, child)
            elif item.name.endswith(".py"):
                content = item.read_bytes()
                target = snapshot / child
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                code_hashes[child.as_posix()] = sha256(content)

    copy_package(files("filing_sentence_classifier"), Path())
    provenance: dict[str, object] = {"code_sha256": code_hashes, "git": None}
    project_file = project_dir / "pyproject.toml"
    if not project_file.is_file():
        return provenance
    project_bytes = project_file.read_bytes()
    if (
        tomllib.loads(project_bytes.decode("utf-8")).get("project", {}).get("name")
        != "filing-sentence-classifier"
    ):
        return provenance
    (output_dir / "source" / "pyproject.toml").write_bytes(project_bytes)
    lock = project_dir / "uv.lock"
    if lock.is_file():
        (output_dir / "source" / "uv.lock").write_bytes(lock.read_bytes())
    python_version = project_dir / ".python-version"
    if python_version.is_file():
        (output_dir / "source" / ".python-version").write_bytes(
            python_version.read_bytes()
        )
    matches_checkout = all(
        (project_dir / "src" / "filing_sentence_classifier" / name).is_file()
        and sha256(
            (project_dir / "src" / "filing_sentence_classifier" / name).read_bytes()
        )
        == expected
        for name, expected in code_hashes.items()
    )
    scope = ["src", "pyproject.toml", "uv.lock", ".python-version"]

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(project_dir), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        changes = git("status", "--porcelain", "--untracked-files=all", "--", *scope)
        diff = git("diff", "--no-ext-diff", "--no-textconv", "HEAD", "--", *scope)
    except (OSError, subprocess.SubprocessError):
        # Git is optional for an installed package; actual code is already saved.
        return provenance
    if diff:
        (output_dir / "source" / "working-tree.patch").write_text(
            diff + "\n", encoding="utf-8"
        )
    provenance["git"] = {
        "commit": commit,
        "scope": scope,
        "changes": changes,
        "matches_imported_code": matches_checkout,
        "reproducible_from_commit": matches_checkout and not changes,
    }
    return provenance


def environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "packages": {
            name: version(name)
            for name in (
                "filing-sentence-classifier",
                "torch",
                "numpy",
                "scipy",
                "scikit-learn",
                "matplotlib",
            )
        },
    }
