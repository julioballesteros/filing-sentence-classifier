"""Runtime compatibility policy shared by bundle-loading backends."""

import re
import sys
from importlib.metadata import PackageNotFoundError, version

from filing_sentence_classifier.artifacts import (
    _DEPENDENCIES,
    BundleError,
    BundleManifest,
)


def check_environment(manifest: BundleManifest) -> None:
    """Require recorded runtime releases and the same Python major/minor.

    Allow Python patch differences and PyTorch local build suffixes such as +cpu.
    Only the model family's required distributions participate in this check.
    """
    recorded_python = manifest.environment["python"]
    match = re.fullmatch(r"(\d+)\.(\d+)\.\d+", recorded_python)
    if match is None or tuple(map(int, match.groups())) != sys.version_info[:2]:
        raise BundleError(
            f"Bundle requires Python major/minor matching {recorded_python}."
        )
    for name in sorted(_DEPENDENCIES[manifest.family] - {"python"}):
        recorded = manifest.environment[name]
        try:
            installed = version(name)
        except PackageNotFoundError as exc:
            raise BundleError(f"Required distribution is missing: {name}.") from exc
        expected, actual = recorded, installed
        if name == "torch":
            expected, actual = recorded.split("+", 1)[0], installed.split("+", 1)[0]
        if expected != actual:
            raise BundleError(
                f"Bundle requires {name} {recorded}; installed version is {installed}."
            )
