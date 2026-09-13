"""Exercise the CLI through source acquisition without network access."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from filing_sentence_classifier import cli
from filing_sentence_classifier.data.spec import DatasetSource


def test_cli_download_and_corruption_error(
    tmp_path: Path,
    dataset_source: DatasetSource,
    hub: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Substitute the selected dataset only at the CLI composition boundary.
    monkeypatch.setattr(cli, "FLS_SOURCE", dataset_source)
    runner = CliRunner()
    args = ["download-data", "--output-dir", str(tmp_path)]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert str(tmp_path / dataset_source.revision) in result.output

    (tmp_path / dataset_source.revision / "README.md").unlink()
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 1
    assert "Missing regular source file" in result.output
    assert "left unchanged" in result.output
