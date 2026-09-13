"""Exercise the CLI through source acquisition without network access."""

from pathlib import Path

from typer.testing import CliRunner

from filing_sentence_classifier.cli import app
from filing_sentence_classifier.data import source


def test_cli_download_and_corruption_error(tmp_path: Path, hub: list[str]) -> None:
    runner = CliRunner()
    args = ["download-data", "--output-dir", str(tmp_path)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert str(tmp_path / source.REVISION) in result.output

    (tmp_path / source.REVISION / "README.md").unlink()
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "Missing regular source file" in result.output
    assert "left unchanged" in result.output
