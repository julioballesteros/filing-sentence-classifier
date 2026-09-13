"""Command-line interface for the filing sentence classifier."""

from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()


@app.command("download-data")
def download_data(
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            file_okay=False,
            help="Root directory for revision-specific source snapshots.",
        ),
    ] = Path("data/raw"),
) -> None:
    """Download the pinned FLS source files, or verify an existing local snapshot."""
    # Keep data dependencies optional for other CLI commands.
    try:
        from filing_sentence_classifier.data.source import DownloadError, download_fls
    except ModuleNotFoundError as exc:
        if exc.name != "huggingface_hub":
            raise
        typer.echo(
            "Data dependencies are missing. Run this command with "
            "uv run --extra data, or install the package with its [data] extra.",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    try:
        snapshot = download_fls(output_dir)
    except DownloadError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Verified FLS snapshot: {snapshot}")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    show_version: Annotated[
        bool,
        typer.Option("--version", help="Show the installed package version and exit."),
    ] = False,
) -> None:
    """Work with forward-looking statement classifiers for financial filing sentences."""
    if show_version:
        typer.echo(version("filing-sentence-classifier"))
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
