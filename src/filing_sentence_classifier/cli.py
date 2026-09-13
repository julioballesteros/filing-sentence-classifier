"""Command-line interface for the filing sentence classifier."""

from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import typer

from filing_sentence_classifier.data.fls import FLS_SOURCE, FLS_TRAIN_FILE, LABEL_NAMES

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
        from filing_sentence_classifier.data.source import (
            DownloadError,
            download_dataset,
        )
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
        snapshot = download_dataset(FLS_SOURCE, output_dir)
    except DownloadError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Verified FLS snapshot: {snapshot}")


@app.command("prepare-data")
def prepare_data(
    raw_dir: Annotated[
        Path,
        typer.Option(
            "--raw-dir", file_okay=False, help="Root of downloaded source snapshots."
        ),
    ] = Path("data/raw"),
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            file_okay=False,
            help="Root for versioned cleaned training artifacts.",
        ),
    ] = Path("data/interim"),
) -> None:
    """Clean the published FLS training split and record changes and exclusions."""
    try:
        from filing_sentence_classifier.data.prepare import (
            PreparationError,
            prepare_training_data,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "pyarrow":
            raise
        typer.echo(
            "Data dependencies are missing. Run this command with "
            "uv run --extra data, or install the package with its [data] extra.",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    try:
        artifact = prepare_training_data(
            FLS_SOURCE, FLS_TRAIN_FILE, LABEL_NAMES, raw_dir, output_dir
        )
    except PreparationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Verified cleaned training data: {artifact}")


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
