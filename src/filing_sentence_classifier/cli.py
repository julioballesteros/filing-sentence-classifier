"""Command-line interface for the filing sentence classifier."""

from importlib.metadata import version
from typing import Annotated

import typer

app = typer.Typer()


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
