"""Command-line interface for the filing sentence classifier."""

import json
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import typer

from filing_sentence_classifier.data.fls import (
    FLS_SOURCE,
    FLS_TEST_FILE,
    FLS_TRAIN_FILE,
    LABEL_NAMES,
)
from filing_sentence_classifier.data.loading import DataLoadError, SplitName

app = typer.Typer()
baseline_app = typer.Typer(
    help="Fit reference classifiers and evaluate saved validation."
)
app.add_typer(baseline_app, name="baseline")


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


@app.command("split-data")
def split_data(
    prepared_dir: Annotated[
        Path | None,
        typer.Option(
            "--prepared-dir",
            file_okay=False,
            help="Cleaned artifact directory; defaults to the pinned clean-v1 artifact.",
        ),
    ] = None,
    raw_dir: Annotated[
        Path,
        typer.Option(
            "--raw-dir",
            file_okay=False,
            help="Source snapshot root for the text-only test overlap check.",
        ),
    ] = Path("data/raw"),
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            file_okay=False,
            help="Root for versioned train/validation artifacts.",
        ),
    ] = Path("data/processed"),
) -> None:
    """Freeze grouped train/validation splits after excluding test text overlaps."""
    try:
        from filing_sentence_classifier.data.cleaning import CLEANING_VERSION
        from filing_sentence_classifier.data.partition import create_development_split
        from filing_sentence_classifier.data.split import SplitError
    except ModuleNotFoundError as exc:
        if exc.name not in {"pyarrow", "sklearn", "numpy", "scipy"}:
            raise
        typer.echo(
            "Data dependencies are missing. Run this command with "
            "uv run --extra data, or install the package with its [data] extra.",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    if prepared_dir is None:
        prepared_dir = (
            Path("data/interim") / FLS_SOURCE.revision / f"clean-v{CLEANING_VERSION}"
        )
    try:
        artifact = create_development_split(
            FLS_SOURCE,
            FLS_TRAIN_FILE,
            FLS_TEST_FILE,
            LABEL_NAMES,
            prepared_dir,
            raw_dir,
            output_dir,
        )
    except SplitError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Verified development splits: {artifact}")


@app.command("build-vocabulary")
def vocabulary_build(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir", file_okay=False, help="Frozen development split directory."
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            file_okay=False,
            help="Directory for the vocabulary artifact.",
        ),
    ] = Path("artifacts/preprocessing/vocabulary-v1"),
    min_frequency: Annotated[
        int,
        typer.Option(
            "--min-frequency", min=1, help="Minimum training token occurrences."
        ),
    ] = 2,
    max_size: Annotated[
        int | None,
        typer.Option(
            "--max-size", min=3, help="Maximum vocabulary size including PAD and UNK."
        ),
    ] = None,
    manifest_sha256: Annotated[
        str | None,
        typer.Option(
            "--manifest-sha256", help="Optional expected dataset manifest checksum."
        ),
    ] = None,
) -> None:
    """Fit a deterministic vocabulary on train and save its counts and provenance."""
    from filing_sentence_classifier.data.vocabulary import build_vocabulary
    from filing_sentence_classifier.text.vocabulary import VocabularyError

    try:
        artifact = build_vocabulary(
            data_dir,
            output_dir,
            min_frequency=min_frequency,
            max_size=max_size,
            expected_manifest_sha256=manifest_sha256,
        )
    except (DataLoadError, VocabularyError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Verified training vocabulary: {artifact}")


@app.command("train")
def train_model(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir", file_okay=False, help="Frozen development split directory."
        ),
    ],
    vocabulary_dir: Annotated[
        Path,
        typer.Option(
            "--vocabulary-dir",
            file_okay=False,
            help="Vocabulary artifact fitted on the same train split.",
        ),
    ],
    config: Annotated[
        Path,
        typer.Option(
            "--config", dir_okay=False, help="Neural training TOML configuration."
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            file_okay=False,
            help="New directory for this run; existing paths are never overwritten.",
        ),
    ],
    max_length: Annotated[
        int,
        typer.Option(
            "--max-length", min=1, help="Keep at most this many tokens per sentence."
        ),
    ] = 128,
    manifest_sha256: Annotated[
        str | None,
        typer.Option(
            "--manifest-sha256", help="Optional expected dataset manifest checksum."
        ),
    ] = None,
) -> None:
    """Train MeanPoolMLP, select with validation, and save a complete local run."""
    try:
        from filing_sentence_classifier.training.run import (
            TrainingRunError,
            run_training,
        )
    except ModuleNotFoundError as exc:
        if exc.name not in {"torch", "sklearn", "numpy", "scipy", "matplotlib"}:
            raise
        typer.echo(
            "Training dependencies are missing. Install the package with its [data,train] extras.",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    from filing_sentence_classifier.training.fit import EpochMetrics

    def progress(row: EpochMetrics) -> None:
        typer.echo(
            f"Epoch {row.epoch:02d} | train loss {row.training.mean_loss:.4f} | val loss {row.validation_mean_loss:.4f} | val macro-F1 {row.validation_metrics.macro_f1:.4f}"
        )

    try:
        artifact = run_training(
            data_dir,
            output_dir,
            config,
            vocabulary_dir,
            max_length=max_length,
            expected_manifest_sha256=manifest_sha256,
            on_epoch=progress,
        )
    except TrainingRunError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Completed training run: {artifact}")


@app.command("evaluate")
def evaluate(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            file_okay=False,
            help="Directory containing a frozen development split manifest.",
        ),
    ],
    predictions: Annotated[
        Path,
        typer.Option(
            "--predictions",
            dir_okay=False,
            help="JSONL rows with sample_id and predicted_label.",
        ),
    ],
    split: Annotated[
        SplitName,
        typer.Option("--split", help="Saved development partition to evaluate."),
    ] = SplitName.VAL,
    manifest_sha256: Annotated[
        str | None,
        typer.Option(
            "--manifest-sha256", help="Optional expected dataset manifest checksum."
        ),
    ] = None,
) -> None:
    """Evaluate saved predictions by sample ID and print a JSON metrics report."""
    try:
        from filing_sentence_classifier.evaluation.evaluate import (
            evaluate_prediction_file,
        )
        from filing_sentence_classifier.evaluation.metrics import EvaluationError
    except ModuleNotFoundError as exc:
        if exc.name not in {"sklearn", "numpy", "scipy"}:
            raise
        typer.echo(
            "Evaluation dependencies are missing. Install the package with its [data] extra.",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    try:
        report = evaluate_prediction_file(
            data_dir, split, predictions, expected_manifest_sha256=manifest_sha256
        )
    except (DataLoadError, EvaluationError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


@baseline_app.command("majority")
def majority_baseline(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir", file_okay=False, help="Frozen development split directory."
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir", file_okay=False, help="Directory for this baseline run."
        ),
    ] = Path("artifacts/runs/majority-v1"),
    manifest_sha256: Annotated[
        str | None,
        typer.Option(
            "--manifest-sha256", help="Optional expected dataset manifest checksum."
        ),
    ] = None,
) -> None:
    """Fit the training majority and save its validation predictions and metrics."""
    try:
        from filing_sentence_classifier.baselines.majority import BaselineError
        from filing_sentence_classifier.baselines.run import run_majority_baseline
    except ModuleNotFoundError as exc:
        if exc.name not in {"sklearn", "numpy", "scipy"}:
            raise
        typer.echo(
            "Evaluation dependencies are missing. Install the package with its [data] extra.",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    try:
        artifact = run_majority_baseline(
            data_dir, output_dir, expected_manifest_sha256=manifest_sha256
        )
    except (DataLoadError, BaselineError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Verified majority baseline run: {artifact}")


@baseline_app.command("tfidf")
def tfidf_baseline(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir", file_okay=False, help="Frozen development split directory."
        ),
    ],
    config: Annotated[
        Path,
        typer.Option(
            "--config", dir_okay=False, help="TF-IDF/logistic regression TOML recipe."
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir", file_okay=False, help="Directory for this baseline run."
        ),
    ] = Path("artifacts/runs/tfidf-logreg-v1"),
    manifest_sha256: Annotated[
        str | None,
        typer.Option(
            "--manifest-sha256", help="Optional expected dataset manifest checksum."
        ),
    ] = None,
) -> None:
    """Fit TF-IDF and logistic regression on train, then evaluate saved validation."""
    try:
        from filing_sentence_classifier.baselines.majority import BaselineError
        from filing_sentence_classifier.baselines.tfidf_run import run_tfidf_baseline
    except ModuleNotFoundError as exc:
        if exc.name not in {"sklearn", "numpy", "scipy", "joblib", "threadpoolctl"}:
            raise
        typer.echo(
            "Baseline dependencies are missing. Install the package with its [data] extra.",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    try:
        artifact = run_tfidf_baseline(
            data_dir, output_dir, config, expected_manifest_sha256=manifest_sha256
        )
    except (DataLoadError, BaselineError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Verified TF-IDF baseline run: {artifact}")


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
