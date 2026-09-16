"""Headless learning curves from completed training epochs."""

from collections.abc import Sequence
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from filing_sentence_classifier.training.fit import EpochMetrics


def save_learning_curves(
    history: Sequence[EpochMetrics], best_epoch: int, path: Path
) -> None:
    """Plot observed training loss, post-epoch validation loss, and macro-F1."""
    if not history or best_epoch not in {row.epoch for row in history}:
        raise ValueError("Learning curves require history containing the best epoch.")
    epochs = [row.epoch for row in history]
    figure = Figure(figsize=(10, 4), dpi=150, layout="constrained")
    FigureCanvasAgg(figure)
    loss, score = figure.subplots(1, 2)
    loss.plot(
        epochs,
        [row.training.mean_loss for row in history],
        label="Train (during updates)",
        color="#2563eb",
    )
    loss.plot(
        epochs,
        [row.validation_mean_loss for row in history],
        label="Validation (after epoch)",
        color="#c2410c",
    )
    loss.set(title="Cross-entropy", ylabel="Mean loss per sentence", ylim=(0, None))
    loss.legend(frameon=False, fontsize=8)
    score.plot(
        epochs,
        [row.validation_metrics.macro_f1 for row in history],
        color="#2563eb",
        label="Validation macro-F1",
    )
    score.set(title="Checkpoint selection", ylabel="Macro-F1", ylim=(0, 1))
    for axis in (loss, score):
        axis.axvline(
            best_epoch,
            color="#475569",
            linestyle="--",
            linewidth=1,
            label=f"Selected epoch: {best_epoch}",
        )
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    score.legend(frameon=False, fontsize=8)
    figure.savefig(path, format="png")
