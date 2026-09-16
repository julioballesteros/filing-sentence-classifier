"""Use controlled epoch scores to verify checkpoint selection and patience."""

import json
from dataclasses import asdict, replace

import pytest
import torch

from filing_sentence_classifier.evaluation.metrics import classification_metrics
from filing_sentence_classifier.training import fit as fitting
from filing_sentence_classifier.training.checkpoints import load_checkpoint
from filing_sentence_classifier.training.config import TrainingConfig
from filing_sentence_classifier.training.engine import (
    EpochResult,
    TrainingError,
    ValidationResult,
)


@pytest.fixture
def controlled_epochs(monkeypatch):
    def setup(scores):
        model = torch.nn.Linear(2, 2)
        calls = {"optimizers": [], "validation_epochs": []}
        base_metrics = classification_metrics((0, 1), (0, 1), label_ids=(0, 1))

        def train_epoch(model, batches, optimizer, *, device):
            calls["optimizers"].append(optimizer)
            epoch = len(calls["optimizers"])
            model.train()
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(epoch)
            return EpochResult(1 / epoch, 2, 1)

        def validate_epoch(model, batches, *, label_ids, expected_sample_ids, device):
            epoch = int(model.weight[0, 0].item())
            calls["validation_epochs"].append(epoch)
            assert tuple(label_ids) == (0, 1)
            assert tuple(expected_sample_ids) == ("a", "b")
            model.eval()
            # Controlled scores isolate selection from stochastic learning behavior.
            return ValidationResult(
                1 / epoch,
                2,
                1,
                replace(base_metrics, macro_f1=scores[epoch - 1]),
                ("a", "b"),
                (0, 1),
                (0, 1),
            )

        monkeypatch.setattr(fitting, "train_epoch", train_epoch)
        monkeypatch.setattr(fitting, "validate_epoch", validate_epoch)
        return model, calls

    return setup


@pytest.mark.parametrize(
    "scores,patience,min_delta,max_epochs,best_epoch,stopped_early",
    [
        ([0.2, 0.5, 0.5, 0.4], 2, 0.0, 10, 2, True),
        ([0.5, 0.5000000001, 0.49, 0.49], 2, 0.0, 10, 2, True),
        ([0.5, 0.505, 0.51], 2, 0.1, 10, 3, True),
        ([0.5, 0.515, 0.53, 0.53, 0.53], 2, 0.02, 10, 3, True),
        ([0.5, 0.75, 0.75], 2, 0.25, 10, 2, True),
        ([0.0, 0.0], 1, 0.0, 10, 1, True),
        ([0.2, 0.3], 5, 0.0, 2, 2, False),
        ([0.2, 0.2], 1, 0.0, 2, 1, False),
    ],
)
def test_selection_is_independent_of_patience_and_restores_best_epoch(
    tmp_path,
    controlled_epochs,
    scores,
    patience,
    min_delta,
    max_epochs,
    best_epoch,
    stopped_early,
):
    model, calls = controlled_epochs(scores)
    config = TrainingConfig(
        max_epochs=max_epochs, patience=patience, min_delta=min_delta
    )
    path = tmp_path / "best.pt"
    result = fitting.fit(
        model,
        [],
        [],
        config=config,
        label_ids=iter((0, 1)),
        validation_sample_ids=iter(("a", "b")),
        checkpoint_path=path,
    )
    assert len(result.history) == len(calls["optimizers"]) == len(scores)
    assert len({id(optimizer) for optimizer in calls["optimizers"]}) == 1
    assert [row.epoch for row in result.history] == list(range(1, len(scores) + 1))
    assert [row.validation_metrics.macro_f1 for row in result.history] == scores
    assert result.best_epoch == best_epoch
    assert result.stopped_early is stopped_early
    assert result.best_validation.mean_loss == 1 / best_epoch
    assert result.best_validation.metrics.macro_f1 == scores[best_epoch - 1]
    assert calls["validation_epochs"] == [*range(1, len(scores) + 1), best_epoch]
    assert all(
        torch.equal(p, torch.full_like(p, best_epoch)) for p in model.parameters()
    )
    assert not model.training
    assert load_checkpoint(path, model, expected_config=config).epoch == best_epoch
    json.dumps([asdict(row) for row in result.history], allow_nan=False)


def test_bad_inputs_do_not_start_training_or_overwrite_existing_checkpoint(
    tmp_path, controlled_epochs
):
    model, calls = controlled_epochs([0.5])
    path = tmp_path / "best.pt"
    kwargs = dict(
        config=TrainingConfig(),
        label_ids=(0, 1),
        validation_sample_ids=("a", "b"),
        checkpoint_path=path,
    )
    for train_batches, val_batches in ((iter([]), []), ([], iter([]))):
        with pytest.raises(TrainingError, match="reusable"):
            fitting.fit(model, train_batches, val_batches, **kwargs)
    path.write_bytes(b"another run")
    with pytest.raises(TrainingError, match="new checkpoint path"):
        fitting.fit(model, [], [], **kwargs)
    assert path.read_bytes() == b"another run"
    assert not calls["optimizers"]


def test_later_training_failure_leaves_the_saved_best_checkpoint(
    tmp_path, controlled_epochs, monkeypatch
):
    model, calls = controlled_epochs([0.4, 0.5])
    train_epoch = fitting.train_epoch

    def fail_second_epoch(*args, **kwargs):
        if calls["optimizers"]:
            raise TrainingError("training failed")
        return train_epoch(*args, **kwargs)

    monkeypatch.setattr(fitting, "train_epoch", fail_second_epoch)
    path = tmp_path / "best.pt"
    with pytest.raises(TrainingError, match="training failed"):
        fitting.fit(
            model,
            [],
            [],
            config=TrainingConfig(),
            label_ids=(0, 1),
            validation_sample_ids=("a", "b"),
            checkpoint_path=path,
        )
    assert load_checkpoint(path, model).epoch == 1


def test_nonfinite_validation_score_cannot_be_selected(tmp_path, controlled_epochs):
    model, calls = controlled_epochs([float("nan")])
    path = tmp_path / "best.pt"
    with pytest.raises(TrainingError, match="macro-F1 must be finite"):
        fitting.fit(
            model,
            [],
            [],
            config=TrainingConfig(),
            label_ids=(0, 1),
            validation_sample_ids=("a", "b"),
            checkpoint_path=path,
        )
    assert not path.exists()
