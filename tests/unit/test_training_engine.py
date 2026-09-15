"""Check explicit updates, loss aggregation, and failed-epoch behavior."""

import json
import math
from dataclasses import asdict

import pytest
import torch
from torch import nn
from torch.optim import SGD

from filing_sentence_classifier.data.collate import SentenceBatch, collate_sentences
from filing_sentence_classifier.training.config import TrainingConfig
from filing_sentence_classifier.training.engine import (
    EpochResult,
    TrainingError,
    train_epoch,
)
from filing_sentence_classifier.training.optimizers import create_optimizer


class ConstantClassifier(nn.Module):
    """Trainable logits with no random initialization or dependence on token IDs."""

    def __init__(self) -> None:
        super().__init__()
        self.scores = nn.Parameter(torch.tensor([2.0, 0.0, -1.0]))
        self.forward_states = []

    def forward(self, input_ids, attention_mask):
        self.forward_states.append(
            (self.training, torch.is_grad_enabled(), self.scores.grad is None)
        )
        return self.scores.unsqueeze(0).expand(input_ids.shape[0], -1)


def batch(targets: list[int]) -> SentenceBatch:
    return collate_sentences(
        [
            {
                "input_ids": torch.tensor([2], dtype=torch.long),
                "label": torch.tensor(target, dtype=torch.long),
                "sample_id": f"example-{index}",
                "original_length": 1,
                "truncated": False,
                "truncated_tokens": 0,
                "unknown_count": 0,
                "original_unknown_count": 0,
            }
            for index, target in enumerate(targets)
        ]
    )


def test_loss_is_weighted_by_examples_including_the_last_partial_batch() -> None:
    model = ConstantClassifier()
    # Zero learning rate holds logits fixed so the epoch loss has a known value.
    optimizer = SGD(model.parameters(), lr=0.0)
    result = train_epoch(model, iter([batch([0, 0, 0]), batch([2])]), optimizer)
    log_normalizer = math.log(math.exp(2) + 1 + math.exp(-1))
    expected = (3 * (log_normalizer - 2) + (log_normalizer + 1)) / 4
    assert result == EpochResult(pytest.approx(expected), 4, 2)
    assert result.mean_loss != pytest.approx(log_normalizer - 0.5)
    assert json.loads(json.dumps(asdict(result))) == asdict(result)


def test_training_enables_gradients_clears_each_batch_and_reuses_adamw_state() -> None:
    model = ConstantClassifier().eval()
    optimizer = create_optimizer(model, TrainingConfig(learning_rate=0.05))
    initial = model.scores.detach().clone()
    model.scores.grad = torch.full_like(model.scores, 999.0)
    batches = [batch([1, 1]), batch([1])]
    with torch.no_grad():
        first = train_epoch(model, batches, optimizer)
    second = train_epoch(model, batches, optimizer)
    assert model.training
    assert model.forward_states == [(True, True, True)] * 4
    assert not torch.equal(model.scores, initial)
    assert second.mean_loss < first.mean_loss
    assert (first.num_examples, first.num_batches) == (3, 2)
    assert optimizer.state[model.scores]["step"].item() == 4
    assert torch.isfinite(model.scores).all()


@pytest.mark.parametrize("failure", ["logits", "loss", "gradient", "initial_parameter"])
def test_numerical_errors_stop_before_the_optimizer_step(failure) -> None:
    model = ConstantClassifier()
    handles = []
    if failure == "logits":
        handles.append(
            model.register_forward_hook(
                lambda module, args, output: output * float("nan")
            )
        )
    elif failure == "loss":
        with torch.no_grad():
            model.scores.copy_(torch.tensor([3.4e38, -3.4e38, 0.0]))
    elif failure == "gradient":
        handles.append(
            model.scores.register_hook(lambda gradient: gradient * float("nan"))
        )
    else:
        with torch.no_grad():
            model.scores.fill_(float("inf"))
    optimizer = create_optimizer(model, TrainingConfig())
    initial = model.scores.detach().clone()
    expected = (
        "before training" if failure == "initial_parameter" else f"Batch 1:.*{failure}"
    )
    try:
        with pytest.raises(TrainingError, match=expected):
            train_epoch(model, [batch([1])], optimizer)
    finally:
        for handle in handles:
            handle.remove()
    assert torch.equal(model.scores, initial)
    assert not optimizer.state


def test_nonfinite_parameters_after_an_update_abort_the_epoch() -> None:
    model = ConstantClassifier()
    optimizer = create_optimizer(model, TrainingConfig())

    def corrupt_parameters(optimizer, args, kwargs):
        with torch.no_grad():
            model.scores.fill_(float("nan"))

    handle = optimizer.register_step_post_hook(corrupt_parameters)
    try:
        with pytest.raises(TrainingError, match="Batch 1:.*after optimizer step"):
            train_epoch(model, [batch([1]), batch([2])], optimizer)
    finally:
        handle.remove()
    assert optimizer.state[model.scores]["step"].item() == 1
    assert len(model.forward_states) == 1


def test_a_later_failed_batch_does_not_roll_back_completed_updates() -> None:
    model = ConstantClassifier()
    optimizer = create_optimizer(model, TrainingConfig())
    invalid = batch([0])
    invalid["labels"] = torch.tensor([-100])
    initial = model.scores.detach().clone()
    with pytest.raises(TrainingError, match="Batch 2:.*class indices"):
        train_epoch(model, [batch([1]), invalid], optimizer)
    assert optimizer.state[model.scores]["step"].item() == 1
    assert not torch.equal(model.scores, initial)


@pytest.mark.parametrize(
    "labels",
    [torch.tensor([-100]), torch.tensor([3]), torch.tensor([0.0]), torch.tensor([[0]])],
)
def test_invalid_labels_cannot_be_ignored_or_broadcast(labels) -> None:
    model = ConstantClassifier()
    optimizer = create_optimizer(model, TrainingConfig())
    invalid = batch([0])
    invalid["labels"] = labels
    with pytest.raises(TrainingError, match="Batch 1"):
        train_epoch(model, [invalid], optimizer)
    assert not optimizer.state


def test_empty_iterables_and_batches_are_rejected() -> None:
    model = ConstantClassifier()
    optimizer = create_optimizer(model, TrainingConfig())
    with pytest.raises(TrainingError, match="empty batch iterable"):
        train_epoch(model, [], optimizer)
    invalid = batch([0])
    invalid["labels"] = torch.empty(0, dtype=torch.long)
    with pytest.raises(TrainingError, match="Batch 1"):
        train_epoch(model, [invalid], optimizer)


def test_optimizer_must_belong_to_the_supplied_model() -> None:
    model = ConstantClassifier()
    optimizer = create_optimizer(ConstantClassifier(), TrainingConfig())
    with pytest.raises(TrainingError, match="own exactly"):
        train_epoch(model, [batch([0])], optimizer)


def test_trainable_parameters_cannot_be_disconnected() -> None:
    model = ConstantClassifier()
    model.unused = nn.Parameter(torch.ones(1))
    optimizer = create_optimizer(model, TrainingConfig())
    with pytest.raises(TrainingError, match="gradient for unused"):
        train_epoch(model, [batch([0])], optimizer)
    assert not optimizer.state


def test_detached_outputs_fail_with_a_training_error() -> None:
    model = ConstantClassifier()
    optimizer = create_optimizer(model, TrainingConfig())
    handle = model.register_forward_hook(lambda module, args, output: output.detach())
    try:
        with pytest.raises(TrainingError, match="disconnected from autograd"):
            train_epoch(model, [batch([0])], optimizer)
    finally:
        handle.remove()


def test_inference_context_and_mismatched_devices_fail_before_updates() -> None:
    model = ConstantClassifier()
    optimizer = create_optimizer(model, TrainingConfig())
    with torch.inference_mode(), pytest.raises(TrainingError, match="inference_mode"):
        train_epoch(model, [batch([0])], optimizer)
    with pytest.raises(TrainingError, match="training device"):
        train_epoch(model, [batch([0])], optimizer, device="meta")
    assert not optimizer.state
