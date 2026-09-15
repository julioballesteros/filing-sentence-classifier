"""Check whole-partition metrics, identity coverage, and evaluation state."""

import json
import math
from dataclasses import asdict

import pytest
import torch
from torch import nn

from filing_sentence_classifier.data.collate import collate_sentences
from filing_sentence_classifier.evaluation import metrics as shared_metrics
from filing_sentence_classifier.training.engine import TrainingError, validate_epoch


class LookupClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scores = nn.Parameter(
            torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, -1.0], [0.0, 2.0, -1.0]]
            )
        )

    def forward(self, input_ids, attention_mask):
        return self.scores[input_ids[:, 0]]


def batch(rows):
    return collate_sentences(
        [
            {
                "input_ids": torch.tensor([token], dtype=torch.long),
                "label": torch.tensor(target, dtype=torch.long),
                "sample_id": sample_id,
                "original_length": 1,
                "truncated": False,
                "truncated_tokens": 0,
                "unknown_count": int(token == 1),
                "original_unknown_count": int(token == 1),
            }
            for sample_id, target, token in rows
        ]
    )


ROWS = [("a", 0, 2), ("b", 0, 3), ("c", 1, 3), ("d", 1, 3)]
IDS = ("a", "b", "c", "d")
CLASSES = (0, 1, 2)


def test_metrics_are_computed_once_over_all_samples_with_aligned_predictions(
    monkeypatch,
) -> None:
    calls = []
    original = shared_metrics.classification_metrics

    def counted_metrics(targets, predictions, *, label_ids):
        calls.append((tuple(targets), tuple(predictions)))
        return original(targets, predictions, label_ids=label_ids)

    monkeypatch.setattr(shared_metrics, "classification_metrics", counted_metrics)
    result = validate_epoch(
        LookupClassifier(),
        iter([batch([ROWS[2], ROWS[0], ROWS[3]]), batch([ROWS[1]])]),
        label_ids=CLASSES,
        expected_sample_ids=IDS,
    )
    assert (result.num_examples, result.num_batches) == (4, 2)
    assert result.sample_ids == ("c", "a", "d", "b")
    assert result.targets == (1, 0, 1, 0)
    assert result.predicted_labels == (1, 0, 1, 1)
    assert calls == [(result.targets, result.predicted_labels)]
    log_normalizer = math.log(math.exp(2) + 1 + math.exp(-1))
    assert result.mean_loss == pytest.approx(log_normalizer - 1.5)
    assert result.metrics.accuracy == 0.75
    assert result.metrics.macro_f1 == pytest.approx(22 / 45)
    # Batch F1 averaging would instead give 1/3 (unweighted) or 1/2 (weighted).
    assert result.metrics.confusion_matrix == ((1, 1, 0), (0, 2, 0), (0, 0, 0))
    assert result.metrics.label_ids == CLASSES
    assert result.metrics.sample_count == 4
    assert result.metrics.per_class[2].support == result.metrics.per_class[2].f1 == 0
    assert json.loads(json.dumps(asdict(result)))["sample_ids"] == list(
        result.sample_ids
    )


def test_rebatching_and_reordering_preserve_metrics_and_predictions_by_id() -> None:
    model = LookupClassifier()
    first = validate_epoch(
        model, [batch(ROWS)], label_ids=CLASSES, expected_sample_ids=IDS
    )
    second = validate_epoch(
        model,
        [batch([row]) for row in reversed(ROWS)],
        label_ids=CLASSES,
        expected_sample_ids=IDS,
    )
    assert first.metrics == second.metrics
    assert first.mean_loss == pytest.approx(second.mean_loss)
    assert dict(zip(first.sample_ids, first.predicted_labels, strict=True)) == dict(
        zip(second.sample_ids, second.predicted_labels, strict=True)
    )


def test_eval_disables_dropout_and_preserves_buffers_gradients_and_rng() -> None:
    class StatefulClassifier(LookupClassifier):
        def __init__(self):
            super().__init__()
            self.batch_norm = nn.BatchNorm1d(3)
            self.dropout = nn.Dropout(0.8)
            self.modes = []

        def forward(self, input_ids, attention_mask):
            self.modes.append(
                (
                    self.training,
                    torch.is_grad_enabled(),
                    torch.is_inference_mode_enabled(),
                )
            )
            return self.dropout(
                self.batch_norm(super().forward(input_ids, attention_mask))
            )

    model = StatefulClassifier().train()
    model.scores.grad = torch.full_like(model.scores, 0.25)
    gradient = model.scores.grad
    state = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    outer_grad_mode = torch.is_grad_enabled()
    batches = [batch([row]) for row in ROWS]
    first = validate_epoch(model, batches, label_ids=CLASSES, expected_sample_ids=IDS)
    second = validate_epoch(model, batches, label_ids=CLASSES, expected_sample_ids=IDS)
    assert first == second
    assert model.modes == [(False, False, True)] * 8
    assert not any(module.training for module in model.modules())
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name])
    assert model.scores.grad is gradient
    assert torch.all(gradient == 0.25)
    assert model.batch_norm.weight.grad is model.batch_norm.bias.grad is None
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.is_grad_enabled() == outer_grad_mode


def test_argmax_ties_select_the_lowest_class_id() -> None:
    result = validate_epoch(
        LookupClassifier(),
        [batch([("tie", 2, 1)])],
        label_ids=CLASSES,
        expected_sample_ids=("tie",),
    )
    assert result.predicted_labels == (0,)
    assert result.mean_loss == pytest.approx(math.log(3))


@pytest.mark.parametrize("classes", [(), (0,), (0, 2), (1, 0, 2), (0, True, 2)])
def test_class_ids_must_match_zero_based_logit_order(classes) -> None:
    with pytest.raises(TrainingError, match="class IDs"):
        validate_epoch(
            LookupClassifier(),
            [batch(ROWS)],
            label_ids=classes,
            expected_sample_ids=IDS,
        )


@pytest.mark.parametrize("expected", [(), ("a", "a"), ("",), (3,)])
def test_expected_ids_are_nonempty_unique_strings(expected) -> None:
    with pytest.raises(TrainingError, match="Expected validation sample IDs"):
        validate_epoch(
            LookupClassifier(),
            [batch(ROWS)],
            label_ids=CLASSES,
            expected_sample_ids=expected,
        )


@pytest.mark.parametrize(
    "rows,expected,message",
    [
        ([ROWS[0], ROWS[0]], IDS, "duplicate"),
        ([ROWS[0]], IDS, "missing 3"),
        (ROWS, ("a", "b", "c"), "unexpected"),
    ],
)
def test_incomplete_or_repeated_validation_samples_are_rejected(
    rows, expected, message
) -> None:
    with pytest.raises(TrainingError, match=message):
        validate_epoch(
            LookupClassifier(),
            [batch([row]) for row in rows],
            label_ids=CLASSES,
            expected_sample_ids=expected,
        )


def test_duplicate_ids_inside_a_batch_and_empty_iterables_are_rejected() -> None:
    for batches, message in (
        ([batch([ROWS[0], ROWS[0]])], "duplicate"),
        ([], "empty batch iterable"),
    ):
        with pytest.raises(TrainingError, match=message):
            validate_epoch(
                LookupClassifier(), batches, label_ids=CLASSES, expected_sample_ids=IDS
            )


@pytest.mark.parametrize("sample_id", ["", 3])
def test_invalid_batch_ids_are_rejected(sample_id) -> None:
    invalid = batch(ROWS)
    invalid["sample_ids"][0] = sample_id
    with pytest.raises(TrainingError, match="invalid validation sample ID"):
        validate_epoch(
            LookupClassifier(), [invalid], label_ids=CLASSES, expected_sample_ids=IDS
        )


@pytest.mark.parametrize(
    "labels", [torch.tensor([-100]), torch.tensor([3]), torch.tensor([0.0])]
)
def test_invalid_targets_cannot_be_ignored(labels) -> None:
    invalid = batch([ROWS[0]])
    invalid["labels"] = labels
    with pytest.raises(TrainingError, match="Batch 1"):
        validate_epoch(
            LookupClassifier(), [invalid], label_ids=CLASSES, expected_sample_ids=("a",)
        )


@pytest.mark.parametrize("failure", ["logits", "loss", "columns", "shape", "dtype"])
def test_invalid_outputs_abort_validation_and_restore_grad_mode(failure) -> None:
    model = LookupClassifier()
    if failure == "loss":
        with torch.no_grad():
            model.scores[2].copy_(torch.tensor([-3.4e38, 3.4e38, 0.0]))

        def transform(output):
            return output
    else:
        transform = {
            "logits": lambda output: output * float("nan"),
            "columns": lambda output: output[:, :2],
            "shape": lambda output: output[0],
            "dtype": lambda output: output.long(),
        }[failure]
    handle = model.register_forward_hook(lambda module, args, output: transform(output))
    before = torch.is_grad_enabled()
    try:
        with pytest.raises(TrainingError, match="Batch 1"):
            validate_epoch(
                model, [batch([ROWS[0]])], label_ids=CLASSES, expected_sample_ids=("a",)
            )
    finally:
        handle.remove()
    assert torch.is_grad_enabled() == before
    assert model.scores.grad is None


def test_invalid_model_state_and_device_are_rejected_before_forward() -> None:
    model = LookupClassifier()
    model.register_buffer("invalid", torch.tensor(float("nan")))
    with pytest.raises(TrainingError, match="before validation"):
        validate_epoch(model, [batch(ROWS)], label_ids=CLASSES, expected_sample_ids=IDS)
    with pytest.raises(TrainingError, match="validation device"):
        validate_epoch(
            LookupClassifier().to("meta"),
            [batch(ROWS)],
            label_ids=CLASSES,
            expected_sample_ids=IDS,
        )
