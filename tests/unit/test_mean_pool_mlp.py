"""Verify sentence representations, logits, gradients, and model contracts."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from filing_sentence_classifier.models.mean_pool_mlp import MeanPoolMLP


@pytest.fixture(autouse=True)
def isolated_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        yield


def test_known_weights_produce_masked_means_relu_and_raw_logits() -> None:
    model = MeanPoolMLP(6, embedding_dim=2, hidden_dim=2).eval()
    with torch.no_grad():
        # A nonzero PAD vector exposes missing masking in the numerator.
        model.embedding.weight[:4].copy_(
            torch.tensor([[100.0, 100.0], [2.0, 4.0], [4.0, 2.0], [-2.0, 6.0]])
        )
        model.classifier[0].weight.copy_(torch.eye(2))
        model.classifier[0].bias.zero_()
        model.classifier[3].weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
        )
        model.classifier[3].bias.copy_(torch.tensor([-10.0, 0.0, 1.0]))
    ids = torch.tensor([[1, 2, 0], [3, 0, 0]])
    logits = model(ids, ids != 0)
    torch.testing.assert_close(
        logits, torch.tensor([[-7.0, 3.0, 1.0], [-10.0, 6.0, -5.0]])
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_padding_and_other_batch_members_do_not_change_eval_logits(dtype) -> None:
    model = MeanPoolMLP(9, embedding_dim=8, hidden_dim=4, dropout=0.5).to(dtype).eval()
    short = torch.tensor([[1, 2]])
    mixed = torch.tensor([[8, 7, 6, 5], [1, 2, 0, 0]])
    padded = F.pad(mixed, (0, 17), value=0)
    with torch.inference_mode():
        alone_logits = model(short, short != 0)
        mixed_logits = model(mixed, mixed != 0)
        padded_logits = model(padded, padded != 0)
    assert alone_logits.shape == (1, 3)
    assert mixed_logits.dtype == dtype
    torch.testing.assert_close(alone_logits[0], mixed_logits[1])
    torch.testing.assert_close(mixed_logits, padded_logits)


def test_single_unknown_token_and_configurable_dimensions() -> None:
    model = MeanPoolMLP(2, embedding_dim=7, hidden_dim=5, num_classes=4)
    logits = model(torch.tensor([[1]]), torch.tensor([[True]]))
    assert logits.shape == (1, 4)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    assert model.embedding.weight.shape == (2, 7)
    assert model.classifier[0].out_features == 5


def test_gradients_reach_embeddings_and_classifier_but_not_padding() -> None:
    model = MeanPoolMLP(8, embedding_dim=4, hidden_dim=4)
    with torch.no_grad():
        model.embedding.weight[1:].fill_(0.5)
        model.classifier[0].weight.copy_(torch.eye(4))
        model.classifier[0].bias.fill_(0.1)
        model.classifier[3].weight.copy_(
            torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 1.0]]
            )
        )
        model.classifier[3].bias.zero_()
    ids = torch.tensor([[1, 2, 0], [3, 0, 0]])
    mask = ids != 0
    original_ids, original_mask = ids.clone(), mask.clone()
    loss = nn.CrossEntropyLoss()(model(ids, mask), torch.tensor([0, 1]))
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name
    gradients = model.embedding.weight.grad
    assert gradients is not None
    assert torch.count_nonzero(gradients[0]) == 0
    assert torch.count_nonzero(model.embedding.weight[0]) == 0
    assert torch.all(gradients[1:4].abs().sum(1) > 0)  # Includes trainable UNK.
    assert torch.count_nonzero(gradients[4:]) == 0
    assert torch.equal(ids, original_ids)
    assert torch.equal(mask, original_mask)


def test_dropout_follows_caller_mode_and_eval_preserves_rng() -> None:
    model = MeanPoolMLP(3, embedding_dim=4, hidden_dim=32, dropout=0.5)
    with torch.no_grad():
        model.classifier[0].weight.zero_()
        model.classifier[0].bias.fill_(1.0)
        model.classifier[3].weight.fill_(1.0)
    ids = torch.ones((16, 1), dtype=torch.long)
    mask = ids != 0
    first, second = model(ids, mask), model(ids, mask)
    assert model.training
    assert not torch.equal(first, second)
    model.eval()
    before = torch.get_rng_state().clone()
    first, second = model(ids, mask), model(ids, mask)
    assert not model.training
    assert first.requires_grad  # eval() alone does not disable autograd.
    assert torch.equal(first, second)
    assert torch.equal(before, torch.get_rng_state())


def test_standard_state_dict_restores_eval_logits() -> None:
    options = {"vocab_size": 7, "embedding_dim": 8, "hidden_dim": 4, "dropout": 0.3}
    original = MeanPoolMLP(**options).eval()
    restored = MeanPoolMLP(**options).eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    with torch.inference_mode():
        torch.testing.assert_close(
            original(ids, ids != 0), restored(ids, ids != 0), rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "options",
    [
        {"vocab_size": 1},
        {"vocab_size": True},
        {"embedding_dim": 0},
        {"embedding_dim": 2.5},
        {"hidden_dim": -1},
        {"hidden_dim": False},
        {"num_classes": 1},
        {"dropout": -0.1},
        {"dropout": 1.0},
        {"dropout": float("nan")},
        {"dropout": float("inf")},
        {"dropout": True},
    ],
)
def test_invalid_dimensions_and_dropout_are_rejected(options: dict) -> None:
    with pytest.raises(ValueError):
        MeanPoolMLP(**({"vocab_size": 8} | options))


@pytest.mark.parametrize(
    "ids,mask,message",
    [
        (torch.tensor([1]), torch.tensor([True]), "shapes"),
        (torch.tensor([[1]]), torch.tensor([[True, False]]), "shapes"),
        (
            torch.empty((0, 2), dtype=torch.long),
            torch.empty((0, 2), dtype=torch.bool),
            "shapes",
        ),
        (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((2, 0), dtype=torch.bool),
            "shapes",
        ),
        (torch.tensor([[1.0]]), torch.tensor([[True]]), "dense long"),
        (torch.tensor([[1]]), torch.tensor([[1]]), "boolean"),
        (torch.tensor([[-1]]), torch.tensor([[True]]), "vocabulary range"),
        (torch.tensor([[8]]), torch.tensor([[True]]), "vocabulary range"),
        (torch.tensor([[1, 0]]), torch.tensor([[True, True]]), "exactly"),
        (torch.tensor([[1, 2]]), torch.tensor([[True, False]]), "exactly"),
        (
            torch.tensor([[1, 0], [0, 0]]),
            torch.tensor([[True, False], [False, False]]),
            "at least one",
        ),
        (
            torch.tensor([[1]]),
            torch.ones((1, 1), dtype=torch.bool, device="meta"),
            "device",
        ),
    ],
)
def test_invalid_inputs_fail_before_pooling(ids, mask, message: str) -> None:
    model = MeanPoolMLP(8)
    with pytest.raises(ValueError, match=message):
        model(ids, mask)


def test_model_and_inputs_must_share_a_device() -> None:
    model = MeanPoolMLP(8).to("meta")
    with pytest.raises(ValueError, match="device"):
        model(torch.tensor([[1]]), torch.tensor([[True]]))
