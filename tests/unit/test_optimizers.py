"""Check AdamW configuration and parameter ownership."""

import pytest
import torch
from torch import nn
from torch.optim import AdamW

from filing_sentence_classifier.training.config import TrainingConfig
from filing_sentence_classifier.training.optimizers import create_optimizer


def test_adamw_uses_configured_values_and_only_trainable_parameters() -> None:
    model = nn.ParameterDict(
        {
            "weight": nn.Parameter(torch.ones(2)),
            "bias": nn.Parameter(torch.zeros(1)),
            "frozen": nn.Parameter(torch.ones(1), requires_grad=False),
        }
    )
    config = TrainingConfig(learning_rate=0.003, weight_decay=0.02)
    optimizer = create_optimizer(model, config)
    assert isinstance(optimizer, AdamW)
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert {id(parameter) for parameter in group["params"]} == {
        id(model["weight"]),
        id(model["bias"]),
    }
    assert group["lr"] == 0.003
    assert group["weight_decay"] == 0.02
    assert group["betas"] == (0.9, 0.999)
    assert group["eps"] == 1e-8
    assert not group["amsgrad"]
    assert not group["foreach"]
    assert not group["fused"]
    assert not optimizer.state


@pytest.mark.parametrize(
    "model",
    [
        nn.Identity(),
        nn.ParameterList([nn.Parameter(torch.ones(1), requires_grad=False)]),
    ],
)
def test_models_without_trainable_parameters_are_rejected(model) -> None:
    with pytest.raises(ValueError, match="trainable parameter"):
        create_optimizer(model, TrainingConfig())
