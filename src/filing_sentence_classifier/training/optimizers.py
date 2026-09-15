"""Optimizer construction, separate from model architecture and epoch execution."""

from torch import nn
from torch.optim import AdamW

from filing_sentence_classifier.training.config import TrainingConfig


def create_optimizer(model: nn.Module, config: TrainingConfig) -> AdamW:
    """Create AdamW once per run, after placing the model on its training device.

    All trainable parameters share one group, including biases and embeddings.
    The initial configuration disables weight decay. Use the same optimizer across
    epochs to preserve its step counters and moment estimates.
    """
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("The model must have at least one trainable parameter.")
    return AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        foreach=False,
        fused=False,
    )
