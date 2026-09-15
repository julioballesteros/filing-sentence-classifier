"""Learned token embeddings, masked mean pooling, and a sentence classifier."""

from typing import cast

import torch
from torch import Tensor, nn

from filing_sentence_classifier.text.vocabulary import PAD_ID


class MeanPoolMLP(nn.Module):
    """Map input_ids and attention_mask [B, L] to unnormalized logits [B, C].

    Embeddings are learned from scratch. PAD=0 has no embedding gradient and is
    excluded from both the pooling sum and denominator; UNK=1 is a real token.
    The classifier is Linear -> ReLU -> Dropout -> Linear, with three outputs by
    default. Mean pooling discards token order.

    Inputs and model must share a device. Construction uses PyTorch's standard
    initialization and the caller's RNG state; seeding, device placement, loss,
    optimization, and train/eval mode are controlled outside this module.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        for name, value, minimum in (
            ("vocab_size", vocab_size, 2),
            ("embedding_dim", embedding_dim, 1),
            ("hidden_dim", hidden_dim, 1),
            ("num_classes", num_classes, 2),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if type(dropout) not in (int, float) or not 0 <= dropout < 1:
            raise ValueError("dropout must be a finite number in [0, 1).")

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, num_classes),
        )

    def _validate_inputs(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if (
            not isinstance(input_ids, Tensor)
            or not isinstance(attention_mask, Tensor)
            or input_ids.layout != torch.strided
            or attention_mask.layout != torch.strided
            or input_ids.dtype != torch.long
            or attention_mask.dtype != torch.bool
        ):
            raise ValueError(
                "Expected dense long input_ids and boolean attention_mask."
            )
        if (
            input_ids.ndim != 2
            or attention_mask.shape != input_ids.shape
            or input_ids.numel() == 0
        ):
            raise ValueError("Inputs must have matching, nonempty [B, L] shapes.")
        if (
            input_ids.device != attention_mask.device
            or input_ids.device != self.embedding.weight.device
        ):
            raise ValueError("Inputs, attention_mask, and model must share a device.")
        if bool(((input_ids < 0) | (input_ids >= self.embedding.num_embeddings)).any()):
            raise ValueError("Token IDs must be within the model's vocabulary range.")
        if not torch.equal(attention_mask, input_ids != PAD_ID):
            raise ValueError(
                "attention_mask must be True exactly where input_ids != PAD."
            )
        lengths = attention_mask.sum(dim=1, keepdim=True)
        if bool((lengths == 0).any()):
            raise ValueError(
                "Every sentence must contain at least one non-padding token."
            )
        return lengths

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Return raw logits, preserving batch order and the batch dimension."""
        lengths = self._validate_inputs(input_ids, attention_mask)
        embeddings = cast(Tensor, self.embedding(input_ids))
        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        # Dividing by padded width would make predictions depend on batch members.
        pooled = masked_embeddings.sum(dim=1) / lengths.to(dtype=embeddings.dtype)
        return cast(Tensor, self.classifier(pooled))
