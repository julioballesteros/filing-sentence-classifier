"""Reproducible DataLoader construction with the shared sentence collator."""

from typing import cast

import torch
from torch.utils.data import DataLoader, Dataset

from filing_sentence_classifier.data.collate import SentenceBatch, collate_sentences
from filing_sentence_classifier.data.dataset import SentenceItem


def create_dataloader(
    dataset: Dataset[SentenceItem],
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = 2026,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader[SentenceBatch]:
    """Create CPU batches with an independent generator and no dropped examples.

    Set shuffle=True for training; leave it False for validation. Reuse each loader
    across epochs: its generator advances, giving reproducible changing training
    orders. Rebuilding with the same seed restarts that sequence. Each loader owns
    its generator, so validation iteration cannot change the training shuffle or
    the main process's global RNG state.

    num_workers=0 is the initial default. Positive values use spawn with fresh
    workers per iterator. PyTorch derives worker seeds from this generator. The
    dataset and collator are deterministic and perform no random augmentation.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer.")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64).")
    if type(shuffle) is not bool or type(pin_memory) is not bool:
        raise ValueError("shuffle and pin_memory must be booleans.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=collate_sentences,
        drop_last=False,
        num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers else None,
        persistent_workers=False,
        pin_memory=pin_memory,
        in_order=True,
    )
    # PyTorch's generic parameter describes the dataset item, not collate_fn's output.
    return cast(DataLoader[SentenceBatch], loader)
