"""Check that the neural pipeline can memorize a tiny separable dataset."""

import pytest
import torch

from filing_sentence_classifier.data.dataloader import create_dataloader
from filing_sentence_classifier.data.dataset import SentenceDataset
from filing_sentence_classifier.data.loading import (
    LoadedSplit,
    SentenceRecord,
    SplitName,
)
from filing_sentence_classifier.models.mean_pool_mlp import MeanPoolMLP
from filing_sentence_classifier.text.encoding import TextEncoder
from filing_sentence_classifier.text.tokenization import tokenize
from filing_sentence_classifier.text.vocabulary import Vocabulary
from filing_sentence_classifier.training.config import ModelConfig, TrainingConfig
from filing_sentence_classifier.training.engine import train_epoch, validate_epoch
from filing_sentence_classifier.training.optimizers import create_optimizer


@pytest.mark.training
def test_neural_pipeline_memorizes_tiny_dataset() -> None:
    examples = (
        ("We expect growth.", 0),
        ("We expect revenue growth next year.", 0),
        ("Our sales will increase.", 0),
        ("Revenue increased.", 1),
        ("We reported revenue last year.", 1),
        ("Sales were stable.", 1),
        ("Results may vary.", 2),
        ("Economic conditions could change.", 2),
        ("Future outcomes are uncertain.", 2),
    )
    names = ("specific", "historical", "generic")
    split = LoadedSplit(
        name=SplitName.TRAIN,
        records=tuple(
            SentenceRecord(f"sample-{i}", i, f"group-{i}", text, label, names[label])
            for i, (text, label) in enumerate(examples)
        ),
        label_ids=(0, 1, 2),
        label_names=names,
        # In-memory synthetic fixture; no source artifact or downloads.
        manifest_sha256="",
        records_sha256="",
    )
    encoder = TextEncoder(
        Vocabulary.fit(map(tokenize, split.texts), min_frequency=1), max_length=16
    )
    dataset = SentenceDataset(split, encoder)
    config = TrainingConfig(
        model=ModelConfig(embedding_dim=16, hidden_dim=16, dropout=0.0),
        batch_size=4,
        max_epochs=50,
        learning_rate=0.03,
        weight_decay=0.0,
    )
    train_loader = create_dataloader(
        dataset, batch_size=config.batch_size, shuffle=True, seed=config.runtime.seed
    )
    evaluation_loader = create_dataloader(
        dataset, batch_size=config.batch_size, seed=config.runtime.seed
    )
    threads = torch.get_num_threads()
    # Keep this small CPU check fast and restore process state for other tests.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.runtime.seed)
        torch.set_num_threads(1)
        try:
            model = MeanPoolMLP(
                len(encoder.vocabulary),
                num_classes=len(split.label_ids),
                embedding_dim=config.model.embedding_dim,
                hidden_dim=config.model.hidden_dim,
                dropout=config.model.dropout,
            ).to(device="cpu", dtype=torch.float32)
            optimizer = create_optimizer(model, config)
            for _ in range(config.max_epochs):
                train_epoch(model, train_loader, optimizer)
                result = validate_epoch(
                    model,
                    evaluation_loader,
                    label_ids=split.label_ids,
                    expected_sample_ids=split.sample_ids,
                )
                if result.metrics.accuracy == 1.0 and result.mean_loss <= 0.05:
                    break
            assert result.sample_ids == split.sample_ids
            assert result.predicted_labels == split.targets
            assert result.mean_loss <= 0.05, (
                f"Failed to memorize {len(dataset)} examples in {config.max_epochs} "
                f"epochs: accuracy={result.metrics.accuracy}, loss={result.mean_loss}."
            )
        finally:
            torch.set_num_threads(threads)
