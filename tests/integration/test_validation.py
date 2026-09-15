"""Exercise validation on saved development rows between training epochs."""

from pathlib import Path

import pytest
import torch

from filing_sentence_classifier.data.dataloader import create_dataloader
from filing_sentence_classifier.data.dataset import SentenceDataset
from filing_sentence_classifier.data.loading import load_split
from filing_sentence_classifier.evaluation.evaluate import (
    Prediction,
    evaluate_predictions,
)
from filing_sentence_classifier.models.mean_pool_mlp import MeanPoolMLP
from filing_sentence_classifier.text.encoding import TextEncoder
from filing_sentence_classifier.text.tokenization import tokenize
from filing_sentence_classifier.text.vocabulary import Vocabulary
from filing_sentence_classifier.training.config import TrainingConfig
from filing_sentence_classifier.training.engine import train_epoch, validate_epoch
from filing_sentence_classifier.training.optimizers import create_optimizer


def test_validation_preserves_training_state_and_matches_shared_evaluation(
    development_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = load_split(development_artifact, "train")
    val = load_split(
        development_artifact, "val", expected_manifest_sha256=train.manifest_sha256
    )
    encoder = TextEncoder(
        Vocabulary.fit(tokenize(text) for text in train.texts), max_length=4
    )
    train_dataset, val_dataset = (
        SentenceDataset(train, encoder),
        SentenceDataset(val, encoder),
    )
    train_loader = create_dataloader(
        train_dataset, batch_size=13, shuffle=True, seed=17
    )
    val_loader = create_dataloader(val_dataset, batch_size=7, seed=17)
    shuffled_val = create_dataloader(val_dataset, batch_size=3, shuffle=True, seed=29)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = MeanPoolMLP(
            len(encoder.vocabulary), embedding_dim=8, hidden_dim=4, dropout=0.5
        )
        optimizer = create_optimizer(model, TrainingConfig())
        trained = train_epoch(model, train_loader, optimizer)
        state = {name: value.clone() for name, value in model.state_dict().items()}
        gradients = {
            name: parameter.grad.clone() for name, parameter in model.named_parameters()
        }
        optimizer_state = {
            parameter: {name: value.clone() for name, value in values.items()}
            for parameter, values in optimizer.state.items()
        }
        rng = torch.get_rng_state().clone()

        def forbidden(*args, **kwargs):
            pytest.fail("Validation must not read files or fit/encode text.")

        with monkeypatch.context() as context:
            context.setattr(Path, "read_bytes", forbidden)
            context.setattr(Path, "read_text", forbidden)
            context.setattr(Vocabulary, "fit", forbidden)
            context.setattr(TextEncoder, "encode", forbidden)
            result = validate_epoch(
                model,
                val_loader,
                label_ids=val.label_ids,
                expected_sample_ids=val.sample_ids,
            )
            shuffled = validate_epoch(
                model,
                shuffled_val,
                label_ids=val.label_ids,
                expected_sample_ids=val.sample_ids,
            )
        assert result.metrics == shuffled.metrics
        assert result.mean_loss == pytest.approx(shuffled.mean_loss)
        assert result.num_examples == len(val_dataset)
        assert result.num_batches == len(val_loader)
        assert result.sample_ids == val.sample_ids
        assert result.targets == val.targets
        assert result.metrics == evaluate_predictions(
            val,
            (
                Prediction(sample_id, prediction)
                for sample_id, prediction in zip(
                    shuffled.sample_ids, shuffled.predicted_labels, strict=True
                )
            ),
        )
        assert not model.training
        assert torch.equal(torch.get_rng_state(), rng)
        for name, value in model.state_dict().items():
            assert torch.equal(value, state[name])
        for name, parameter in model.named_parameters():
            assert torch.equal(parameter.grad, gradients[name])
        for parameter, values in optimizer.state.items():
            for name, value in values.items():
                assert torch.equal(value, optimizer_state[parameter][name])

        next_epoch = train_epoch(model, train_loader, optimizer)
        assert model.training
        assert next_epoch.num_examples == trained.num_examples
        assert all(
            values["step"].item() == trained.num_batches + next_epoch.num_batches
            for values in optimizer.state.values()
        )
