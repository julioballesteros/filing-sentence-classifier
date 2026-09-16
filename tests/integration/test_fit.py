"""Exercise epoch orchestration and checkpoint loading in a fresh process."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from filing_sentence_classifier.data.dataloader import create_dataloader
from filing_sentence_classifier.data.dataset import SentenceDataset
from filing_sentence_classifier.data.loading import load_split
from filing_sentence_classifier.models.mean_pool_mlp import MeanPoolMLP
from filing_sentence_classifier.text.encoding import TextEncoder
from filing_sentence_classifier.text.tokenization import tokenize
from filing_sentence_classifier.text.vocabulary import Vocabulary
from filing_sentence_classifier.training.config import ModelConfig, TrainingConfig
from filing_sentence_classifier.training.fit import fit


@pytest.mark.training
def test_fit_restores_selected_model_and_fresh_process_recovers_logits(
    development_artifact: Path, tmp_path: Path
) -> None:
    train, val = (
        load_split(development_artifact, "train"),
        load_split(development_artifact, "val"),
    )
    encoder = TextEncoder(Vocabulary.fit(map(tokenize, train.texts), min_frequency=1))
    config = TrainingConfig(
        model=ModelConfig(16, 8, 0.5),
        batch_size=13,
        max_epochs=4,
        patience=2,
        learning_rate=0.03,
    )
    train_loader = create_dataloader(
        SentenceDataset(train, encoder),
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.runtime.seed,
    )
    val_loader = create_dataloader(
        SentenceDataset(val, encoder),
        batch_size=config.batch_size,
        seed=config.runtime.seed,
    )
    checkpoint = tmp_path / "run" / "best.pt"
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.runtime.seed)
        torch.set_num_threads(1)
        try:
            model = MeanPoolMLP(
                len(encoder.vocabulary), embedding_dim=16, hidden_dim=8, dropout=0.5
            )
            initial = {
                name: value.clone() for name, value in model.state_dict().items()
            }
            result = fit(
                model,
                train_loader,
                val_loader,
                config=config,
                label_ids=val.label_ids,
                validation_sample_ids=val.sample_ids,
                checkpoint_path=checkpoint,
            )
            scores = [row.validation_metrics.macro_f1 for row in result.history]
            assert result.best_epoch == scores.index(max(scores)) + 1
            assert result.best_validation.metrics.macro_f1 == max(scores)
            assert result.best_validation.sample_ids == val.sample_ids
            assert result.best_validation.targets == val.targets
            assert all(
                row.training.num_examples == len(train.records)
                for row in result.history
            )
            assert all(
                row.validation_metrics.sample_count == len(val.records)
                for row in result.history
            )
            assert any(
                not torch.equal(value, initial[name])
                for name, value in model.state_dict().items()
            )
            assert not model.training
            assert all(parameter.grad is None for parameter in model.parameters())
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert payload["epoch"] == result.best_epoch
            assert all(
                torch.equal(value, payload["model_state_dict"][name])
                for name, value in model.state_dict().items()
            )
            json.dumps([asdict(row) for row in result.history], allow_nan=False)
            batch = next(iter(val_loader))
            with torch.inference_mode():
                expected = model(batch["input_ids"], batch["attention_mask"])
        finally:
            torch.set_num_threads(threads)

    request = tmp_path / "reload.json"
    request.write_text(
        json.dumps(
            {
                "config": config.to_dict(),
                "vocab_size": len(encoder.vocabulary),
                "input_ids": batch["input_ids"].tolist(),
                "attention_mask": batch["attention_mask"].tolist(),
            }
        ),
        encoding="utf-8",
    )
    script = """import json
import sys
from pathlib import Path
import torch
from filing_sentence_classifier.models.mean_pool_mlp import MeanPoolMLP
from filing_sentence_classifier.training.checkpoints import load_checkpoint
from filing_sentence_classifier.training.config import TrainingConfig

request = json.loads(Path(sys.argv[2]).read_text())
config = TrainingConfig.from_dict(request["config"])
torch.set_num_threads(1)
torch.manual_seed(29)
model = MeanPoolMLP(request["vocab_size"], embedding_dim=config.model.embedding_dim,
    hidden_dim=config.model.hidden_dim, dropout=config.model.dropout)
metadata = load_checkpoint(Path(sys.argv[1]), model, expected_config=config)
assert not model.training
with torch.inference_mode():
    logits = model(torch.tensor(request["input_ids"], dtype=torch.long),
        torch.tensor(request["attention_mask"], dtype=torch.bool))
print(json.dumps({"epoch": metadata.epoch, "logits": logits.tolist()}))
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(checkpoint), str(request)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    restored = json.loads(child.stdout)
    assert restored["epoch"] == result.best_epoch
    torch.testing.assert_close(
        torch.tensor(restored["logits"]), expected, rtol=0, atol=0
    )
