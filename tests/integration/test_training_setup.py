"""Verify configured CPU training and batch/dropout sequences across processes."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_cpu_training_reproduces_weights_losses_and_epoch_sequences_across_processes(
    development_artifact: Path,
    tmp_path: Path,
) -> None:
    script = tmp_path / "check_setup.py"
    script.write_text(
        """import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from filing_sentence_classifier.data.dataloader import create_dataloader
from filing_sentence_classifier.data.dataset import SentenceDataset
from filing_sentence_classifier.data.loading import load_split
from filing_sentence_classifier.models.mean_pool_mlp import MeanPoolMLP
from filing_sentence_classifier.text.encoding import TextEncoder
from filing_sentence_classifier.text.tokenization import tokenize
from filing_sentence_classifier.text.vocabulary import Vocabulary
from filing_sentence_classifier.training.config import ModelConfig, RuntimeConfig, TrainingConfig
from filing_sentence_classifier.training.engine import train_epoch
from filing_sentence_classifier.training.optimizers import create_optimizer
from filing_sentence_classifier.training.reproducibility import configure_runtime

config = TrainingConfig(
    model=ModelConfig(8, 4, 0.5),
    runtime=RuntimeConfig(seed=int(sys.argv[2])),
    batch_size=13,
)
metadata = configure_runtime(config.runtime)
train = load_split(Path(sys.argv[1]), "train")
encoder = TextEncoder(Vocabulary.fit(tokenize(text) for text in train.texts))
dataset = SentenceDataset(train, encoder)
model = MeanPoolMLP(
    len(encoder.vocabulary),
    embedding_dim=config.model.embedding_dim,
    hidden_dim=config.model.hidden_dim,
    dropout=config.model.dropout,
    num_classes=len(train.label_ids),
)
initial = {name: value.tolist() for name, value in model.state_dict().items()}
loader = create_dataloader(
    dataset, batch_size=config.batch_size, shuffle=True,
    seed=config.runtime.seed, num_workers=config.runtime.num_workers,
)
validation = create_dataloader(dataset, batch_size=17, seed=config.runtime.seed)
epochs, logits = [], []
model.train()
with torch.no_grad():
    for _ in range(2):
        order, outputs = [], []
        for batch in loader:
            order.extend(batch["sample_ids"])
            outputs.extend(model(batch["input_ids"], batch["attention_mask"]).tolist())
        epochs.append(order)
        logits.append(outputs)
        # Evaluation loading must not consume the model's dropout RNG.
        before = torch.get_rng_state().clone()
        list(validation)
        assert torch.equal(before, torch.get_rng_state())
assert epochs[0] != epochs[1]
assert all(sorted(order) == sorted(train.sample_ids) for order in epochs)
optimizer = create_optimizer(model, config)
history = [asdict(train_epoch(model, loader, optimizer)) for _ in range(2)]
trained = {name: value.tolist() for name, value in model.state_dict().items()}
assert trained != initial
assert all(result["num_examples"] == len(dataset) for result in history)
assert all(result["num_batches"] == len(loader) for result in history)
assert all(state["step"].item() == 2 * len(loader) for state in optimizer.state.values())
assert torch.count_nonzero(model.embedding.weight[0]) == 0
fingerprint = hashlib.sha256(json.dumps({
    "weights": initial, "epochs": epochs, "logits": logits,
    "trained_weights": trained, "training_history": history,
    "python_draw": random.random(), "numpy_draw": np.random.random(3).tolist(),
}, sort_keys=True).encode()).hexdigest()
print(json.dumps({"fingerprint": fingerprint, "runtime": metadata}))
""",
        encoding="utf-8",
    )

    def execute(seed: int, hash_seed: str) -> dict:
        result = subprocess.run(
            [sys.executable, str(script), str(development_artifact), str(seed)],
            cwd=tmp_path,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return json.loads(result.stdout)

    first = execute(17, "1")
    assert first == execute(17, "2")
    assert first["fingerprint"] != execute(29, "1")["fingerprint"]


def test_config_module_works_without_optional_dependencies(tmp_path: Path) -> None:
    # -S removes site-packages; only the package source is supplied explicitly.
    source = Path(__file__).parents[2] / "src"
    script = f"""import json
import sys
sys.path.insert(0, {str(source)!r})
from filing_sentence_classifier.training.config import TrainingConfig
config = TrainingConfig()
assert TrainingConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
assert "torch" not in sys.modules
assert "numpy" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-I", "-S", "-c", script],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
