"""Exercise the architecture with verified partitions and real batching code."""

from pathlib import Path

import torch
from torch import nn

from filing_sentence_classifier.data.dataloader import create_dataloader
from filing_sentence_classifier.data.dataset import SentenceDataset
from filing_sentence_classifier.data.loading import load_split
from filing_sentence_classifier.models.mean_pool_mlp import MeanPoolMLP
from filing_sentence_classifier.text.encoding import TextEncoder
from filing_sentence_classifier.text.tokenization import tokenize
from filing_sentence_classifier.text.vocabulary import Vocabulary


def test_saved_partition_batches_produce_aligned_logits_and_finite_gradients(
    development_artifact: Path,
) -> None:
    train = load_split(development_artifact, "train")
    encoder = TextEncoder(Vocabulary.fit(tokenize(text) for text in train.texts))
    dataset = SentenceDataset(train, encoder)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = MeanPoolMLP(len(encoder.vocabulary))
    loader = create_dataloader(dataset, batch_size=13, shuffle=True, seed=17)
    by_id = {sample_id: index for index, sample_id in enumerate(train.sample_ids)}
    seen_ids = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["input_ids"], batch["attention_mask"])
            assert logits.shape == (len(batch["sample_ids"]), len(train.label_ids))
            assert torch.isfinite(logits).all()
            for row, sample_id in enumerate(batch["sample_ids"]):
                item = dataset[by_id[sample_id]]
                ids = item["input_ids"].unsqueeze(0)
                individual = model(ids, ids != 0)
                torch.testing.assert_close(logits[row], individual[0])
                seen_ids.append(sample_id)
    assert sorted(seen_ids) == sorted(train.sample_ids)

    model.train()
    batch = next(iter(loader))
    loss = nn.CrossEntropyLoss()(
        model(batch["input_ids"], batch["attention_mask"]), batch["labels"]
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
