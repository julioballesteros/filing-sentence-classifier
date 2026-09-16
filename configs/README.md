# Experiment configurations

## TF-IDF references

[experiments/tfidf-logreg-v1.toml](experiments/tfidf-logreg-v1.toml) defines the initial TF-IDF/logistic regression reference. Pass it to `baseline tfidf --config PATH`; paths are relative to the working directory. All fields are required, and unknown fields are rejected.

| Field | Initial value | Meaning |
| --- | --- | --- |
| `tfidf.ngram_range` | `[1, 2]` | Word unigram and bigram features |
| `tfidf.min_df` | `2` | Minimum number of training sentences containing a feature |
| `logistic_regression.c` | `1.0` | Inverse regularization strength |
| `logistic_regression.max_iter` | `1000` | Solver iteration limit; nonconvergence is an error |
| `logistic_regression.tol` | `0.0001` | Solver stopping tolerance |

The fixed recipe uses lowercase word features, tokens of at least two word characters, no stop-word list or accent stripping, sublinear TF, smoothed IDF, L2 normalization, and float64 sparse matrices. There is no feature-count cap. Logistic regression uses L-BFGS, L2 regularization (`l1_ratio=0`), an intercept, and equal row weights. This solver does not use a random seed. See the [scikit-learn solver documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

The complete effective recipe and original TOML bytes are saved with the run. For a new experiment, copy the configuration and choose a distinct output directory. Model fitting always consumes the saved training partition; configuration changes do not resplit data. The pinned dataset source remains in [data/fls.py](../src/filing_sentence_classifier/data/fls.py).

The [bounded selection study](../reports/tfidf-selection-v1/README.md) adds three recipes to the initial reference: [unigrams, C=1](experiments/tfidf-unigram-c1-v1.toml), [unigrams, C=10](experiments/tfidf-unigram-c10-v1.toml), and [unigrams/bigrams, C=10](experiments/tfidf-bigram-c10-v1.toml). Only n-gram range and regularization strength vary. Unigrams/bigrams with `C=10` is the selected TF-IDF reference; all four original configurations remain available.

## PyTorch training

[experiments/mean-pool-mlp-v1.toml](experiments/mean-pool-mlp-v1.toml) is the initial neural reference. Schema version `2` requires every declared field and rejects extras, invalid types, nonfinite numbers, and unsupported devices. Earlier schema-1 configurations must add `patience` and `min_delta` to `[training]` and set `schema_version = 2`. Settings are frozen dataclasses, independent of file I/O and PyTorch imports.

| Section | Initial settings |
| --- | --- |
| `model` | Embedding dimension `128`, hidden dimension `64`, dropout `0.0` |
| `training` | Batch size `32`, maximum epochs `30`, patience `5`, minimum improvement `0.0`, AdamW learning rate `0.001`, weight decay `0.0` |
| `runtime` | Training seed `17`, device `"cpu"`, workers `0`, computation threads `1` |

The initial learning rate and stopping settings are starting choices, not selected results. Dropout and weight decay are disabled for this reference. `patience` must be a positive integer; `min_delta` must be finite and in `[0, 1)`. Patience counts epochs without a macro-F1 gain strictly greater than `min_delta` over the last patience-reset score. Checkpoint selection independently uses every strict macro-F1 improvement, with ties keeping the earliest epoch. The training CLI accepts data paths, a saved training vocabulary, and `--max-length` separately; these are recorded with the run. It never resplits data or refits preprocessing. Vocabulary size and the number of output classes come from the verified artifacts.

Pass this file to [`train --config PATH`](../README.md#run-neural-training). The command saves its original bytes and effective settings alongside the constructed encoder and selected checkpoint. Configuration changes require a distinct run directory.

Optional [MLflow tracking](../README.md#track-training-with-mlflow) is configured through `--mlflow-dir` and `--experiment-name`, separately from this model configuration. Its storage paths and run identity are recorded in the local manifest.

The [MLP study plan](../reports/mlp-selection-v1/README.md) declares five new candidates: dropout, weight decay, their combination, smaller embeddings, and an expanded train-only vocabulary. Their complete TOMLs are in `experiments/mean-pool-mlp-*-v1.toml`; the plan records their hashes, vocabulary paths, hypotheses, and selection order before execution. The vocabulary candidate requires `artifacts/preprocessing/vocabulary-min1-v1/`; its training TOML alone does not distinguish it from the reference's effective settings.

[`create_optimizer`](../src/filing_sentence_classifier/training/optimizers.py) creates [AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html) with the configured learning rate and weight decay, `betas=(0.9, 0.999)`, `eps=1e-8`, and `amsgrad=False`. Both `foreach` and `fused` are disabled for an explicit implementation choice. One parameter group contains all trainable parameters, including biases and embeddings; frozen parameters are excluded. Construct it after placing the model on its device, and retain it across calls to `train_epoch`. [`fit`](../src/filing_sentence_classifier/training/fit.py) creates and retains its own optimizer and consumes the epoch/stopping limits.

Using the existing `encoder`, verified `train` partition, and `dataset` from the main README:

```python
from pathlib import Path

from filing_sentence_classifier.data.dataloader import create_dataloader
from filing_sentence_classifier.models.mean_pool_mlp import MeanPoolMLP
from filing_sentence_classifier.training.config import TrainingConfig
from filing_sentence_classifier.training.reproducibility import configure_runtime

config_path = Path("configs/experiments/mean-pool-mlp-v1.toml")
config = TrainingConfig.from_toml(config_path.read_bytes())
runtime_metadata = configure_runtime(config.runtime)
model = MeanPoolMLP(
    vocab_size=len(encoder.vocabulary),
    num_classes=len(train.label_ids),
    embedding_dim=config.model.embedding_dim,
    hidden_dim=config.model.hidden_dim,
    dropout=config.model.dropout,
)
train_loader = create_dataloader(
    dataset,
    batch_size=config.batch_size,
    shuffle=True,
    seed=config.runtime.seed,
    num_workers=config.runtime.num_workers,
)
config_metadata = config.to_dict()
```

For validation, supply its Dataset with `shuffle=False` and the same loader settings. Keep loaders across epochs. `configure_runtime` changes process-wide RNG, default device/dtype, thread count, and determinism settings; call it once per run, not per epoch. It does not reset existing loader generators or independently created Python/NumPy RNG instances. Runtime metadata records whether NumPy was seeded and the applied numerical settings. The training CLI saves this metadata in the run manifest.

This version supports CPU explicitly, with float32 and deterministic algorithms that raise on unsupported operations. Positive worker counts use the existing loader's `spawn` behavior; the initial reference uses zero. Accelerators are rejected rather than silently replaced with CPU. Reproducibility requires the same environment and execution schedule; Python hash randomization cannot be changed retroactively by setting an environment variable inside the running process.
