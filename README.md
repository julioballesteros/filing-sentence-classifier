# Filing Sentence Classifier

Sentence-level classification of forward-looking statements in English financial filings. The project combines reproducible data preparation with shared evaluation, building toward a comparison of classical baselines and a PyTorch model trained from scratch.

**Work in progress:** the data pipeline, classical baselines, and complete PyTorch training CLI are implemented, with shared evaluation, early stopping, checkpoints, learning curves, reproducibility artifacts, optional local MLflow tracking, and CI. Neural model selection and deployment are next.

## Classification task

Inputs are individual sentences already extracted from a filing. The dataset defines three classes:

| ID | Label | Meaning |
| --- | --- | --- |
| 0 | `specific fls` | A forward-looking statement about the particular company. |
| 1 | `not-fls` | A statement that is not forward-looking. |
| 2 | `non-specific fls` | A generic forward-looking statement that could apply to any company. |

The implemented baselines are a majority-class classifier and TF-IDF with logistic regression. The PyTorch architecture uses learned embeddings, masked mean pooling, and an MLP. All models use the same saved development partitions and evaluation functions.

## Data preparation results

The pipeline uses a pinned revision of FinanceMTEB/FLS. Of its 2,600 published training sentences, three were quarantined for conflicting labels and four were excluded for normalized text overlap with the published test. Cleaning changed the text of 223 rows while preserving the original labels.

| Partition | Sentences | Role |
| --- | ---: | --- |
| Development train | 2,074 | Model fitting and learned preprocessing |
| Validation | 519 | Model selection and error analysis |
| Published test | 1,000 | Reserved for final evaluation |

Train and validation keep identical cleaned-text groups together, using a fixed grouped, stratified split with seed `2026`. Test access has been limited to automated text checks; its examples and labels have not been explored.

The [dataset documentation](data/README.md) records provenance, cleaning rules, class counts, and limitations. The executed [training audit notebook](notebooks/01_data_exploration.ipynb) provides the exploratory evidence behind those rules.

## Reproduce the data pipeline

Requires Python 3.13 and `uv`. From the repository root:

```bash
uv sync --locked --extra data
uv run --locked --extra data filing-sentence-classifier download-data
uv run --locked --extra data filing-sentence-classifier prepare-data
uv run --locked --extra data filing-sentence-classifier split-data
```

Downloads require access to Hugging Face; preparation and splitting then run locally. Outputs are stored under `data/raw/<revision>/`, `data/interim/<revision>/clean-v1/`, and `data/processed/<revision>/split-v1/`, all excluded from Git. Each command supports `--help` for path options.

The `data` extra supplies acquisition, preparation, and evaluation dependencies. Development tools are included by default; notebook dependencies are available separately through `--group notebooks`.

## Run the majority baseline

After preparing the data:

```bash
DATASET_DIR=data/processed/39b6719f1d7197df4498fea9fce20d4ad782a083/split-v1
uv run --locked --extra data filing-sentence-classifier baseline majority \
  --data-dir "$DATASET_DIR" --output-dir artifacts/runs/majority-v1
```

The classifier counts training labels and predicts the most frequent class for every sentence. Each training row has one vote; ties select the smallest class ID. Validation is loaded after fitting. The fitted class is **`1: not-fls`**, with 1,093 of 2,074 training rows.

The run saves `model.json`, `predictions.val.jsonl`, `metrics.val.json`, and `manifest.json`. The model can be restored with [`MajorityClassifier.from_dict`](src/filing_sentence_classifier/baselines/majority.py) without fitting or optional dependencies. The run manifest records input/output hashes, the effective recipe, code hashes, and environment versions. Repeating an identical run verifies its files; changed results require another output directory. `--manifest-sha256` optionally pins the input artifact.

The [validation report](reports/majority-v1/README.md) contains per-class scores, the confusion matrix, and selected run artifacts. The published test remains reserved for final evaluation.

## Run TF-IDF + logistic regression

```bash
DATASET_DIR=data/processed/39b6719f1d7197df4498fea9fce20d4ad782a083/split-v1
uv run --locked --extra data filing-sentence-classifier baseline tfidf \
  --data-dir "$DATASET_DIR" --config configs/experiments/tfidf-bigram-c10-v1.toml \
  --output-dir artifacts/runs/tfidf-bigram-c10-v1
```

The [selected configuration](configs/experiments/tfidf-bigram-c10-v1.toml) uses word unigrams/bigrams, `min_df=2`, sublinear TF, and L2-regularized multinomial logistic regression (`C=10`, L-BFGS). Vocabulary, IDF, and classifier coefficients are fitted together on train in a scikit-learn `Pipeline`. Validation uses only `transform`/`predict`. The solver runs with one native computation thread; nonconvergence stops publication.

| Model | Evaluation split | Macro-F1 | Accuracy |
| --- | --- | ---: | ---: |
| Majority class | Validation (519 sentences) | 0.2298 | 0.5260 |
| TF-IDF + logistic regression, initial `C=1` | Validation (519 sentences) | 0.6375 | 0.7360 |
| TF-IDF + logistic regression, selected `C=10` | Validation (519 sentences) | **0.6924** | **0.7495** |

A [four-configuration comparison](reports/tfidf-selection-v1/README.md) tested unigrams versus unigrams/bigrams and `C=1` versus `C=10`, selecting by unrounded validation macro-F1 under a declared tie rule. The selected model improves macro-F1 by 0.0548 over the initial reference. Its advantage over unigrams with `C=10` is only 0.0031 on this partition; specific FLS recall remains **0.4651**. The report includes per-class scores, the confusion matrix, and the selected model's hash. Test remains reserved for final evaluation.

The run contains `config.toml`, `model.joblib` (the fitted vectorizer and classifier), `model.json` (metadata), predictions, metrics, and a manifest. [`load_tfidf_model`](src/filing_sentence_classifier/baselines/tfidf.py) checks the supplied model hash and restores the pipeline without training data. Load joblib files only from trusted runs and use the recorded dependency versions: joblib can execute code during loading. [Configuration options](configs/README.md) are recorded in each run's effective recipe.

## Tokenization for the PyTorch model

The versioned [`tokenize`](src/filing_sentence_classifier/text/tokenization.py) function accepts prepared text and returns an immutable tuple of tokens using only the Python standard library:

```python
from filing_sentence_classifier.text.tokenization import tokenize

tokenize("We don’t expect long-term growth of 12.5%.")
# ('we', "don't", 'expect', 'long', '-', 'term', 'growth', 'of', '12.5', '%', '.')
```

Version `1` lowercases text and maps the curly apostrophe `’` to `'`. It keeps Unicode alphanumeric words and internal apostrophes together, preserves numbers such as `2027`, `12.5`, and `1,250.50`, and emits other punctuation and symbols individually. Hyphens, signs, currencies, and percent symbols remain separate tokens. Decimal points require digits on both sides; `.5` becomes `('.', '5')`. Whitespace is discarded; blank inputs raise `ValueError`.

Negation, function words, and word inflections are retained. The tokenizer performs no fitting or truncation. [`tokenization_recipe()`](src/filing_sentence_classifier/text/tokenization.py) exposes the version and exact rules as JSON-serializable metadata for training artifacts. Source cleaning remains upstream, and the TF-IDF baseline retains its own vectorizer's tokenization.

## Build the vocabulary

After preparing the data, build the neural model's vocabulary from the saved training partition:

```bash
DATASET_DIR=data/processed/39b6719f1d7197df4498fea9fce20d4ad782a083/split-v1
uv run --locked filing-sentence-classifier build-vocabulary \
  --data-dir "$DATASET_DIR" --output-dir artifacts/preprocessing/vocabulary-v1
```

[`Vocabulary`](src/filing_sentence_classifier/text/vocabulary.py) reserves **`<PAD>=0`** and **`<UNK>=1`**. Ordinary tokens are ordered by descending training occurrence count, with ties resolved by ascending Unicode token value. Counts include repetitions within a sentence. The default `--min-frequency 2` removes singletons; there is no size cap unless `--max-size` is supplied. That cap includes the two reserved entries and is applied after frequency filtering. Empty documents, reserved-token collisions, and settings that retain no ordinary tokens are rejected.

On the 2,074 development training sentences, tokenization produces 68,892 occurrences and 5,561 distinct tokens. The default vocabulary retains **2,965 ordinary tokens**, giving **2,967 IDs including PAD/UNK**. The 2,596 removed occurrences map to UNK: **3.77% of training tokens**. These are training statistics; validation and test text are not read by this command.

The artifact contains `vocabulary.json` (ordered tokens, counts, and vocabulary recipe) and `manifest.json` (input hashes, tokenizer rules, code hashes, environment versions, statistics, and the vocabulary checksum). Identical builds verify existing bytes without rewriting them; changed data, rules, or options require another output directory. `--manifest-sha256` optionally pins the development data manifest. Building and restoring the vocabulary use the standard library without additional dependencies.

```python
import json
from pathlib import Path

from filing_sentence_classifier.text.vocabulary import Vocabulary

path = Path("artifacts/preprocessing/vocabulary-v1/vocabulary.json")
vocabulary = Vocabulary.from_dict(json.loads(path.read_text(encoding="utf-8")))
token_id = vocabulary["growth"]  # Returns 1 if absent; never changes the vocabulary.
token = vocabulary.tokens[token_id]
```

The immutable vocabulary handles counting and lookup independently of data loading. [`build_vocabulary`](src/filing_sentence_classifier/data/vocabulary.py) owns train selection, tokenization, and artifact publication. Restoring the JSON validates the schema, reserved IDs, counts, and ordering without fitting again.

## Encode text for the PyTorch model

[`TextEncoder`](src/filing_sentence_classifier/text/encoding.py) combines tokenization, frozen vocabulary lookup, and prefix truncation. Using the vocabulary loaded above:

```python
from filing_sentence_classifier.text.encoding import TextEncoder

encoder = TextEncoder(vocabulary, max_length=128)
encoded = encoder.encode("We expect long-term growth.")
input_ids = encoded.input_ids  # Immutable tuple; unknown tokens use ID 1.
truncated = encoded.truncated

# Store this JSON state with the model; restoration needs no training data.
state = json.dumps(encoder.to_dict(), indent=2, sort_keys=True)
restored = TextEncoder.from_dict(json.loads(state))
assert restored.encode("We expect long-term growth.") == encoded
```

The input contract is prepared text. At inference, apply the existing [`clean_text`](src/filing_sentence_classifier/data/cleaning.py) function before calling the same encoder. Empty or invalid text is rejected. Returned IDs contain no padding or added boundary tokens. The Dataset converts them to tensors; the batch collator adds dynamic padding.

Encoding version `1` keeps the first **128 tokens** by default. This initial limit was chosen using train only: its nearest-rank length percentiles are **p95=64** and **p99=91**, with a maximum of 398. The limit preserves 99.66% of training sentences completely. Longer sentences lose their suffix; `original_length`, `truncated`, and `truncated_tokens` make that loss explicit. `original_unknown_count` covers the full sentence, while `unknown_count` counts only retained IDs. The JSON state includes the vocabulary, tokenization rules, and encoding configuration; incompatible saved recipes are rejected.

| Partition | Truncated sentences | Discarded tokens | UNK before truncation | UNK after truncation |
| --- | ---: | ---: | ---: | ---: |
| Train | 7 / 2,074 (0.34%) | 445 / 68,892 (0.65%) | 3.77% | 3.78% |
| Validation | 3 / 519 (0.58%) | 258 / 17,582 (1.47%) | 6.06% | 6.04% |

Validation was encoded for diagnostics after fixing the limit, without refitting or changing it. UNK rates use token occurrences as the denominator. [`encoding_statistics`](src/filing_sentence_classifier/text/encoding.py) reproduces these aggregate diagnostics from encoded sentences, for example:

```python
from filing_sentence_classifier.data.loading import load_split
from filing_sentence_classifier.text.encoding import encoding_statistics

data_dir = Path("data/processed/39b6719f1d7197df4498fea9fce20d4ad782a083/split-v1")
train = load_split(data_dir, "train")
statistics = encoding_statistics(encoder.encode(text) for text in train.texts)
```

## Access examples with PyTorch

Install the optional training dependency alongside the data tools:

```bash
uv sync --locked --extra data --extra train
```

The `train` extra adds PyTorch and Matplotlib for learning curves. The lockfile selects PyTorch `2.14.0`; uv uses the official CPU wheel (`2.14.0+cpu`) on Linux/Windows and the native PyPI wheel on macOS. The index is explicitly scoped to PyTorch, following the [uv integration guidance](https://docs.astral.sh/uv/guides/integration/pytorch/). Data preparation, text encoding, and classical baselines remain usable with their existing extras.

[`SentenceDataset`](src/filing_sentence_classifier/data/dataset.py) implements PyTorch's [integer-indexed Dataset interface](https://docs.pytorch.org/docs/2.14/data.html#map-style-datasets). It consumes a verified `LoadedSplit` and an existing `TextEncoder`. Using the encoder and data directory above:

```python
from filing_sentence_classifier.data.dataset import SentenceDataset

train = load_split(data_dir, "train")
dataset = SentenceDataset(train, encoder)
example = dataset[0]
assert example["input_ids"].ndim == 1
assert example["label"].ndim == 0
```

Each item contains a CPU `torch.long` tensor of unpadded `input_ids`, a scalar `torch.long` `label`, the original `sample_id`, and the encoder's length, truncation, and unknown-token metadata. Saved row order and class IDs are preserved; IDs must be contiguous from zero for use as cross-entropy targets. `dataset.split` retains the source hashes and class mapping, and `dataset.encodings` exposes immutable cached results for diagnostics.

The Dataset encodes each sentence once during construction, keeping the small corpus in memory. Each access creates fresh tensors, so in-place changes cannot corrupt later reads. It performs no file access or fitting and does not retain the encoder, allowing worker processes to receive the Dataset using `spawn`. Shuffling, sampling, and padding are controlled outside this component.

## Build batches

[`collate_sentences`](src/filing_sentence_classifier/data/collate.py) pads sentences on the right with `PAD=0` up to the longest sequence in each batch. It returns CPU tensors: `input_ids` (`long`) and `attention_mask` (`bool`) with shape `[B, L]`, plus `labels` (`long`) with shape `[B]`. The mask is `True` for real tokens, including `UNK=1`, so padding can be excluded from mean pooling. Empty sequences are rejected.

Using the training Dataset above, [`create_dataloader`](src/filing_sentence_classifier/data/dataloader.py) supplies this collator and a separate seeded generator for each loader:

```python
from filing_sentence_classifier.data.dataloader import create_dataloader

val = load_split(data_dir, "val", expected_manifest_sha256=train.manifest_sha256)
val_dataset = SentenceDataset(val, encoder)
train_loader = create_dataloader(dataset, batch_size=32, shuffle=True, seed=2026)
val_loader = create_dataloader(val_dataset, batch_size=32)

for batch in train_loader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
```

`sample_ids` preserve the order within each batch; `lengths`, `original_lengths`, `truncated`, `truncated_tokens`, `unknown_counts`, and `original_unknown_counts` stay aligned with them. Collation neither encodes again nor truncates or sorts sentences. [`collation_recipe()`](src/filing_sentence_classifier/data/collate.py) exposes the versioned padding and mask rules for training artifacts.

Keep the same loaders across epochs: training order changes reproducibly, while validation preserves saved row order. Recreating a loader with the same seed restarts its sequence. Validation iteration does not consume the training generator or the main process's global RNG. Reproducibility assumes the same environment and iteration schedule.

Defaults are `num_workers=0`, `pin_memory=False`, and `drop_last=False`, so the final partial batch is retained. Positive worker counts use `spawn` with ordered results and fresh workers per iterator; scripts using workers must create and iterate loaders inside an `if __name__ == "__main__":` guard. Batches remain on CPU until the training loop moves the required tensors to its device.

## PyTorch model architecture

[`MeanPoolMLP`](src/filing_sentence_classifier/models/mean_pool_mlp.py) maps a batch to three raw logits through learned embeddings, masked mean pooling, and a `Linear → ReLU → Dropout → Linear` classifier. Defaults are `embedding_dim=128`, `hidden_dim=64`, `num_classes=3`, and `dropout=0.0`; each is configurable at construction. With the encoder and validation loader above:

```python
import torch

from filing_sentence_classifier.models.mean_pool_mlp import MeanPoolMLP

model = MeanPoolMLP(vocab_size=len(encoder.vocabulary))
batch = next(iter(val_loader))
model.eval()
with torch.inference_mode():
    logits = model(batch["input_ids"], batch["attention_mask"])
assert logits.shape == (len(batch["sample_ids"]), 3)
```

This inspects an untrained model. Pooling divides by each sentence's real token count, so additional padding or longer batch companions do not change its evaluation logits within numerical tolerance. `PAD=0` is excluded from pooling and has no embedding gradient; `UNK=1` contributes normally and is trainable. Empty rows, out-of-range IDs, and masks inconsistent with padding are rejected. Mean pooling loses word order.

The forward pass consumes only tensors on the model's device and returns logits without softmax, as expected by [`CrossEntropyLoss`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html). Tokenization, labels, loss, seeding, and optimization are handled by their callers. Dropout follows `train()`/`eval()`; the model never changes its own mode. Parameters support PyTorch's standard `state_dict` interface, used by the [checkpoint implementation](src/filing_sentence_classifier/training/checkpoints.py).

## Training configuration and reproducibility

The [initial neural configuration](configs/experiments/mean-pool-mlp-v1.toml) declares the architecture, batch size, epoch limit, early-stopping patience and minimum improvement, AdamW learning rate and weight decay, and CPU runtime settings. [`TrainingConfig`](src/filing_sentence_classifier/training/config.py) validates every field and supports TOML input and JSON round trips using only the standard library. Vocabulary size, class mapping, and preprocessing remain properties of the supplied artifacts.

[`configure_runtime`](src/filing_sentence_classifier/training/reproducibility.py) explicitly initializes Python, PyTorch, and NumPy's global RNG if NumPy is installed. The reference uses training seed **17**, CPU, float32, one computation thread, and zero DataLoader workers. It enables deterministic algorithms in error mode. Call it once before creating the model and loaders, and pass the training seed to each loader's independent generator. The saved partition continues to use seed `2026`.

The [configuration documentation](configs/README.md#pytorch-training) shows how to connect these components. Tests reproduce initialization, batch orders, dropout outputs, training histories, saved predictions, and selected weights across fresh processes in the same environment. Reproduction across different platforms or dependency versions is outside this guarantee, consistent with [PyTorch's reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

## Run a training epoch

[`train_epoch`](src/filing_sentence_classifier/training/engine.py) implements `zero_grad → forward → CrossEntropyLoss → backward → optimizer.step` over the supplied batches. With the configured model and loader from the configuration example:

```python
from filing_sentence_classifier.training.engine import train_epoch
from filing_sentence_classifier.training.optimizers import create_optimizer

optimizer = create_optimizer(model, config)
result = train_epoch(model, train_loader, optimizer, device=config.runtime.device)
print(result.mean_loss, result.num_examples, result.num_batches)
```

Create the optimizer once and reuse it across epochs so AdamW retains its moment estimates and step counters. Each call sets `model.train()`, enables autograd, and clears gradients before each batch. The optimizer uses the configured learning rate and weight decay; its [fixed numerical choices](configs/README.md#pytorch-training) are explicit.

The returned `EpochResult` contains detached Python values. Mean loss weights each batch by its actual number of examples, including the last partial batch. It describes the forward passes observed during training, before each update. Cross-entropy uses raw logits, equal example weights, no label smoothing, and no ignored targets.

Non-finite logits, loss, gradients, or updated parameters abort the epoch with the batch number; completed updates are not rolled back. The optimizer must own exactly the model's trainable parameters, and each must receive a gradient. The epoch function consumes batches without loading data, fitting preprocessing, reseeding, or writing artifacts. Epoch orchestration and checkpoint I/O live in separate training modules.

## Validate an epoch

[`validate_epoch`](src/filing_sentence_classifier/training/engine.py) evaluates the current model over the saved validation partition. Using `val` and `val_loader` from the batching example:

```python
from filing_sentence_classifier.training.engine import validate_epoch

validation = validate_epoch(
    model,
    val_loader,
    label_ids=val.label_ids,
    expected_sample_ids=val.sample_ids,
    device=config.runtime.device,
)
print(validation.mean_loss, validation.metrics.macro_f1)
```

The function sets `model.eval()` and runs under `torch.inference_mode()`. It leaves the model in evaluation mode; the next `train_epoch` call restores training mode. Validation does not clear existing gradients, update parameters or BatchNorm statistics, or call an optimizer. Tests verify that validation between training epochs preserves model state, gradients, optimizer state, and the training RNG.

`ValidationResult` contains mean cross-entropy, sample/batch counts, shared classification metrics, and aligned `sample_ids`, `targets`, and `predicted_labels` tuples in loader traversal order. Loss is weighted by actual batch size. Macro-F1, accuracy, per-class scores, and the confusion matrix are computed once over all predictions; every declared class contributes, including absent classes. Predictions use the highest logit, with ties selecting the lowest class ID.

Declared classes must match the zero-based logit columns. The expected sample IDs define the complete evaluation scope: duplicates, missing rows, unexpected IDs, and invalid or non-finite outputs raise an error instead of returning partial metrics. Validation requires the existing `data` extra for shared metrics alongside `train`. Data loading and artifact publication stay outside the epoch function.

The [memorization integration test](tests/integration/test_overfit.py) checks that the complete neural pipeline can fit a tiny synthetic dataset, with a fixed seed, dropout and weight decay disabled, and a bounded epoch count. It requires 100% accuracy and mean cross-entropy ≤ 0.05 on those same examples. A separate manual check on 24 FLS training sentences also passed after 6 epochs (24/24 correct, loss 0.0171); this checks implementation correctness, not generalization.

## Fit and restore the best checkpoint

[`fit`](src/filing_sentence_classifier/training/fit.py) coordinates the existing epoch functions and creates one AdamW optimizer for the run. Supply an initialized model matching `config` and reusable train/validation loaders, constructed after runtime initialization:

```python
from pathlib import Path
from filing_sentence_classifier.training.fit import fit

result = fit(
    model,
    train_loader,
    val_loader,
    config=config,
    label_ids=val.label_ids,
    validation_sample_ids=val.sample_ids,
    checkpoint_path=Path("artifacts/runs/mean-pool-mlp-v1/checkpoints/best.pt"),
)
print(result.best_epoch, result.best_validation.metrics.macro_f1)
```

The highest **unrounded validation macro-F1** selects the checkpoint; exact ties keep the earliest epoch, regardless of loss. The initial configuration allows 30 epochs with `patience=5` and `min_delta=0.0`. Patience resets when macro-F1 exceeds its last reset value by more than `min_delta`; smaller gains can accumulate from that reference. Every strict improvement still saves the best weights, even when it does not reset patience. Training stops after the configured number of consecutive epochs without a sufficient gain, or at the epoch limit.

On completion, the supplied model contains the best weights in evaluation mode, with gradients cleared. `FitResult` contains aggregate per-epoch losses and metrics, the best epoch, whether training stopped before the epoch limit, and fresh validation predictions from the restored model. It does not retain every epoch's prediction tensors. The caller owns runtime initialization, data loading, and publication of the run history.

[`save_checkpoint` and `load_checkpoint`](src/filing_sentence_classifier/training/checkpoints.py) store a versioned `state_dict`, model type, full training configuration, epoch, and selection score. Each run requires a new checkpoint path; later improvements replace that run's file atomically. Failed writes preserve the previous checkpoint. Loading uses `weights_only=True` on CPU and checks metadata, tensor keys, shapes, dtypes, and finite values before copying weights into a matching model. A fresh-process integration test verifies identical logits after restoration, following [PyTorch's state-dictionary guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

This checkpoint restores model weights. The matching encoder remains a separate artifact; optimizer, loader, and RNG states for exact training resumption are not stored. A deployment bundle is a later component.

## Run neural training

The [initial neural reference](reports/mean-pool-mlp-v1/README.md) reaches validation **macro-F1 0.6919** and **accuracy 0.7360**, compared with 0.6924 and 0.7495 for selected TF-IDF. It selects epoch 8 and stops after epoch 13. A fresh process using the built wheel reproduces the training history, predictions, and selected weights exactly in the same environment. Learning curves show overfitting; this run establishes the starting configuration for subsequent neural model selection.

After preparing the data and building the vocabulary:

```bash
DATASET_DIR=data/processed/39b6719f1d7197df4498fea9fce20d4ad782a083/split-v1
uv run --locked --extra data --extra train filing-sentence-classifier train \
  --data-dir "$DATASET_DIR" \
  --vocabulary-dir artifacts/preprocessing/vocabulary-v1 \
  --config configs/experiments/mean-pool-mlp-v1.toml \
  --output-dir artifacts/runs/mean-pool-mlp-v1
```

The command verifies the saved vocabulary's checksum and training provenance, constructs the encoder without fitting, initializes the runtime, and trains on the frozen development partition. Validation macro-F1 selects the checkpoint; saved predictions come from its restored weights and are checked by the shared evaluator. `--max-length` defaults to `128`; `--manifest-sha256` optionally pins the data artifact. Raw data and the published test are not loaded.

| Run artifact | Contents |
| --- | --- |
| `config.toml`, `encoder.json` | Original training settings and complete frozen text encoder |
| `data-manifest.json`, `vocabulary-manifest.json` | Input provenance |
| `history.jsonl`, `learning-curves.png` | Completed epochs' losses and validation metrics |
| `checkpoints/best.pt` | Selected model weights and checkpoint metadata |
| `predictions.val.jsonl`, `metrics.val.json` | Restored-model predictions and shared evaluation report |
| `summary.json`, `manifest.json` | Selected epoch, timing, run status, effective settings, environment, and file hashes |
| `source/` | Imported Python sources, plus project configuration, lockfile, and Git state when available |

Each run requires a new output directory. History is flushed after every completed epoch; handled failures retain it and any best checkpoint, with `status: failed` in the manifest. A hard interruption may leave `status: running`, identifying an incomplete run. Successful runs end with `status: completed`. Reproduction uses another directory; timestamps and durations are expected to differ.

Source snapshots include untracked Python files and work with installed wheels. The manifest separately records whether the imported code matches the checkout and whether its commit alone reproduces that source state. Run artifacts remain available locally under `artifacts/runs/`; selected aggregate evidence is published in [reports](reports/README.md).

[`run_training`](src/filing_sentence_classifier/training/run.py) owns this lifecycle; `fit` remains responsible for epochs and selection, exposing an optional `on_epoch` callback. [Artifact I/O](src/filing_sentence_classifier/training/artifacts.py) and [headless plots](src/filing_sentence_classifier/training/plots.py) are separate modules.

## Track training with MLflow

Install the optional `tracking` extra and enable tracking for a new run:

```bash
uv sync --locked --extra data --extra train --extra tracking
DATASET_DIR=data/processed/39b6719f1d7197df4498fea9fce20d4ad782a083/split-v1
uv run --no-sync filing-sentence-classifier train \
  --data-dir "$DATASET_DIR" \
  --vocabulary-dir artifacts/preprocessing/vocabulary-v1 \
  --config configs/experiments/mean-pool-mlp-v1.toml \
  --output-dir artifacts/runs/mean-pool-mlp-v1-tracked \
  --mlflow-dir artifacts/mlflow \
  --experiment-name filing-sentence-classifier
```

`--mlflow-dir` opts in to a SQLite database (`mlflow.db`) and an `artifacts/` store inside that directory, both outside Git under the example path. `--experiment-name` defaults to `filing-sentence-classifier`. The local run manifest records the MLflow version, experiment/run IDs, and storage URIs. Storage must be separate from the run directory. Omitting `--mlflow-dir` requires no MLflow installation or import.

The [`MLflowTracker`](src/filing_sentence_classifier/training/tracking.py) implements a small `RunTracker` protocol consumed by run orchestration. It uses an explicitly configured [MLflow client](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html), leaving the fluent tracking URI and active run untouched. It disables MLflow telemetry for the process and uses local storage; no tracking server is required while training.

Parameters include the effective model/training/runtime settings, input hashes, vocabulary size, and sequence limit. Epoch metrics use `train/loss`, `val/loss`, `val/macro_f1`, `val/accuracy`, and per-class precision/recall/F1 with the epoch as their step. The `summary/` metrics describe the selected checkpoint, so the last epoch's score is not mistaken for the final result. Completed runs copy their full local artifacts, including the encoder, checkpoint, history, curves, environment, and source snapshot.

Tracking is synchronous. Its epoch calls contribute to the recorded training duration; final MLflow artifact copying occurs after local timing ends. Tracking errors fail an explicitly tracked run while retaining local evidence. History is flushed before logging each epoch. Handled failures copy available artifacts and close MLflow as `FAILED`; keyboard interruptions use `KILLED`. Failure to record cleanup is captured separately as `tracking_error`, preserving the original error. A hard process termination can leave a run `RUNNING`.

To inspect runs, start the local UI from the repository root and open `http://127.0.0.1:8080`:

```bash
MLFLOW_DISABLE_TELEMETRY=true uv run --no-sync mlflow server \
  --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db \
  --host 127.0.0.1 --port 8080 --no-serve-artifacts
```

Integration tests compare tracked and untracked training in fresh processes, including dropout, and require identical histories, predictions, and selected weights. They also verify artifact copies, failed runs, and isolation from an unrelated active MLflow run.

## Evaluate saved predictions

The `evaluate` command accepts `train` or `val` and requires a prediction file supplied by the caller. Each JSONL row must contain exactly `sample_id` and integer `predicted_label`. For example, with the placeholder replaced by an ID from the selected partition:

```json
{"sample_id": "<sample_id from the saved partition>", "predicted_label": 1}
```

```bash
DATASET_DIR=data/processed/39b6719f1d7197df4498fea9fce20d4ad782a083/split-v1
uv run --locked --extra data filing-sentence-classifier evaluate \
  --data-dir "$DATASET_DIR" --split val --predictions predictions.jsonl
```

Predictions are aligned by ID, so their order is arbitrary; duplicate, missing, or unexpected IDs fail evaluation. Ground truth comes from the verified saved partition. The JSON report contains metrics, class mapping, data/prediction hashes, evaluation code hashes, and environment versions. Use `--manifest-sha256` to require a specific data artifact.

**Macro-F1 is the primary metric**, accompanied by accuracy, per-class precision/recall/F1/support, and a confusion matrix. Every declared class contributes to macro-F1; undefined scores are zero. Matrix rows are true classes and columns are predictions. Metrics are computed over the complete partition rather than averaged across batches.

The same implementation is available through [`classification_metrics`](src/filing_sentence_classifier/evaluation/metrics.py) for aligned vectors and [`evaluate_predictions`](src/filing_sentence_classifier/evaluation/evaluate.py) for predictions identified by sample ID. Class order comes from the [loaded partition](data/README.md#loading-saved-partitions).

## Code organization

```text
src/filing_sentence_classifier/
  data/          Source data, preparation, verified loading, Dataset, and batches
  text/          Tokenization, immutable vocabularies, encoding, and text diagnostics
  models/        PyTorch architectures mapping encoded tensors to logits
  training/      Runtime, epochs, selection, checkpoints, run artifacts, curves, and tracking
  baselines/     Reference classifiers and reproducible run orchestration
  evaluation/    Classification metrics and prediction alignment/reporting
  cli.py         Command wiring and user-facing errors
notebooks/       Exploration using reusable package code
tests/
  unit/          Transformation rules, metrics, and contracts
  integration/   Artifact pipelines, loading, and CLI behavior
data/README.md   Dataset provenance, preparation policy, and artifact formats
reports/         Measured results and selected aggregate run artifacts
```

The source definition is separate from downloading; cleaning and split rules are separate from artifact I/O. The loader consumes frozen partitions using only the standard library. Baseline decision rules are separate from run orchestration and artifact writing. Evaluation operates independently of model implementation, allowing baselines and PyTorch training to share the same metric contract.

Reproducibility is recorded in the lockfile and artifact manifests: source revisions, SHA-256 checksums, transformation versions, split assignments, seeds, code hashes, and environment versions. Data and baseline commands verify existing outputs on repetition. Neural training requires a new run directory, preserves failed attempts, and atomically updates its manifest and best checkpoint.

## Development checks

Install the data, training, and tracking extras to run the complete test suite and type checks:

```bash
uv sync --locked --dev --extra data --extra train --extra tracking
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
uv run --no-sync pytest
```

Tests marked `training`, including the memorization check, are excluded by default. Run them explicitly with `uv run --no-sync pytest -m training`, or run the complete suite with `uv run --no-sync pytest -m ""`.

[CI](.github/workflows/ci.yml) runs the standard suite and training checks as separate steps against a non-editable package installation. Tests use synthetic fixtures without downloading the dataset. Local Git hooks are available with `uv run --no-sync pre-commit install`.

## Scope and limitations

The corpus is small and was sampled partly through forward-looking keywords, so it does not establish performance across complete filings. Exact normalized-text separation does not control near duplicates or overlap by company or period. Dataset licensing and differences between published corpus descriptions remain unresolved; the [dataset documentation](data/README.md#limitations-and-unresolved-questions) records the reviewed evidence.
