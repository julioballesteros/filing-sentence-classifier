# Filing Sentence Classifier

Sentence-level classification of forward-looking statements in English financial filings. The project combines reproducible data preparation with shared evaluation, building toward a comparison of classical baselines and a PyTorch model trained from scratch.

**Work in progress:** the data pipeline, classical baselines, and complete PyTorch training CLI are implemented, with shared evaluation, early stopping, checkpoints, learning curves, reproducibility artifacts, optional local MLflow tracking, and CI. Model selection and three-seed validation are complete, with delivery artifacts frozen and exported as portable bundles. Inference delivery is verified for PyTorch and TF-IDF through the shared Python API, prediction CLI, CPU benchmarks, and an installed wheel. The [final test campaign](reports/test-v1/README.md) is complete for all five frozen models; comparative analysis and final reporting remain pending.

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
| Published test | 1,000 | Final evaluation after freezing all models |

Train and validation keep identical cleaned-text groups together, using a fixed grouped, stratified split with seed `2026`. During development, test access was limited to automated text checks. Its labels were first loaded for the final evaluation after model selection was frozen.

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

The [validation report](reports/majority-v1/README.md) contains per-class scores, the confusion matrix, and selected run artifacts. Final test scores are recorded separately in the [test campaign](reports/test-v1/README.md).

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

A [four-configuration comparison](reports/tfidf-selection-v1/README.md) tested unigrams versus unigrams/bigrams and `C=1` versus `C=10`, selecting by unrounded validation macro-F1 under a declared tie rule. The selected model improves macro-F1 by 0.0548 over the initial reference. Its advantage over unigrams with `C=10` is only 0.0031 on this partition; specific FLS recall remains **0.4651**. The report includes per-class scores, the confusion matrix, and the selected model's hash. This selection was finalized before test evaluation.

The run contains `config.toml`, `model.joblib` (the fitted vectorizer and classifier), `model.json` (metadata), predictions, metrics, and a manifest. [`load_tfidf_model`](src/filing_sentence_classifier/baselines/tfidf.py) checks the supplied model hash and restores the pipeline without training data. Load joblib files only from trusted runs and use the recorded dependency versions: joblib can execute code during loading. [Configuration options](configs/README.md) are recorded in each run's effective recipe.

Fresh-process reproduction checks exact learned parameters and predictions. Joblib can serialize identical state into different bytes because of pickle reference memoization; each published artifact retains its own checksum. Export preserves the selected artifact's bytes exactly.

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

A [30-example validation error review](reports/validation-errors.md) examines temporal meaning, specificity, negation, and annotation ambiguity alongside full-partition diagnostics. The models share 92 errors; only two MLP errors involve truncated sentences, and 31 contain no unknown tokens. Regularization is the first hypothesis proposed for the next experiments.

The [six-configuration neural study](reports/mlp-selection-v1/README.md) selects **combined dropout (`0.3`) and weight decay (`0.1`)** by the highest unrounded validation macro-F1: **0.7024**, with **0.7437 accuracy** at seed 17. Smaller embeddings and a singleton-retaining vocabulary lower macro-F1. The [selection record](reports/mlp-selection-v1/selection.json) identifies the chosen recipe, vocabulary, encoder, and epoch-8 checkpoint. These development results use all ten initial TF-IDF/MLP configuration slots.

The [three-seed evaluation](reports/mlp-selection-v1/README.md#variation-across-training-seeds) reuses seed 17 and repeats the fixed recipe with seeds 29 and 43. Validation macro-F1 is **0.6961 ± 0.0090** and accuracy **0.7431 ± 0.0029** (mean ± sample SD, n=3). One seed scores below TF-IDF in macro-F1; this does not establish a reliable generalization advantage. The recipe was selected using seed 17, so these results describe variation on the same development split. Every run's checkpoint, predictions, and MLflow records were verified. The delivery seed remains 17, fixed before test evaluation.

The [final artifact freeze](reports/mlp-selection-v1/freeze.json) closes model selection and pins the neural **seed-17, epoch-8 checkpoint**, its encoder and configuration, and the selected TF-IDF pipeline. It records artifact paths, checksums, data identity, environment, and the preceding study evidence for subsequent evaluation and packaging. The existing trained models are retained without refitting on train plus validation.

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

## Inference bundle and prediction contracts

[`BundleManifest` and `verify_bundle`](src/filing_sentence_classifier/artifacts.py) define bundle schema `1`. Each bundle is a portable directory with fixed filenames:

| Family | Files alongside `manifest.json` |
| --- | --- |
| `mean_pool_mlp` | `config.toml`, `model.pt`, `encoder.json` |
| `tfidf_logreg` | `config.toml`, `model.joblib` |

`model.pt` uses the existing checkpoint format; `encoder.json` contains the complete vocabulary, tokenizer recipe, and truncation settings. `model.joblib` contains the fitted vectorizer and classifier. Each bundle includes the original run configuration. The manifest records a versioned `model_id`, explicit class IDs and names, the cleaning version, input limits, dependency versions, source run identity and manifest hash, and each payload's size and SHA-256. It requires no paths to training data or the original run.

`verify_bundle(Path(...))` validates metadata, supported schema and cleaning versions, and the complete file inventory without importing model runtimes or deserializing weights. It rejects missing, extra, altered, and symlinked entries. An optional `expected_manifest_sha256` also pins the metadata. Model adapters check payload semantics and installed dependency compatibility when loading. Changing bundle contents requires a new `model_id`; the schema version identifies the format separately.

The shared [`inference.contracts`](src/filing_sentence_classifier/inference/contracts.py) module defines:

- **Inputs:** `prepare_texts` accepts a sequence of raw sentences, applies the existing cleaning policy, and preserves order and duplicates. Default limits are 256 sentences per request and 10,000 Unicode characters per raw sentence, before cleaning. Invalid items reject the complete request with their zero-based index; an empty request returns an empty tuple. Token truncation belongs to the saved encoder.
- **Outputs:** immutable `Prediction` values carry the model ID, class probabilities, and a truncation flag. The winning label is derived from the largest probability, with the lowest class ID winning exact ties. Scores must be finite, in `[0, 1]`, and sum to one within `1e-6`; they are not silently normalized or assumed calibrated.

A synthetic example of the serialized output contract:

```json
{"schema_version": 1, "model_id": "example-v1", "label_id": 0, "label": "specific fls", "probabilities": {"0": 0.7, "1": 0.2, "2": 0.1}, "truncated": false}
```

Both contract modules use only the Python standard library and shared cleaning code.

## Export a saved model

The [`export_bundle`](src/filing_sentence_classifier/exporting.py) function and `export` command copy the selected checkpoint or fitted pipeline without fitting, deserializing, or changing model bytes. Export uses the standard library and records the original training environment. For the frozen neural model:

```bash
uv run --no-sync filing-sentence-classifier export \
  --run-dir artifacts/runs/mean-pool-mlp-regularized-v1 \
  --output-dir artifacts/bundles/mean-pool-mlp-regularized-v1 \
  --model-id mean-pool-mlp-regularized-v1
```

For the selected classical reference:

```bash
DATASET_DIR=data/processed/39b6719f1d7197df4498fea9fce20d4ad782a083/split-v1
uv run --no-sync filing-sentence-classifier export \
  --run-dir artifacts/runs/tfidf-bigram-c10-v1 \
  --output-dir artifacts/bundles/tfidf-bigram-c10-v1 \
  --model-id tfidf-bigram-c10-v1 \
  --data-manifest "$DATASET_DIR/manifest.json"
```

Legacy TF-IDF runs do not contain a dataset manifest, so export requires that JSON to verify the cleaning recipe and labels against the run's recorded dataset hash. Neural runs already contain a copy. No dataset rows are read. Legacy baseline run IDs use the source directory name; the source manifest hash identifies its exact contents.

`--manifest-sha256` optionally pins the **source run** manifest. Export verifies the files it consumes and checks configuration, encoder metadata, classes, and run status before publishing the complete bundle. Identical repetition verifies existing bytes without rewriting them; changed or incomplete destinations fail. Concurrent exporters to the same destination are excluded by a sibling lock. `--max-characters` and `--max-batch-size` set the saved request limits.

The exported neural bundle contains **4 files / 1,637,623 bytes**; the TF-IDF bundle contains **3 files / 303,884 bytes**, including their manifests. Both retain the model hashes recorded in the [selection freeze](reports/mlp-selection-v1/freeze.json). Bundles are local generated artifacts excluded from Git. Integration tests verify exact probability preservation after exporting synthetic models, moving their bundles, and removing their source runs.

## Predict with the PyTorch bundle

[`Predictor`](src/filing_sentence_classifier/inference/predictor.py) loads the model once and reuses it for raw-text requests. The `train` extra supplies PyTorch; inference does not import scikit-learn, Matplotlib, dataset acquisition tools, or MLflow.

```python
from pathlib import Path

from filing_sentence_classifier.inference.predictor import Predictor

predictor = Predictor.from_bundle(
    Path("artifacts/bundles/mean-pool-mlp-regularized-v1"),
)
results = predictor.predict(
    ["We expect revenue to increase next year.", "Revenue increased last year."],
    batch_size=32,
)
for result in results:
    print(result.to_dict())
```

The [`PyTorch backend`](src/filing_sentence_classifier/inference/pytorch.py) restores the saved encoder and checkpoint into a CPU `float32` model, checks weight keys, dimensions, dtypes and finite values, and runs in evaluation mode with inference mode enabled. It cleans and encodes sentences, pads each batch on the right, masks PAD, and applies softmax in the bundle's class order. `batch_size` controls computation chunks; the saved maximum request size still applies to the complete input. Results are an immutable tuple, preserving order and duplicates and exposing truncation.

Loading requires the recorded package version, the same PyTorch release (allowing local build suffixes such as `+cpu`), and the same Python major/minor version. `from_bundle(..., expected_manifest_sha256=...)` can pin the bundle manifest. Payloads are rechecked as they are read, and `weights_only=True` deserializes those verified bytes. The loaded predictor needs no further file access or network connection. Loading and prediction preserve the caller's RNG, thread count, default dtype, and gradient settings; prediction explicitly disables ambient CPU autocast.

For the selected bundle, the predictor reproduced **all 519 saved validation labels**, with **zero probability difference** from the original checkpoint under the same CPU environment and batch size. Three sentences were flagged as truncated. Synthetic tests also cover different batch sizes, moved bundles, unknown tokens, invalid inputs, and incompatible or altered artifacts.

## Predict with the TF-IDF bundle

The same `Predictor` API selects the [`TF-IDF backend`](src/filing_sentence_classifier/inference/tfidf.py) from the bundle manifest. Its runtime dependencies are supplied by the `data` extra; PyTorch is not required.

```python
from pathlib import Path

from filing_sentence_classifier.inference.predictor import Predictor

predictor = Predictor.from_bundle(Path("artifacts/bundles/tfidf-bigram-c10-v1"))
results = predictor.predict(
    ["We expect revenue to increase next year.", "Revenue increased last year."],
    batch_size=32,
)
for result in results:
    print(result.to_dict())
```

Loading verifies the configuration, fitted feature dimensions, numerical state, and class IDs. It requires the recorded package, scikit-learn, NumPy, SciPy, joblib, and threadpoolctl releases, plus the same Python major/minor. Only load trusted bundles: joblib deserialization can execute code, and checksums verify integrity rather than origin.

The pipeline is loaded once from verified bytes and reused without fitting or further file access. Requests use the shared cleaning and input limits, then the fitted vectorizer's own tokenization. Batched `predict_proba` results are aligned with the bundle's class IDs. TF-IDF applies no token truncation, so `truncated` is always `false`; sentences without known features receive probabilities determined by the fitted intercepts. Native computation uses one thread and restores the caller's thread limits afterward.

The selected bundle reproduced **all 519 validation labels**, with **zero probability difference** from the original pipeline in the recorded environment. Integration tests cover binary and multiclass models, reordered class columns, moved bundles, offline operation without neural dependencies, and rejection of incompatible or altered artifacts.

## Predict from the command line

Classify one sentence using an exported bundle:

```bash
uv run --locked --extra train filing-sentence-classifier predict \
  --bundle artifacts/bundles/mean-pool-mlp-regularized-v1 \
  --text "We expect revenue to increase next year."
```

For multiple sentences, supply a UTF-8 JSONL file with exactly `text` and, optionally, `sample_id` on each line:

```jsonl
{"sample_id": "sentence-1", "text": "We expect revenue to increase next year."}
{"sample_id": "sentence-2", "text": "Revenue increased last year."}
```

```bash
uv run --locked --extra data filing-sentence-classifier predict \
  --bundle artifacts/bundles/tfidf-bigram-c10-v1 \
  --input sentences.jsonl --output predictions.jsonl --batch-size 32
```

The command accepts exactly one of `--text` or `--input`; `--input -` reads stdin. Each output line uses the shared prediction schema: model ID, label ID/name, class probabilities, and truncation flag. An input `sample_id` is copied unchanged; order and duplicates are preserved. IDs must be nonblank strings of at most 256 characters. Input lines have a 1 MiB byte limit and must contain unique JSON keys; blank lines and invalid UTF-8 are rejected. Empty files produce empty output successfully.

The [`JSONL adapter`](src/filing_sentence_classifier/inference/jsonl.py) loads no model itself: the CLI creates one `Predictor` and reuses it throughout the file. Processing holds one batch at a time, capped by both `--batch-size` and the bundle's maximum request size. Sentence limits and cleaning come from the same predictor used by the Python API. `--manifest-sha256` optionally pins the bundle manifest.

Output defaults to stdout (`--output -` is equivalent), with diagnostics on stderr. Exit codes are `0` for success, `1` for input, bundle, runtime, or I/O errors, and `2` for invalid command options. A destination file's parent must already exist; files are published only after all rows succeed, and existing paths are never overwritten. Stdout streams completed batches, so a late error can leave partial output there; check the exit status when piping or redirecting.

Integration tests verify CLI/API parity for both model families, stdin and file input, repeated model reuse, manifest pinning, offline execution from an unrelated directory, and failure cleanup.

## Inference performance

The [inference benchmark](reports/inference-v1/README.md) measures both frozen bundles on the same 519 validation sentences, using an Apple M3 Max and one CPU thread. Each model uses three fresh-process trials, separate warmup passes, and batch sizes 1, 32, and 128.

| Model | Bundle MiB | Median latency, batch 1 | Throughput, batch 32 | Throughput, batch 128 |
| --- | ---: | ---: | ---: | ---: |
| MeanPoolMLP | 1.562 | 0.082 ms | 25,578 sentences/s | 24,511 sentences/s |
| TF-IDF + logistic regression | 0.290 | 0.916 ms | 17,637 sentences/s | 28,199 sentences/s |

These warm timings cover the complete Python prediction API, excluding JSON/file I/O. They describe this workload and machine, rather than a service latency guarantee. The report separates startup, full and partial batches, parameters, dependency versions, and raw timings. The `benchmark` command reproduces the protocol with new output files; it uses validation texts without modifying model artifacts or evaluating test.

The [delivery verification record](reports/inference-v1/delivery.json) closes inference packaging. A [complete integration test](tests/integration/test_delivery.py) trains both model families on synthetic data, exports and moves each bundle, removes the original inputs and run, then predicts in a fresh process outside the checkout with networking and MLflow unavailable. Probabilities match the trained state within tolerance, and incompatible bundle versions fail explicitly. The selected real bundles also reproduce the API's example predictions through the installed wheel.

## Code organization

```text
src/filing_sentence_classifier/
  data/          Source data, preparation, verified loading, Dataset, and batches
  text/          Tokenization, immutable vocabularies, encoding, and text diagnostics
  models/        PyTorch architectures mapping encoded tensors to logits
  training/      Runtime, epochs, selection, checkpoints, run artifacts, curves, and tracking
  baselines/     Reference classifiers and reproducible run orchestration
  evaluation/    Classification metrics and prediction alignment/reporting
  inference/     Public prediction API, input/output contracts, and model backends
  artifacts.py   Portable inference bundle metadata and file integrity checks
  exporting.py   Verified run-to-bundle copying and publication
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

[CI](.github/workflows/ci.yml) builds and installs the wheel, then runs the standard suite and training checks as separate steps. Delivery checks use `FSC_REQUIRE_WHEEL=1` to reject imports from an editable checkout. Tests use synthetic fixtures without downloading the dataset. Local Git hooks are available with `uv run --no-sync pre-commit install`.

## Scope and limitations

The corpus is small and was sampled partly through forward-looking keywords, so it does not establish performance across complete filings. Exact normalized-text separation does not control near duplicates or overlap by company or period. Dataset licensing and differences between published corpus descriptions remain unresolved; the [dataset documentation](data/README.md#limitations-and-unresolved-questions) records the reviewed evidence.
