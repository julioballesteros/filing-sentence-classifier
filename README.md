# Filing Sentence Classifier

Sentence-level classification of forward-looking statements in English financial filings. The project combines reproducible data preparation with shared evaluation, building toward a comparison of classical baselines and a PyTorch model trained from scratch.

**Work in progress:** the installable Python package, data pipeline, evaluation API/CLI, both classical baselines, text encoder, PyTorch data loading, model architecture, training and validation epochs, and CI are implemented. Checkpoint selection and full-run orchestration are planned.

## Classification task

Inputs are individual sentences already extracted from a filing. The dataset defines three classes:

| ID | Label | Meaning |
| --- | --- | --- |
| 0 | `specific fls` | A forward-looking statement about the particular company. |
| 1 | `not-fls` | A statement that is not forward-looking. |
| 2 | `non-specific fls` | A generic forward-looking statement that could apply to any company. |

The implemented baselines are a majority-class classifier and TF-IDF with logistic regression. The PyTorch architecture uses learned embeddings, masked mean pooling, and an MLP; a complete neural training run has not yet been published. All models use the same saved development partitions and evaluation functions.

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

The `train` extra adds PyTorch. The lockfile selects version `2.14.0`; uv uses the official CPU wheel (`2.14.0+cpu`) on Linux/Windows and the native PyPI wheel on macOS. The index is explicitly scoped to PyTorch, following the [uv integration guidance](https://docs.astral.sh/uv/guides/integration/pytorch/). Data preparation, text encoding, and classical baselines remain usable with their existing extras.

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

The forward pass consumes only tensors on the model's device and returns logits without softmax, as expected by [`CrossEntropyLoss`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html). Tokenization, labels, loss, seeding, and optimization are handled by their callers. Dropout follows `train()`/`eval()`; the model never changes its own mode. Parameters support PyTorch's standard `state_dict` interface; checkpoint orchestration will accompany the training loop.

## Training configuration and reproducibility

The [initial neural configuration](configs/experiments/mean-pool-mlp-v1.toml) declares the architecture, batch size, epoch limit, AdamW learning rate and weight decay, and CPU runtime settings. [`TrainingConfig`](src/filing_sentence_classifier/training/config.py) validates every field and supports TOML input and JSON round trips using only the standard library. Vocabulary size, class mapping, and preprocessing remain properties of the supplied artifacts.

[`configure_runtime`](src/filing_sentence_classifier/training/reproducibility.py) explicitly initializes Python, PyTorch, and NumPy's global RNG if NumPy is installed. The reference uses training seed **17**, CPU, float32, one computation thread, and zero DataLoader workers. It enables deterministic algorithms in error mode. Call it once before creating the model and loaders, and pass the training seed to each loader's independent generator. The saved partition continues to use seed `2026`.

The [configuration documentation](configs/README.md#pytorch-training) shows how to connect these components. Tests reproduce initialization, batch orders, dropout outputs, and two training epochs' losses and final weights across fresh processes in the same environment. Reproduction across different platforms or dependency versions is outside this guarantee, consistent with [PyTorch's reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html). No neural validation result is reported yet.

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

Non-finite logits, loss, gradients, or updated parameters abort the epoch with the batch number; completed updates are not rolled back. The optimizer must own exactly the model's trainable parameters, and each must receive a gradient. The epoch function consumes batches without loading data, fitting preprocessing, reseeding, or writing artifacts. Checkpoint selection and the training CLI remain separate upcoming components.

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
  training/      Configuration, runtime setup, optimizer construction, and epochs
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

Reproducibility is recorded in the lockfile and artifact manifests: source revisions, SHA-256 checksums, transformation versions, split assignments, seeds, code hashes, and environment versions. Artifact writes are staged. Repeated runs verify existing outputs and report mismatches without silently overwriting them.

## Development checks

Install both extras to run the complete test suite and type checks:

```bash
uv sync --locked --dev --extra data --extra train
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
uv run --no-sync pytest
```

Tests marked `training`, including the memorization check, are excluded by default. Run them explicitly with `uv run --no-sync pytest -m training`, or run the complete suite with `uv run --no-sync pytest -m ""`.

[CI](.github/workflows/ci.yml) runs the standard suite and training checks as separate steps against a non-editable package installation. Tests use synthetic fixtures without downloading the dataset. Local Git hooks are available with `uv run --no-sync pre-commit install`.

## Scope and limitations

The corpus is small and was sampled partly through forward-looking keywords, so it does not establish performance across complete filings. Exact normalized-text separation does not control near duplicates or overlap by company or period. Dataset licensing and differences between published corpus descriptions remain unresolved; the [dataset documentation](data/README.md#limitations-and-unresolved-questions) records the reviewed evidence.
