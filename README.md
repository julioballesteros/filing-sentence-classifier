# Filing Sentence Classifier

Sentence-level classification of forward-looking statements in English financial filings. The project combines reproducible data preparation with shared evaluation, building toward a comparison of classical baselines and a PyTorch model trained from scratch.

**Work in progress:** the installable Python package, data pipeline, evaluation API/CLI, both classical baselines, and CI are implemented. PyTorch training is planned.

## Classification task

Inputs are individual sentences already extracted from a filing. The dataset defines three classes:

| ID | Label | Meaning |
| --- | --- | --- |
| 0 | `specific fls` | A forward-looking statement about the particular company. |
| 1 | `not-fls` | A statement that is not forward-looking. |
| 2 | `non-specific fls` | A generic forward-looking statement that could apply to any company. |

The implemented baselines are a majority-class classifier and TF-IDF with logistic regression. The planned PyTorch classifier uses learned embeddings, masked mean pooling, and an MLP. All models use the same saved development partitions and evaluation functions.

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
  --data-dir "$DATASET_DIR" --config configs/experiments/tfidf-logreg-v1.toml \
  --output-dir artifacts/runs/tfidf-logreg-v1
```

The [initial configuration](configs/experiments/tfidf-logreg-v1.toml) uses word unigrams/bigrams, `min_df=2`, sublinear TF, and L2-regularized multinomial logistic regression (`C=1`, L-BFGS). Vocabulary, IDF, and classifier coefficients are fitted together on train in a scikit-learn `Pipeline`. Validation uses only `transform`/`predict`. The solver runs with one native computation thread; nonconvergence stops publication.

| Model | Evaluation split | Macro-F1 | Accuracy |
| --- | --- | ---: | ---: |
| Majority class | Validation (519 sentences) | 0.2298 | 0.5260 |
| TF-IDF + logistic regression | Validation (519 sentences) | **0.6375** | **0.7360** |

This is one fixed initial configuration, without hyperparameter search. The [TF-IDF report](reports/tfidf-logreg-v1/README.md) provides per-class results and the confusion matrix. Specific FLS remains the weakest class, with recall **0.2791**.

The run contains `config.toml`, `model.joblib` (the fitted vectorizer and classifier), `model.json` (metadata), predictions, metrics, and a manifest. [`load_tfidf_model`](src/filing_sentence_classifier/baselines/tfidf.py) checks the supplied model hash and restores the pipeline without training data. Load joblib files only from trusted runs and use the recorded dependency versions: joblib can execute code during loading. [Configuration options](configs/README.md) are recorded in each run's effective recipe.

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
  data/          Source acquisition, audit, cleaning, splitting, and loading
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

After installing the environment above:

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
uv run --no-sync pytest
```

[CI](.github/workflows/ci.yml) runs these checks against a non-editable package installation. Tests use synthetic fixtures without downloading the dataset. Local Git hooks are available with `uv run --no-sync pre-commit install`.

## Scope and limitations

The corpus is small and was sampled partly through forward-looking keywords, so it does not establish performance across complete filings. Exact normalized-text separation does not control near duplicates or overlap by company or period. Dataset licensing and differences between published corpus descriptions remain unresolved; the [dataset documentation](data/README.md#limitations-and-unresolved-questions) records the reviewed evidence.
