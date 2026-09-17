# Filing Sentence Classifier

Classify forward-looking statements in English financial filings with a PyTorch model trained from scratch and classical baselines. The project covers the full path from pinned source data to a tested, portable inference bundle: data auditing, leakage controls, explicit training, model selection, held-out evaluation, and reproducibility.

**Result:** the neural delivery model reaches **0.7387 macro-F1** on the 1,000-sentence published test. Across three training seeds, its mean is **0.7253 ± 0.0121**, compared with **0.7332** for TF-IDF. The experiment does not establish a consistent neural advantage; it demonstrates a complete, traceable modeling workflow.

[Final report](reports/final-report.md) · [Model card](reports/model-card.md) · [Reproduction](reports/reproduction.md) · [Test analysis](reports/test-v1/analysis.md)

## Task and results

Inputs are individual sentences already extracted from filings; document parsing and sentence segmentation are outside scope.

| ID | Label | Meaning |
| --- | --- | --- |
| 0 | `specific fls` | A forward-looking statement about the particular company |
| 1 | `not-fls` | A statement that is not forward-looking |
| 2 | `non-specific fls` | Generic forward-looking or cautionary language |

| Frozen model | Test macro-F1 | Test accuracy |
| --- | ---: | ---: |
| Majority class | 0.2335 | 0.5390 |
| TF-IDF + logistic regression | 0.7332 | 0.7910 |
| MeanPoolMLP, seed 17 (delivery) | 0.7387 | 0.7860 |
| MeanPoolMLP, seed 29 | 0.7152 | 0.7710 |
| MeanPoolMLP, seed 43 | 0.7221 | 0.7730 |
| MeanPoolMLP, mean ± sample SD | 0.7253 ± 0.0121 | 0.7767 ± 0.0081 |

All models use the same test rows. Hyperparameters, checkpoints, and delivery seed 17 were frozen before reading test labels. The three-seed SD describes training variation on one split, not a confidence interval. The [analysis](reports/test-v1/analysis.md) includes per-class scores, confusion matrices, and reviewed errors.

## Engineering approach

- **Data:** pinned FinanceMTEB/FLS revision and file checksums; deterministic cleaning; duplicate-group separation; development/test overlap checks; row-level change and exclusion ledgers.
- **Modeling:** majority and TF-IDF references; a train-only vocabulary; learned embeddings → masked mean pooling → MLP. The neural model has 388,227 parameters and a 128-token limit.
- **Training:** explicit PyTorch loops, AdamW, cross-entropy, CPU determinism, validation macro-F1 checkpoint selection, early stopping, and optional local MLflow tracking.
- **Delivery:** self-contained, versioned bundles; verified payloads and dependencies; one Python prediction API and JSONL CLI for both model families; explicit input limits and truncation flags.
- **Verification:** unit and integration tests, synthetic memorization checks, fresh-process reproduction, installed-wheel tests, and CPU inference benchmarks.

The prepared data contains **2,074 training**, **519 validation**, and **1,000 test** sentences. Three conflicting training rows were quarantined and four development rows overlapping test were excluded. The complete test was retained. See [data provenance and preparation](data/README.md).

## Run a local demo

Requires **Python 3.13.9** and **uv 0.9.13** for the recorded environment. Commands below run from the repository root. Network access is needed for uncached dependencies and the first source download. Fitted models and the corpus are not committed; this builds your own bundle from the selected recipe.

```bash
uv sync --locked --extra data --extra train
uv run --no-sync filing-sentence-classifier download-data
uv run --no-sync filing-sentence-classifier prepare-data
uv run --no-sync filing-sentence-classifier split-data

DATASET_DIR=data/processed/39b6719f1d7197df4498fea9fce20d4ad782a083/split-v1
uv run --no-sync filing-sentence-classifier build-vocabulary \
  --data-dir "$DATASET_DIR"
uv run --no-sync filing-sentence-classifier train \
  --data-dir "$DATASET_DIR" \
  --vocabulary-dir artifacts/preprocessing/vocabulary-v1 \
  --config configs/experiments/mean-pool-mlp-regularized-v1.toml \
  --output-dir artifacts/runs/mean-pool-mlp-regularized-v1
uv run --no-sync filing-sentence-classifier export \
  --run-dir artifacts/runs/mean-pool-mlp-regularized-v1 \
  --model-id mean-pool-mlp-regularized-v1 \
  --output-dir artifacts/bundles/mean-pool-mlp-regularized-v1
uv run --no-sync filing-sentence-classifier predict \
  --bundle artifacts/bundles/mean-pool-mlp-regularized-v1 \
  --input examples/sentences.jsonl
```

The two [synthetic demo sentences](examples/sentences.jsonl) are not corpus examples:

| Sentence | Recorded prediction |
| --- | --- |
| We expect revenue to increase next year. | `specific fls` |
| Revenue increased last year. | `not-fls` |

The CLI emits the model ID, class ID/name, uncalibrated class probabilities, sample ID, and a truncation flag. It also accepts `--text "..."` or `--input -` for stdin. Existing training/output artifacts are protected; use a fresh directory for repeated training. A newly trained bundle has its own manifest even when it reproduces the original predictions.

For Python applications, load once and reuse the predictor:

```python
from pathlib import Path
from filing_sentence_classifier.inference.predictor import Predictor

predictor = Predictor.from_bundle(
    Path("artifacts/bundles/mean-pool-mlp-regularized-v1")
)
result = predictor.predict(["We expect revenue to increase next year."])[0]
print(result.to_dict())
```

## Reproduce and test

The [reproduction recipe](reports/reproduction.md) covers all five selected runs, a clean dependency installation, wheel-based execution, and comparison against the recorded test predictions. It distinguishes retraining the fixed recipes from replaying the exact original artifacts. The [verification record](reports/reproduction-v1.json) identifies the checked source, wheel, environment, and observed differences.

```bash
uv sync --locked --dev --extra data --extra train --extra tracking
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
uv run --no-sync pytest -m ""
```

[CI](.github/workflows/ci.yml) installs a built wheel and runs standard and training tests separately. Tests use synthetic fixtures without downloading FLS. `pytest` alone omits the iterative checks marked `training`. The [benchmark report](reports/inference-v1/README.md) records startup, latency, throughput, model size, hardware, and raw measurements.

## Repository structure

```text
src/filing_sentence_classifier/
  data/          Acquisition, preparation, partitions, datasets, and batching
  text/          Tokenization, vocabulary, and encoding
  baselines/     Majority and TF-IDF models and run orchestration
  models/        Neural architecture
  training/      Loops, checkpoint selection, reproducibility, and tracking
  evaluation/    Shared metrics and the frozen test campaign
  inference/     Prediction contracts, model adapters, JSONL, and benchmarks
  artifacts.py   Bundle schema and integrity verification
  exporting.py   Export of unchanged model payloads
  cli.py         Thin command-line entry points
configs/         Validated experiment configurations
examples/        Synthetic prediction inputs
notebooks/       Executed training-data audit
reports/         Results, analysis, model card, and reproduction evidence
scripts/         Verification of the fixed recipes after retraining
tests/           Unit and integration checks
```

Generated data, fitted runs, bundles, and MLflow stores stay outside Git. Their identities and scientific results are recorded in the versioned reports. Package version **0.1.0** is retained for compatibility with the frozen bundles; wheel and model checksums identify the exact artifacts.

## Scope and limitations

This is a completed **CPU modeling and local inference project**, not a deployed service. It does not ingest whole filings or support GPU training. The corpus is small, partly sampled by future-oriented keywords, and lacks company/date identifiers needed to measure those distribution shifts. Specificity, temporal scope, and annotation conventions remain difficult; probabilities are uncalibrated.

Dataset licensing and differences between published corpus descriptions remain unresolved in the reviewed source. The original corpus and fitted bundles are not distributed by this repository. Test evaluation uses one declared Unicode repair beyond the frozen inference contract. These constraints and intended uses are documented in the [model card](reports/model-card.md). A future study informed by the exposed test errors needs independent evaluation data.
