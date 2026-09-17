# Reproducing the completed project

This recipe rebuilds the five fixed model runs and verifies their validation and test predictions against the recorded campaign. It does not repeat hyperparameter selection or choose a new delivery seed. Commands assume a POSIX shell and a fresh repository directory; existing run directories are deliberately not overwritten.

The recorded environment uses **Python 3.13.9**, **uv 0.9.13**, and the committed `uv.lock`. Package version stays **0.1.0**, matching the frozen inference contract. Original models, data, and MLflow stores are not in Git. The [verification record](reproduction-v1.json) identifies the actual environment, source, wheel, and observed reproduction results.

## 1. Install independently and use the wheel

Start with no `.venv`. Install the locked dependencies, then the built package rather than an editable source checkout:

```bash
uv sync --locked --python 3.13.9 --dev \
  --extra data --extra train --extra tracking \
  --no-install-project --link-mode copy
uv build --wheel
uv pip install --python .venv/bin/python --no-deps \
  dist/filing_sentence_classifier-0.1.0-py3-none-any.whl
uv run --no-sync filing-sentence-classifier --version
```

All following `uv run` commands use `--no-sync` so they keep the installed wheel. `--link-mode copy` gives this environment its own dependency files; using cached distribution downloads does not borrow packages from another environment. Add `--offline` to installation/build commands when the required distributions and interpreter are already cached. An uncached setup requires network access.

## 2. Rebuild data and the selected recipes

The source is pinned by revision and SHA-256. A previously downloaded, unmodified `data/raw/` snapshot can be copied into the fresh directory; `download-data` verifies it locally. Without that snapshot, the command downloads the pinned source. No prepared data, vocabulary, or fitted models are needed to begin.

```bash
uv run --no-sync filing-sentence-classifier download-data
uv run --no-sync filing-sentence-classifier prepare-data
uv run --no-sync filing-sentence-classifier split-data

DATASET_DIR=data/processed/39b6719f1d7197df4498fea9fce20d4ad782a083/split-v1
uv run --no-sync filing-sentence-classifier build-vocabulary \
  --data-dir "$DATASET_DIR"

uv run --no-sync filing-sentence-classifier baseline majority \
  --data-dir "$DATASET_DIR" --output-dir artifacts/runs/majority-v1
uv run --no-sync filing-sentence-classifier baseline tfidf \
  --data-dir "$DATASET_DIR" \
  --config configs/experiments/tfidf-bigram-c10-v1.toml \
  --output-dir artifacts/runs/tfidf-bigram-c10-v1

for RUN_ID in mean-pool-mlp-regularized-v1 \
  mean-pool-mlp-regularized-seed29-v1 \
  mean-pool-mlp-regularized-seed43-v1; do
  uv run --no-sync filing-sentence-classifier train \
    --data-dir "$DATASET_DIR" \
    --vocabulary-dir artifacts/preprocessing/vocabulary-v1 \
    --config "configs/experiments/$RUN_ID.toml" \
    --output-dir "artifacts/runs/$RUN_ID"
  uv run --no-sync filing-sentence-classifier export \
    --run-dir "artifacts/runs/$RUN_ID" --model-id "$RUN_ID" \
    --output-dir "artifacts/bundles/$RUN_ID"
done

uv run --no-sync filing-sentence-classifier export \
  --run-dir artifacts/runs/tfidf-bigram-c10-v1 \
  --model-id tfidf-bigram-c10-v1 \
  --data-manifest "$DATASET_DIR/manifest.json" \
  --output-dir artifacts/bundles/tfidf-bigram-c10-v1
```

The vocabulary defaults to minimum frequency 2 and the neural encoder to 128 tokens. The configurations pin architecture, regularization, learning rate, batch size, CPU threads, and seeds. Neural runs choose their checkpoint only from validation. Optional local MLflow tracking is enabled by adding `--mlflow-dir artifacts/mlflow` to `train`; the reproduction above does not require a tracking server.

## 3. Verify the rebuilt results and try inference

```bash
uv run --no-sync python scripts/verify_reproduction.py \
  --output-dir artifacts/reproduction-check
uv run --no-sync filing-sentence-classifier predict \
  --bundle artifacts/bundles/mean-pool-mlp-regularized-v1 \
  --input examples/sentences.jsonl
uv run --no-sync filing-sentence-classifier predict \
  --bundle artifacts/bundles/tfidf-bigram-c10-v1 \
  --input examples/sentences.jsonl
```

The verification script first checks the original development identities, configuration bytes, neural encoders, validation prediction hashes, and each rebuilt run's export. Only then does it access the published test, using the already registered preparation policy. It compares all 1,000 prediction IDs/labels and metrics for every model against [results.json](test-v1/results.json), saves the new outputs, and records probability-byte and model-payload equality separately. Prediction or metric differences fail verification and remain available for diagnosis; they are not a reason to tune to the exposed test. Payload and probability byte equality are additional observations, not required across platforms.

This is a technical replay of fixed recipes on an already exposed evaluation set. It is not an additional independent test or model-selection result. The saved `manifest.json` identifies the inputs, installed code, environment, and status. The two demo sentences are synthetic, with expected labels `specific fls` and `not-fls` for both model families in the recorded environment.

For a deployment check, move a bundle outside the checkout and invoke the installed command from another directory using absolute paths. A bundle needs its own files and compatible runtime dependencies, not the original training tree or network access. The test suite automates this check for both families.

## 4. Run the software checks

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
FSC_REQUIRE_WHEEL=1 uv run --no-sync pytest -m ""
```

All tests use synthetic fixtures; none downloads FLS. The `FSC_REQUIRE_WHEEL` check prevents delivery tests from silently importing the source checkout. The [CI workflow](../.github/workflows/ci.yml) performs the same wheel installation and separates ordinary and iterative training tests. Local execution is evidence for the recorded machine; it is not a claim that a remote CI run was triggered.

## Reproduction boundaries

**Data and predictions:** with the recorded runtime, the preparation manifests, assignments, vocabulary, and validation/test prediction vectors are checked against the originals. Cross-platform numerical equality is not promised. The reported seed mean and SD describe the original campaign, not additional independent samples from these technical repetitions.

**Run and bundle identity:** retraining creates new timestamps, durations, source snapshots, and run manifests. An exported bundle pins its new source-run manifest, so its manifest normally differs even when weights and predictions match. In the recorded clean rebuild, all five model payloads, validation/test predictions, and all four probability outputs matched the originals byte-for-byte. Each rebuilt artifact retains its own checksum; new run and bundle metadata is not substituted into the historical freeze.

**Exact original-artifact replay:** `evaluate-test` in the [campaign recipe](test-v1/README.md#reproduce) requires the original bundles and majority manifest whose hashes are registered there. Newly trained run manifests will correctly fail that gate. Use `verify_reproduction.py` for a from-scratch reconstruction; use `evaluate-test` only when the original trusted artifacts are available. The historical selection, metrics, and benchmark reports are preserved in both cases.

**Costs and analysis:** inference timings describe a particular workload and machine, not values that should repeat exactly. The [benchmark recipe](inference-v1/README.md#reproduce) documents remeasurement. The [test analysis](test-v1/analysis.md#reproduce-the-analysis) documents reconstruction of its statistics and figure from the original campaign artifacts.
