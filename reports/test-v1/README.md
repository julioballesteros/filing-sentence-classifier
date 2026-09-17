# Final test evaluation

This campaign evaluates the complete published FinanceMTEB/FLS test split with the previously frozen majority baseline, TF-IDF pipeline, and three neural checkpoints (seeds 17, 29, and 43). Every model uses the same rows and shared evaluator. There is no fitting, hyperparameter search, or seed selection; the neural delivery seed remains 17.

The [selection manifest](selection.json) was registered before reading test labels. It pins the source Parquet, the development manifest, the majority run, and the four inference bundles. The command follows the existing [delivery freeze](../mlp-selection-v1/freeze.json) and its seed evidence, checking bundle source identities and unchanged payload bytes before opening test.

## Recorded results

| Frozen model | Macro-F1 | Accuracy |
| --- | ---: | ---: |
| Majority | 0.2335 | 0.5390 |
| TF-IDF + logistic regression | 0.7332 | 0.7910 |
| MeanPoolMLP, seed 17 (delivery) | 0.7387 | 0.7860 |
| MeanPoolMLP, seed 29 | 0.7152 | 0.7710 |
| MeanPoolMLP, seed 43 | 0.7221 | 0.7730 |

[Full results](results.json) include per-class metrics and confusion matrices. The [data manifest](data-manifest.json) records 1,000 retained rows: 169 specific FLS, 539 not-FLS, and 292 non-specific FLS. There are no exact duplicates after normalization and no overlap with development. Text normalization changed 78 rows (42 whitespace repairs, 40 existing punctuation repairs, and one trademark repair; operations can coexist). Each neural run truncates the same three sentences using its frozen 128-token encoder.

The [execution record](execution.json) captures the actual source and environment. It records uncommitted implementation files explicitly and preserves their complete local source snapshot; it does not claim reproduction from the recorded Git commit alone. The [verification record](verification.json) confirms **919 passing tests**, Ruff and mypy checks, independent metric recomputation, and byte-identical prepared data, predictions, probabilities, and metrics in a second process. Both executions use the same selection; no real campaign attempt failed and no model or preparation changes followed test exposure.

## Preparation contract

All published rows and labels are retained in source order, including duplicates and conflicting annotations. Sample IDs use the same source-identity/row-index convention as development. Invalid records stop the campaign instead of changing the evaluation population.

An earlier text-only overlap check identified `U+0099` in one reserved row. The versioned `published-test-v1` adapter converts that Windows-1252 character to `U+2122` (`™`) before applying the existing clean-v1 text normalization. This exception was declared before test labels or model results were read. It applies equally to every model and is recorded in the changes ledger. The original source, development data, clean-v1 implementation, and frozen bundles remain unchanged. Direct bundle prediction still rejects raw `U+0099`; these test metrics include the explicit adapter.

Preparation also checks the final normalized test groups against both development partitions. A newly introduced overlap stops evaluation; it does not trigger exclusions or retraining.

## Reproduce

From the repository root, with the pinned source snapshot, development partitions, frozen reports, majority run, and bundles available:

```bash
uv run --locked --extra data --extra train filing-sentence-classifier evaluate-test \
  --selection reports/test-v1/selection.json \
  --selection-sha256 2b30e12bf79c7a7c6437072ffd15d0f6bc527edea9531693fee9152e532834ae \
  --output-dir artifacts/evaluations/test-v1
```

Use a new output directory, such as `artifacts/evaluations/test-v1-reproduction`, for a technical repetition of the same selection. Existing destinations are rejected. Paths in the selection are relative to `--project-dir` (default `.`); `--raw-dir` defaults to `data/raw`. In addition to the delivered seed-17 and TF-IDF bundles, seeds 29 and 43 use unchanged exports of their existing runs:

```bash
for SEED in 29 43; do
  RUN_ID="mean-pool-mlp-regularized-seed${SEED}-v1"
  uv run --locked filing-sentence-classifier export \
    --run-dir "artifacts/runs/$RUN_ID" \
    --model-id "$RUN_ID" \
    --output-dir "artifacts/bundles/$RUN_ID"
done
```

The command restores existing model bytes and needs no network or MLflow server. Rebuilding models is a separate development operation: different model bytes require a new, explicitly documented selection and cannot silently replace the registered artifacts.

## Artifacts

The local output under `artifacts/evaluations/` is excluded from Git:

- `selection.json` and `evidence/`: the exact selection and supporting manifests.
- `data/test.jsonl`, `data/changes.jsonl`, and `data/manifest.json`: all prepared rows, transformation ledger, source identity, and checksums.
- `<model_id>/predictions.test.jsonl`: exactly one class prediction per sample ID, in source order.
- `<model_id>/scores.test.jsonl`: uncalibrated class probabilities and truncation flags for TF-IDF and the neural models. The majority baseline has no learned probability output.
- `<model_id>/metrics.test.json`: macro-F1, accuracy, precision/recall/F1/support per class, confusion matrix (true rows, predicted columns), and input/prediction hashes.
- `manifest.json` and `source/`: execution status, timestamps, environment, runtime settings, complete file inventory, imported Python source snapshot, and checkout provenance.

Failures after preflight retain an explicit test-access record and any partial artifacts; only `status: completed` represents the full five-model campaign. Technical reproduction does not permit test-driven tuning. Comparative interpretation, seed aggregation, and the final model card belong to the subsequent reporting task.
