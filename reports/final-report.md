# Filing Sentence Classifier — final report

## Outcome

The project delivers a reproducible CPU workflow for classifying forward-looking statements in financial-report sentences: audited and versioned data, classical baselines, an explicitly trained PyTorch model, bounded model selection, held-out evaluation, and portable inference bundles. Package version **0.1.0** and the frozen model IDs identify the completed scope; exact wheel and bundle checksums distinguish their contents.

The neural model is competitive with the TF-IDF baseline but does **not demonstrate a consistent improvement**. Its predeclared seed-17 delivery run obtains 0.7387 test macro-F1, while the three-seed mean is 0.7253 ± 0.0121 and TF-IDF obtains 0.7332. The result supports the engineering workflow and a transparent comparison, rather than a claim that the more complex model necessarily wins.

## Data and experimental controls

The source is FinanceMTEB/FLS at revision `39b6719f1d7197df4498fea9fce20d4ad782a083`: 2,600 published training and 1,000 published test sentences, with three labels (`specific fls`, `not-fls`, and `non-specific fls`). The [data report](../data/README.md) and [executed audit](../notebooks/01_data_exploration.ipynb) record the provenance, observed issues, and transformations.

Cleaning repairs six observed C1 punctuation characters, normalizes Unicode to NFC, and collapses whitespace. It preserves case, negation, numbers, punctuation, and labels. Three conflicting training rows were quarantined. Four development rows overlapping the reserved test under the fixed normalization rule were excluded. Grouped stratification with seed 2026 then produced **2,074 training** and **519 validation** rows; exact cleaned-text groups do not cross partitions. Vocabulary and TF-IDF statistics are fitted only on development train.

Before the final campaign, test access was restricted to automated text-overlap checks. All models and seeds were frozen before test labels were loaded. Test preparation retained all 1,000 rows and original labels. An explicitly registered adapter converts the previously detected `U+0099` into `™` before clean-v1 normalization; frozen inference bundles retain the original cleaning contract. No newly normalized overlap with development was found, and no test row was excluded.

## Models and selection

| Model | Fixed recipe |
| --- | --- |
| Majority | Most frequent development-training class; predicts `not-fls` |
| TF-IDF + logistic regression | Word unigrams/bigrams, `min_df=2`, sublinear TF, smoothed IDF, L2 normalization; L-BFGS logistic regression, `C=10` |
| MeanPoolMLP | Vocabulary of 2,967 IDs including PAD/UNK; 128-dimensional learned embeddings; masked mean pooling; 64-unit ReLU layer; dropout 0.3; three output logits |

The neural model has **388,227 parameters** and keeps the first **128 tokens**. Training uses unweighted cross-entropy, AdamW with learning rate 0.001 and weight decay 0.1, batch size 32, one CPU computation thread, and no loader workers. Validation macro-F1 selects the checkpoint, retaining the earlier epoch on ties. Training allows 30 epochs with patience 5. The seed-17 delivery checkpoint is epoch 8.

The bounded study evaluated four TF-IDF and six neural configurations. It used unrounded validation macro-F1 under predeclared rules. The selected neural recipe was repeated at seeds 17, 29, and 43, without changing partitions or refitting on train+validation. Seed 17 was declared for delivery before those repetitions and before test. The [selection freeze](mlp-selection-v1/freeze.json) preserves all relevant artifact identities; the [report index](README.md) links the study history.

## Held-out results

| Model | Test macro-F1 | Test accuracy |
| --- | ---: | ---: |
| Majority | 0.2335 | 0.5390 |
| TF-IDF | 0.7332 | 0.7910 |
| MeanPoolMLP, seed 17 | 0.7387 | 0.7860 |
| MeanPoolMLP, seed 29 | 0.7152 | 0.7710 |
| MeanPoolMLP, seed 43 | 0.7221 | 0.7730 |
| MeanPoolMLP, mean ± sample SD | 0.7253 ± 0.0121 | 0.7767 ± 0.0081 |

The SD uses `ddof=1` over three separately evaluated runs. It measures conditional training variation, not a confidence interval or split uncertainty. Seed 17's 0.0055 macro-F1 margin is accompanied by five fewer correct predictions than TF-IDF. The neural mean is lower than TF-IDF, so the strongest seed must not be presented as the recipe's consistent advantage.

Specific FLS is the weakest class. Seed 17 improves its recall from TF-IDF's **0.5089** to **0.5917**, while reducing precision from **0.7227** to **0.6135**. TF-IDF performs better on the dominant not-FLS class. The two models share 134 errors, with 75 additional sentences correct only for the MLP and 80 correct only for TF-IDF.

The [detailed analysis](test-v1/analysis.md) includes per-class metrics, confusion matrices, 15 systematically sampled error cases, and vocabulary diagnostics. It identifies difficulties with temporal scope, conditionals, and the boundary between company outlooks and generic risk language. Mean pooling loses token order; these observations make clause structure a plausible limitation, not a demonstrated cause of individual mistakes. All three truncated test sentences are correct for seed 17, and 55 of its 214 errors contain no unknown tokens. No labels or model settings were changed in response to test findings.

## Inference and cost

Both model families share `Predictor.from_bundle(...).predict(...)` and a JSONL command-line interface. Bundles include the fitted preprocessing, labels, model state, runtime versions, and integrity metadata. Loading verifies these contracts; inference preserves input order and duplicates, reports truncation, and works after relocation without the training data or network access.

On the recorded Apple M3 Max with one CPU computation thread, the MLP bundle occupies **1.562 MiB** and TF-IDF **0.290 MiB**, excluding dependencies. Warm single-sentence median API latency is **0.082 ms** for the MLP and **0.916 ms** for TF-IDF. At batch 32 the MLP has higher throughput; at batch 128 TF-IDF does. These are measurements on the 519 validation texts, including preprocessing and API overhead but excluding serialization, I/O, and service concurrency. The [benchmark report](inference-v1/README.md) preserves the full protocol and raw timings; it does not claim a production latency guarantee or process-memory measurement.

## Delivery and verification

The package separates data preparation, text representation, models, training, evaluation, and inference. Configuration validation, deterministic runtime settings, immutable artifact identities, explicit failure records, and shared metrics connect those components. Tracking is optional and local. CI checks lint, formatting, types, and synthetic unit/integration tests against an installed wheel.

The [reproduction recipe](reproduction.md) installs dependencies into a fresh environment, rebuilds all five fixed runs from the pinned raw source, exports the rebuilt models, verifies validation/test predictions, and exercises the synthetic demo. Its [verification record](reproduction-v1.json) identifies the installed wheel, imported source, dependency isolation, complete test result, and observed differences. The clean rebuild reproduced all five saved model payloads and validation/test prediction files byte-for-byte, together with all four probability outputs and every test metric. New run and bundle manifests differ because they record the new execution; they do not replace the historical artifact identities.

The original selection, final predictions, seed-17 delivery choice, and benchmark evidence remain unchanged. The [model card](model-card.md) documents the supported contract and limitations. The delivered scope is local CPU modeling and inference; hosting, document extraction, GPU support, and service operations are outside this project version.

## Limits and future claims

The corpus is small and partly sampled using future-oriented keywords. Unknown company/date overlap, annotation conventions, and possible semantic overlap limit generalization claims. The reviewed source does not declare a dataset license and does not explain differences between published corpus descriptions; this repository does not redistribute the corpus or fitted bundles. Probabilities are uncalibrated, and the model does not verify the truth of a statement or make an investment recommendation.

The exposed test errors can motivate later research into richer context, order-sensitive representations, or annotation review. Any resulting development needs independent evaluation evidence for a new final claim. This project closes with the fixed comparison and its reproducible artifacts, without an additional round of test-directed tuning.
