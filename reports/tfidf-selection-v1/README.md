# TF-IDF selection study

A bounded 2 × 2 comparison of word features and L2 regularization, using the frozen FLS development partitions. The selected reference is **`tfidf-bigram-c10-v1`**: unigrams/bigrams with `C=10`, validation **macro-F1 0.6924** and **accuracy 0.7495**.

## Comparison protocol

The [plan](plan.json) was recorded before running the three new configurations. The existing unigrams/bigrams, `C=1` reference had already been measured and was reused. The study asks whether bigrams improve over unigrams and whether weaker regularization (`C=10`) improves over `C=1`.

Only n-gram range and `C` vary. All runs use the same 2,074 training sentences, 519 validation sentences, label mapping, source code, environment, `min_df=2`, feature normalization, solver settings, and one native computation thread. Vocabulary, IDF, and coefficients are fitted only on train. All four configurations converged; there were no failed candidates or additional configurations tried.

Selection maximizes validation macro-F1 at full precision. An exact tie prefers unigrams, then the smaller `C`; this rule was fixed before the new results. The published test remains reserved. Four distinct TF-IDF configurations consume four of the initial ten TF-IDF/MLP experiment slots, leaving six. Reproduction runs do not count as new configurations.

## Results

| Word features | C | Macro-F1 | Accuracy | Specific FLS recall | Vocabulary |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Unigrams](tfidf-unigram-c1-v1/manifest.json) | 1 | 0.6696 | 0.7437 | 0.3721 | 2,751 |
| [Unigrams](tfidf-unigram-c10-v1/manifest.json) | 10 | 0.6892 | 0.7360 | 0.5349 | 2,751 |
| [Unigrams + bigrams](tfidf-logreg-v1/manifest.json) | 1 | 0.6375 | 0.7360 | 0.2791 | 9,459 |
| [Unigrams + bigrams](tfidf-bigram-c10-v1/manifest.json) | 10 | 0.6924 | 0.7495 | 0.4651 | 9,459 |

Weaker regularization improves macro-F1 in both feature settings: by 0.0196 for unigrams and 0.0548 for unigrams/bigrams. The contribution of bigrams depends on regularization: at `C=1`, unigrams perform better; at `C=10`, unigrams/bigrams lead by only 0.0031. Unigrams use 2,751 features versus 9,459 for the selected model, making them a useful smaller alternative. No speed advantage was benchmarked.

The selected model follows the declared macro-F1 rule. These are validation results used for selection; the small lead over unigrams with `C=10` is not evidence of a reliable advantage on new data. Unigrams with `C=10` also have higher specific-FLS recall (0.5349 versus 0.4651), but recall was not the selection criterion.

## Selected model

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `0: specific fls` | 0.7143 | 0.4651 | 0.5634 | 86 |
| `1: not-fls` | 0.8123 | 0.8718 | 0.8410 | 273 |
| `2: non-specific fls` | 0.6529 | 0.6937 | 0.6727 | 160 |

Confusion matrix (true classes in rows, predictions in columns):

| True / Predicted | Specific FLS | Not-FLS | Non-specific FLS |
| --- | ---: | ---: | ---: |
| Specific FLS | 40 | 20 | 26 |
| Not-FLS | 2 | 238 | 33 |
| Non-specific FLS | 14 | 35 | 111 |

Compared with the initial reference, correct specific-FLS predictions increase from 24 to 40 of 86. This class still has the lowest recall. The selected pipeline is retained as fitted on train; it has not been refitted on train+validation.

## Reproduction and evidence

From the repository root, with the prepared data and locked environment available:

```bash
DATASET_DIR=data/processed/39b6719f1d7197df4498fea9fce20d4ad782a083/split-v1
for TFIDF_RUN in tfidf-logreg-v1 tfidf-unigram-c1-v1 tfidf-unigram-c10-v1 tfidf-bigram-c10-v1; do
  uv run --locked --extra data filing-sentence-classifier baseline tfidf \
    --data-dir "$DATASET_DIR" --config "configs/experiments/$TFIDF_RUN.toml" \
    --output-dir "artifacts/runs/$TFIDF_RUN" \
    --manifest-sha256 4ee27835475f8b5395e0b67bc29b3ae56faab36d365b8e1175f28f2a876b065a
done
```

The [selection manifest](selection.json) records the full-precision ranking, selection rule, shared data identity, run hashes, and the selected model checksum. The [execution record](execution.json) contains commands, completion status, and wall times for the three new CLI runs. These timings include process startup, data loading, fitting, serialization, prediction, and evaluation; they are not training-only or inference benchmarks. The reused initial run has no new timing measurement.

All four candidates' metrics were recomputed with the common evaluator, and their data, source, environment, configuration, and output hashes were verified. Each new run was reproduced in a separate process with identical artifact bytes and preserved modification times. Numerical reproducibility across different platforms or libraries is not implied.

Selected run evidence: [configuration](tfidf-bigram-c10-v1/config.toml), [model metadata](tfidf-bigram-c10-v1/model.json), [validation metrics](tfidf-bigram-c10-v1/metrics.val.json), and [run manifest](tfidf-bigram-c10-v1/manifest.json). These are copies from `artifacts/runs/tfidf-bigram-c10-v1/`; the fitted `model.joblib` and per-sample predictions remain in that local run, outside Git. The plan identifies the unchanged training code by commit and SHA-256 hashes; the new configurations are captured in the plan and run artifacts.
