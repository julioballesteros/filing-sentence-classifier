# TF-IDF + logistic regression

Run `tfidf-logreg-v1` fits a scikit-learn Pipeline on the 2,074 saved training sentences and evaluates the same 519 validation sentences as the majority baseline. The published test has not been evaluated. This is the first fixed TF-IDF configuration, without hyperparameter search.

This report preserves the initial reference. A subsequent [four-configuration study](../README.md) selected unigrams/bigrams with `C=10` for future comparisons.

## Configuration and fit

Word unigrams/bigrams, `min_df=2`, lowercase, sublinear TF, smoothed IDF, and L2 normalization. No stop-word removal, feature-count cap, or class reweighting. Multinomial logistic regression uses L2 regularization (`C=1`), L-BFGS, `tol=0.0001`, and `max_iter=1000`, with one native computation thread. See the [saved configuration](config.toml) and [complete effective recipe](manifest.json).

Vocabulary and IDF use training sentences only. The fitted vocabulary contains **9,459 features**; the solver converged in **46 iterations**. The complete vectorizer and classifier are stored together for inference.

## Validation results

| Model | Macro-F1 | Accuracy |
| --- | ---: | ---: |
| Majority class | 0.2298 | 0.5260 |
| TF-IDF + logistic regression | **0.6375** | **0.7360** |

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `0: specific fls` | 0.6857 | 0.2791 | 0.3967 | 86 |
| `1: not-fls` | 0.7500 | 0.9231 | 0.8276 | 273 |
| `2: non-specific fls` | 0.7162 | 0.6625 | 0.6883 | 160 |

Confusion matrix (true classes in rows, predicted classes in columns):

| True / Predicted | Specific FLS | Not-FLS | Non-specific FLS |
| --- | ---: | ---: | ---: |
| Specific FLS | 24 | 40 | 22 |
| Not-FLS | 1 | 252 | 20 |
| Non-specific FLS | 10 | 44 | 106 |

Macro-F1 improves by 0.4077 over the majority reference on this partition. Specific FLS remains difficult: only 24 of 86 examples are correctly classified, with 40 predicted as not-FLS and 22 as generic FLS. These aggregate counts identify a weakness for subsequent error analysis; they do not establish why particular sentences failed or demonstrate generalization to unseen companies or periods.

## Reproduction and artifacts

Use the [TF-IDF command](../../../README.md#run-tf-idf--logistic-regression) with `--config configs/experiments/tfidf-logreg-v1.toml` and `--output-dir artifacts/runs/tfidf-logreg-v1` to reproduce this initial run. The data is the frozen `split-v1` artifact for source revision `39b6719f1d7197df4498fea9fce20d4ad782a083`. Repeated runs in separate processes reproduced the model, predictions, and reports byte-for-byte in the recorded environment. Loading the pipeline preserves predictions without access to training data.

- [Configuration](config.toml): original experiment settings.
- [Model metadata](model.json): class order, training counts, vocabulary size, and convergence.
- [Validation metrics](metrics.val.json): full-precision scores and evaluation provenance.
- [Run manifest](manifest.json): data identity, effective recipe, source hashes, dependency versions, and output checksums.

These files are exact copies from `artifacts/runs/tfidf-logreg-v1/`. The fitted `model.joblib` and per-sample predictions remain in that local run, outside Git. Serialization omits a process-local stop-word validation cache; vocabulary, IDF, and coefficients are preserved. Byte-level reproducibility was checked in the recorded environment and is not a guarantee across different platforms or numerical libraries. Dataset limitations are described in [data/README.md](../../../data/README.md).
