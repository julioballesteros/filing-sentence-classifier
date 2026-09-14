# Experiment reports

Measured results and selected artifacts from reproducible runs on the frozen FLS development partitions. The published test has not been evaluated.

| Run | Model | Evaluation split | Macro-F1 | Accuracy |
| --- | --- | --- | ---: | ---: |
| [majority-v1](majority-v1/README.md) | Majority class | Validation | 0.2298 | 0.5260 |
| [tfidf-logreg-v1](tfidf-selection-v1/tfidf-logreg-v1/README.md) | TF-IDF + logistic regression | Validation | 0.6375 | 0.7360 |
| [tfidf-unigram-c1-v1](tfidf-selection-v1/tfidf-unigram-c1-v1/metrics.val.json) | TF-IDF unigrams, C=1 | Validation | 0.6696 | 0.7437 |
| [tfidf-unigram-c10-v1](tfidf-selection-v1/tfidf-unigram-c10-v1/metrics.val.json) | TF-IDF unigrams, C=10 | Validation | 0.6892 | 0.7360 |
| [tfidf-bigram-c10-v1](tfidf-selection-v1/README.md) | TF-IDF unigrams/bigrams, C=10 — selected | Validation | **0.6924** | **0.7495** |

TF-IDF candidate reports are grouped under `tfidf-selection-v1/`. Run directories contain fitted model metadata, metrics, manifests, and the original configuration where applicable. Complete local runs, including fitted pipelines and per-sample predictions, live under `artifacts/runs/` outside Git.

All models use the same 2,074 training rows and 519 validation rows. The [selection study](tfidf-selection-v1/README.md) compares four TF-IDF configurations under a predefined rule and records the selected model's identity. It uses four of the initial ten TF-IDF/MLP configuration slots. Validation was used for model selection; test has not been evaluated.
