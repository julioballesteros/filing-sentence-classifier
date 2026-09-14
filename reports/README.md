# Experiment reports

Measured results and selected artifacts from reproducible runs on the frozen FLS development partitions. The published test has not been evaluated.

| Run | Model | Evaluation split | Macro-F1 | Accuracy |
| --- | --- | --- | ---: | ---: |
| [majority-v1](majority-v1/README.md) | Majority class | Validation | 0.2298 | 0.5260 |
| [tfidf-logreg-v1](tfidf-logreg-v1/README.md) | TF-IDF + logistic regression | Validation | 0.6375 | 0.7360 |

Each run directory contains its summary, fitted model metadata, metrics, and manifest. Complete local runs, including per-sample predictions, live under `artifacts/runs/` outside Git.

Both references use the same 2,074 training rows and 519 validation rows. TF-IDF improves macro-F1 by 0.4077 on this partition, with the lowest recall on specific FLS. It is the initial fixed configuration; no hyperparameter search or test evaluation has been performed.
