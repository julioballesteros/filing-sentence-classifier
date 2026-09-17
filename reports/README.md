# Experiment reports

Measured results and selected artifacts from reproducible runs on the frozen FLS development partitions. The published test has not been evaluated.

| Run | Model | Evaluation split | Macro-F1 | Accuracy |
| --- | --- | --- | ---: | ---: |
| [majority-v1](majority-v1/README.md) | Majority class | Validation | 0.2298 | 0.5260 |
| [tfidf-logreg-v1](tfidf-selection-v1/tfidf-logreg-v1/README.md) | TF-IDF + logistic regression | Validation | 0.6375 | 0.7360 |
| [tfidf-unigram-c1-v1](tfidf-selection-v1/tfidf-unigram-c1-v1/metrics.val.json) | TF-IDF unigrams, C=1 | Validation | 0.6696 | 0.7437 |
| [tfidf-unigram-c10-v1](tfidf-selection-v1/tfidf-unigram-c10-v1/metrics.val.json) | TF-IDF unigrams, C=10 | Validation | 0.6892 | 0.7360 |
| [tfidf-bigram-c10-v1](tfidf-selection-v1/README.md) | TF-IDF unigrams/bigrams, C=10 — selected | Validation | **0.6924** | **0.7495** |
| [mean-pool-mlp-v1](mean-pool-mlp-v1/README.md) | Initial PyTorch mean-pooling MLP | Validation | 0.6919 | 0.7360 |
| [mean-pool-mlp-dropout-v1](mlp-selection-v1/README.md) | MLP, dropout 0.3 | Validation | 0.6923 | 0.7360 |
| [mean-pool-mlp-weight-decay-v1](mlp-selection-v1/README.md) | MLP, weight decay 0.1 | Validation | 0.6916 | 0.7360 |
| [mean-pool-mlp-regularized-v1](mlp-selection-v1/README.md) | MLP, dropout + weight decay — selected neural configuration | Validation | **0.7024** | 0.7437 |
| [mean-pool-mlp-small-v1](mlp-selection-v1/README.md) | MLP, 64-dimensional embeddings | Validation | 0.6880 | 0.7380 |
| [mean-pool-mlp-vocab-min1-v1](mlp-selection-v1/README.md) | MLP, expanded vocabulary | Validation | 0.6788 | 0.7360 |

TF-IDF candidate reports are grouped under `tfidf-selection-v1/`. Run directories contain fitted model metadata, metrics, manifests, and the original configuration where applicable. Complete local runs, including fitted pipelines and per-sample predictions, live under `artifacts/runs/` outside Git.

The neural report also contains epoch history, learning curves, a training summary, and fresh-process reproduction evidence. Its full encoder, checkpoint, predictions, and source snapshot remain in the local run outside Git.

All models use the same 2,074 training rows and 519 validation rows. The [TF-IDF selection study](tfidf-selection-v1/README.md) compares four configurations under a predefined rule and records the selected model's identity. Together with the six evaluated neural configurations, this uses all ten initial TF-IDF/MLP configuration slots. Reproduction runs do not consume additional slots. Validation was used for model and checkpoint selection; test has not been evaluated.

The [validation error analysis](validation-errors.md) compares both models across the full partition and reviews 30 deterministically selected MLP errors. It records semantic patterns, vocabulary and truncation diagnostics, and questions about annotation context, with [sample identities and quantitative evidence](validation-errors.json). It motivates the next experiments without changing labels or consuming configuration slots.

The [MLP selection study](mlp-selection-v1/README.md) compares all six neural configurations at seed 17, following the immutable registration. Its [selection record](mlp-selection-v1/selection.json) chooses combined regularization by unrounded macro-F1 and pins its configuration, vocabulary, encoder, checkpoint, and MLflow identity. The [comparison data](mlp-selection-v1/comparison.json) publishes full per-class metrics, numerical learning curves, paired error counts, the same 30 reviewed IDs, artifact hashes, and MLflow identities. The [execution record](mlp-selection-v1/execution.json) preserves all five new commands and successful attempts; there were no failures or retries.

The selected recipe's [seed study](mlp-selection-v1/README.md#variation-across-training-seeds) covers seeds 17, 29, and 43, reusing the first run. Validation macro-F1 averages **0.6961 ± 0.0090** and accuracy **0.7431 ± 0.0029** (sample SD, n=3). The [results](mlp-selection-v1/seeds.json) and [execution record](mlp-selection-v1/seed-execution.json) preserve all three outcomes and identities. These repetitions leave the original configuration budget unchanged.

The [final freeze](mlp-selection-v1/freeze.json) closes experimentation and model selection, retaining the predeclared neural seed 17 at epoch 8 and the previously selected TF-IDF pipeline. It pins their artifacts and the study evidence for subsequent evaluation and packaging. Full models remain in local run storage; the freeze record is versioned here. Test remains reserved.
