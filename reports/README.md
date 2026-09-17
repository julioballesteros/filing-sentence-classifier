# Results and reports

The project is complete within its CPU modeling and local inference scope. The published test was evaluated after model selection was frozen; all five models used the same 1,000 sentences.

| Model | Test macro-F1 | Test accuracy |
| --- | ---: | ---: |
| Majority | 0.2335 | 0.5390 |
| TF-IDF + logistic regression | 0.7332 | 0.7910 |
| MeanPoolMLP, seed 17 (delivery) | 0.7387 | 0.7860 |
| MeanPoolMLP, seed 29 | 0.7152 | 0.7710 |
| MeanPoolMLP, seed 43 | 0.7221 | 0.7730 |
| MeanPoolMLP, mean ± sample SD | 0.7253 ± 0.0121 | 0.7767 ± 0.0081 |

The neural recipe does not show a consistent advantage over TF-IDF. Seed 17 remains the delivery model chosen before test; the three-seed SD describes training variation on this split, not a confidence interval.

## Final deliverables

- [Final report](final-report.md): experimental controls, results, interpretation, costs, and limitations.
- [Model card](model-card.md): intended use, input/output contract, training, evaluation, and artifact identities.
- [Reproduction recipe](reproduction.md) and [verification record](reproduction-v1.json): independent dependency installation, wheel execution, reconstruction of all five fixed recipes, and software checks.
- [Test campaign](test-v1/README.md), [machine-readable results](test-v1/results.json), and [error analysis](test-v1/analysis.md): frozen registration, class metrics, confusion matrices, seed aggregation, and reviewed examples.
- [Inference benchmarks](inference-v1/README.md): model size, startup, warm latency, throughput, and measurement boundaries.

## Development history

These reports preserve the evidence available at each stage. References to a reserved test in older study reports describe their historical state.

| Study | Evidence |
| --- | --- |
| Data audit and preparation | [Data report](../data/README.md) |
| Majority reference | [Baseline report](majority-v1/README.md) |
| Four TF-IDF configurations | [Selection study](tfidf-selection-v1/README.md) |
| Initial neural model | [Training report](mean-pool-mlp-v1/README.md) |
| Validation errors | [Analysis](validation-errors.md) |
| Six neural configurations and three selected-recipe seeds | [Selection and seed study](mlp-selection-v1/README.md) |
| Final model identities before test | [Freeze record](mlp-selection-v1/freeze.json) |

All development comparisons use the same 2,074 training and 519 validation rows. Fitted models, full predictions, source snapshots, and MLflow stores remain in local `artifacts/` storage outside Git. Versioned reports preserve configurations, hashes, metrics, and conclusions; the reproduction recipe rebuilds models without substituting them into the original freeze.
