# Majority-class baseline

Measured results on the frozen FLS development partitions. The published test has not been evaluated. Scores below are rounded to four decimals; the linked JSON files retain full precision.

## Validation performance

Run `majority-v1` fits only the 2,074 training labels and predicts `1: not-fls` for every validation sentence. Training counts in class order `[0, 1, 2]` are `[340, 1093, 641]`. Each row contributes one vote, ties select the smallest numeric ID, and the classifier uses no text features or randomness.

| Model | Split | Sentences | Macro-F1 | Accuracy |
| --- | --- | ---: | ---: | ---: |
| Majority class | Validation | 519 | **0.2298** | **0.5260** |

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `0: specific fls` | 0.0000 | 0.0000 | 0.0000 | 86 |
| `1: not-fls` | 0.5260 | 1.0000 | 0.6894 | 273 |
| `2: non-specific fls` | 0.0000 | 0.0000 | 0.0000 | 160 |

Confusion matrix (rows are true classes; columns are predictions):

| True / Predicted | Specific FLS | Not-FLS | Non-specific FLS |
| --- | ---: | ---: | ---: |
| Specific FLS | 0 | 86 | 0 |
| Not-FLS | 0 | 273 | 0 |
| Non-specific FLS | 0 | 160 | 0 |

Accuracy reflects the validation frequency of `not-fls`. The classifier misses every forward-looking statement, giving both FLS classes zero recall and F1. This establishes the reference that a model using sentence content should improve upon. Macro-F1 includes all three classes, with undefined scores set to zero.

## Reproduction and evidence

Use the [baseline command](../reproduction.md#2-rebuild-data-and-the-selected-recipes) with the saved `split-v1` artifact for source revision `39b6719f1d7197df4498fea9fce20d4ad782a083`. The run was reproduced with identical output bytes, and the fitted model was restored in a separate process.

- [Fitted model](model.json): selected class, training counts, class order, and tie rule.
- [Validation metrics](metrics.val.json): scores, class mapping, prediction/data hashes, and evaluation provenance.
- [Run manifest](manifest.json): effective recipe, train/validation identity, code hashes, environment versions, and output checksums.

These files are exact copies of the selected local run artifacts. Manifest filenames refer to the complete run under `artifacts/runs/majority-v1/`; per-sample predictions stay there, outside Git. The [dataset documentation](../../data/README.md) describes the partitioning procedure and its limitations. Results apply to this validation partition, without a claim of generalization across companies or periods.
