# Model card: MeanPoolMLP for forward-looking statements

## Identity and purpose

| Field | Value |
| --- | --- |
| Model ID | `mean-pool-mlp-regularized-v1` |
| Package / bundle schema | `filing-sentence-classifier` 0.1.0 / schema 1 |
| Task | Single-label classification of one English financial-report sentence |
| Architecture | Learned embeddings → masked mean pooling → Linear/ReLU/Dropout → Linear |
| Parameters | 388,227, CPU float32 |
| Delivery run | Seed 17, checkpoint epoch 8, selected using validation |
| Vocabulary / maximum length | 2,967 IDs including PAD/UNK / first 128 tokens |
| Reference baseline | `tfidf-bigram-c10-v1` |

Intended use is research, education, and experimentation with sentence-level classification. The model assumes sentences have already been extracted. It does not parse filings, retrieve current information, determine whether a forecast is true, or replace financial or regulatory review. Languages, genres, and deployment settings beyond the evaluated corpus have not been validated.

## Inputs and outputs

The Python API accepts an ordered sequence of raw strings. Labels are:

| ID | Label |
| --- | --- |
| 0 | `specific fls` |
| 1 | `not-fls` |
| 2 | `non-specific fls` |

Each prediction contains model ID, class ID/name, one uncalibrated probability per class, and a `truncated` flag. Argmax determines the label, with the smallest class ID on exact ties. The JSONL CLI preserves optional sample IDs, row order, and duplicate inputs.

The original bundle permits at most **10,000 raw characters per sentence** and **256 sentences per API request**. The CLI batches files separately. Blank/non-string inputs and unsupported Unicode fail explicitly. Cleaning applies the six fixed punctuation repairs, NFC, and whitespace normalization; tokenization lowercases and retains punctuation, numbers, and negation. Unknown words map to UNK. Sentences beyond 128 tokens retain their prefix and set `truncated=true`.

The final test campaign adds a documented `U+0099 → ™` adapter before the bundle's cleaning policy to retain one otherwise unsupported test row. **Direct bundle prediction still rejects raw `U+0099`.** Reported test performance therefore includes that evaluation adapter; the bundle was not silently changed.

## Training data and procedure

Source: FinanceMTEB/FLS, revision `39b6719f1d7197df4498fea9fce20d4ad782a083`. After cleaning and overlap exclusions, grouped stratification produced **2,074 training** and **519 validation** sentences. The published **1,000 test** rows were reserved until model selection was frozen. Vocabulary fitting uses train only; labels are preserved.

The [data report](../data/README.md) documents three quarantined conflicting training rows, four development/test overlaps excluded from development, all transformations, and source limitations. The described corpus covers historical English financial filings and was sampled partly through forward-looking keywords. It does not represent the natural mix of all sentences in all filings.

The selected configuration uses 128-dimensional embeddings, a 64-unit hidden layer, dropout 0.3, AdamW, learning rate 0.001, weight decay 0.1, batch size 32, and unweighted cross-entropy. Training runs on CPU with one computation thread and deterministic algorithms. Validation macro-F1 selects the earliest best checkpoint; patience is five epochs and the limit is 30. There is no train+validation refit. The seed-17 delivery choice predates test, and seeds 29 and 43 repeat the same selected recipe.

## Evaluation

| Result | Macro-F1 | Accuracy |
| --- | ---: | ---: |
| Seed-17 validation | 0.7024 | 0.7437 |
| Seed-17 published test | 0.7387 | 0.7860 |
| Three-seed test mean ± sample SD | 0.7253 ± 0.0121 | 0.7767 ± 0.0081 |
| TF-IDF published test | 0.7332 | 0.7910 |

| Test class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Specific FLS | 169 | 0.6135 | 0.5917 | 0.6024 |
| Not-FLS | 539 | 0.8535 | 0.8646 | 0.8590 |
| Non-specific FLS | 292 | 0.7560 | 0.7534 | 0.7547 |

Class metrics describe the delivered seed-17 model. Macro-F1 weights the three declared classes equally, with undefined precision/recall/F1 set to zero. The seed SD is descriptive training variation on this split, not a confidence interval. The neural recipe does not demonstrate a consistent advantage over TF-IDF. [Full evaluation and errors](test-v1/analysis.md) and [machine-readable results](test-v1/results.json) provide the evidence.

## Limitations

- Mean pooling discards word order. Temporal scope, negation/conditionals, and company-specific versus generic cautionary wording are recurring difficulties.
- Specific-FLS recall is 0.5917 and precision 0.6135; substantial false negatives and false positives remain.
- Vocabulary coverage and prefix truncation limit available information. These alone do not explain the observed errors: all three truncated test sentences are correct for seed 17, and 55 errors have no unknown tokens.
- Company and date identifiers are absent, so performance on unseen companies or periods is not established. Exact normalized overlap checks do not eliminate paraphrases or all semantic overlap.
- Published annotation procedures and differences between corpus descriptions are incompletely documented. Review flags are not independently adjudicated label corrections.
- Scores are uncalibrated. No abstention threshold, out-of-distribution detector, or operational monitoring has been validated.
- Inference costs were measured on one CPU machine and validation workload; service overhead, concurrency, memory use, and other hardware are not covered.

The test has now been used for error analysis. New development informed by these examples needs independent evaluation data for a new final generalization claim.

## Artifact integrity and runtime

The **original** delivered bundle manifest has SHA-256:

```text
77ced5004162a6a85f173ca1ada2649fed61891e92e20d08e896fac6c6596f2b
```

Its checkpoint payload has SHA-256:

```text
564282e17ac5eddfa1ce904826730a7f8a48315148782614b39bfba94bd8364f
```

The original TF-IDF bundle manifest has SHA-256:

```text
6c6be38bd1a9e28fcf6e3fabbd0f514a283bad2555242a300cafccb4a357a962
```

These identities are fixed by the [selection freeze](mlp-selection-v1/freeze.json) and [test registration](test-v1/selection.json). Newly trained runs have separate metadata and normally different bundle-manifest hashes, even when predictions match. Use the [reproduction recipe](reproduction.md) and its verification tool rather than substituting new bytes into the original registration.

The original neural runtime records Python 3.13.9, PyTorch 2.14.0, and package 0.1.0. Loading requires matching Python major/minor and recorded runtime releases; PyTorch local build suffixes are permitted. TF-IDF requires its recorded scikit-learn, NumPy, SciPy, joblib, and threadpoolctl versions. `uv.lock` captures the tested dependency set. Load bundles only from trusted sources: hashes check integrity, and TF-IDF's joblib format can execute code when deserialized.

The bundle works offline after loading and is independent of the training tree. The shared API and CLI are documented in the [project README](../README.md). The repository does not distribute fitted bundles or corpus text. No dataset license was declared in the reviewed pinned source; reuse and redistribution conditions remain unresolved. The repository also does not assign an open-source license to the project code.
