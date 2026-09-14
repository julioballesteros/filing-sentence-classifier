# Experiment configurations

[experiments/tfidf-logreg-v1.toml](experiments/tfidf-logreg-v1.toml) defines the initial TF-IDF/logistic regression reference. Pass it to `baseline tfidf --config PATH`; paths are relative to the working directory. All fields are required, and unknown fields are rejected.

| Field | Initial value | Meaning |
| --- | --- | --- |
| `tfidf.ngram_range` | `[1, 2]` | Word unigram and bigram features |
| `tfidf.min_df` | `2` | Minimum number of training sentences containing a feature |
| `logistic_regression.c` | `1.0` | Inverse regularization strength |
| `logistic_regression.max_iter` | `1000` | Solver iteration limit; nonconvergence is an error |
| `logistic_regression.tol` | `0.0001` | Solver stopping tolerance |

The fixed recipe uses lowercase word features, tokens of at least two word characters, no stop-word list or accent stripping, sublinear TF, smoothed IDF, L2 normalization, and float64 sparse matrices. There is no feature-count cap. Logistic regression uses L-BFGS, L2 regularization (`l1_ratio=0`), an intercept, and equal row weights. This solver does not use a random seed. See the [scikit-learn solver documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

The complete effective recipe and original TOML bytes are saved with the run. For a new experiment, copy the configuration and choose a distinct output directory. Model fitting always consumes the saved training partition; configuration changes do not resplit data. The pinned dataset source remains in [data/fls.py](../src/filing_sentence_classifier/data/fls.py).

The [bounded selection study](../reports/tfidf-selection-v1/README.md) adds three recipes to the initial reference: [unigrams, C=1](experiments/tfidf-unigram-c1-v1.toml), [unigrams, C=10](experiments/tfidf-unigram-c10-v1.toml), and [unigrams/bigrams, C=10](experiments/tfidf-bigram-c10-v1.toml). Only n-gram range and regularization strength vary. Unigrams/bigrams with `C=10` is the selected TF-IDF reference; all four original configurations remain available.
