"""TF-IDF configuration and recipe without importing model runtimes."""

import math
import tomllib
from dataclasses import dataclass
from typing import Self

from filing_sentence_classifier.baselines.majority import BaselineError


@dataclass(frozen=True)
class TfidfConfig:
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    c: float = 1.0
    max_iter: int = 1000
    tol: float = 0.0001

    def __post_init__(self) -> None:
        if (
            len(self.ngram_range) != 2
            or any(type(value) is not int for value in self.ngram_range)
            or not 1 <= self.ngram_range[0] <= self.ngram_range[1]
            or type(self.min_df) is not int
            or self.min_df < 1
            or type(self.max_iter) is not int
            or self.max_iter < 1
            or any(
                type(value) not in (float, int)
                or not math.isfinite(value)
                or value <= 0
                for value in (self.c, self.tol)
            )
        ):
            raise BaselineError("Invalid TF-IDF/logistic regression configuration.")

    @classmethod
    def from_toml(cls, content: bytes) -> Self:
        config = tomllib.loads(content.decode("utf-8"))
        vectorizer, classifier = config.get("tfidf"), config.get("logistic_regression")
        if (
            set(config) != {"tfidf", "logistic_regression"}
            or not isinstance(vectorizer, dict)
            or set(vectorizer) != {"ngram_range", "min_df"}
            or not isinstance(vectorizer["ngram_range"], list)
            or len(vectorizer["ngram_range"]) != 2
            or not isinstance(classifier, dict)
            or set(classifier) != {"c", "max_iter", "tol"}
        ):
            raise BaselineError("Unexpected or missing TF-IDF configuration fields.")
        return cls(
            (vectorizer["ngram_range"][0], vectorizer["ngram_range"][1]),
            vectorizer["min_df"],
            classifier["c"],
            classifier["max_iter"],
            classifier["tol"],
        )

    def recipe(self) -> dict[str, object]:
        """Record configurable values and fixed feature/optimization decisions."""
        return {
            "family": "tfidf_logreg",
            "fit_split": "train",
            "evaluation_split": "val",
            "tfidf": {
                "analyzer": "word",
                "ngram_range": list(self.ngram_range),
                "min_df": self.min_df,
                "max_df": 1.0,
                "max_features": None,
                "lowercase": True,
                "strip_accents": None,
                "stop_words": None,
                "token_pattern": r"(?u)\b\w\w+\b",
                "norm": "l2",
                "use_idf": True,
                "smooth_idf": True,
                "sublinear_tf": True,
                "binary": False,
                "dtype": "float64",
            },
            "logistic_regression": {
                "solver": "lbfgs",
                "l1_ratio": 0.0,
                "C": float(self.c),
                "class_weight": None,
                "fit_intercept": True,
                "max_iter": self.max_iter,
                "tol": float(self.tol),
                "random_state": None,
                "warm_start": False,
            },
            "thread_limit": 1,
            "convergence_warning": "error",
        }
