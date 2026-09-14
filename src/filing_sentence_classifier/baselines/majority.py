"""A deterministic majority-class classifier with a portable JSON state."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral
from typing import Self


class BaselineError(ValueError):
    """A baseline input or saved artifact violates its contract."""


@dataclass(frozen=True)
class MajorityClassifier:
    label_ids: tuple[int, ...]
    class_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not self.label_ids
            or any(type(label) is not int or label < 0 for label in self.label_ids)
            or len(set(self.label_ids)) != len(self.label_ids)
            or len(self.class_counts) != len(self.label_ids)
            or any(type(count) is not int or count < 0 for count in self.class_counts)
            or not sum(self.class_counts)
        ):
            raise BaselineError(
                "Expected unique class IDs and nonnegative training counts."
            )

    @classmethod
    def fit(cls, targets: Iterable[int], *, label_ids: Iterable[int]) -> Self:
        """Count training rows equally; resolve ties by the smallest numeric ID."""
        labels = tuple(label_ids)
        values = tuple(targets)
        if any(
            isinstance(target, bool) or not isinstance(target, Integral)
            for target in values
        ) or any(target not in labels for target in values):
            raise BaselineError("Training targets must be declared integer class IDs.")
        counts = Counter(int(target) for target in values)
        return cls(labels, tuple(counts[label] for label in labels))

    @property
    def majority_label(self) -> int:
        return min(
            zip(self.label_ids, self.class_counts, strict=True),
            key=lambda item: (-item[1], item[0]),
        )[0]

    def predict(self, texts: Iterable[str]) -> tuple[int, ...]:
        """Return the fitted constant label for each text; no text features are used."""
        label = self.majority_label
        predictions = []
        for text in texts:
            if not isinstance(text, str):
                raise BaselineError("Prediction inputs must be strings.")
            predictions.append(label)
        return tuple(predictions)

    def to_dict(self) -> dict[str, object]:
        """Serialize fitted counts and the deterministic decision rule without pickle."""
        return {
            "schema_version": 1,
            "family": "majority",
            "tie_break": "smallest_label_id",
            "label_ids": list(self.label_ids),
            "class_counts": list(self.class_counts),
            "majority_label": self.majority_label,
        }

    @classmethod
    def from_dict(cls, state: object) -> Self:
        """Restore a fitted model, validating its schema and selected label."""
        if (
            not isinstance(state, dict)
            or set(state)
            != {
                "schema_version",
                "family",
                "tie_break",
                "label_ids",
                "class_counts",
                "majority_label",
            }
            or type(state["schema_version"]) is not int
            or state["schema_version"] != 1
            or state["family"] != "majority"
            or state["tie_break"] != "smallest_label_id"
            or not isinstance(state["label_ids"], list)
            or not isinstance(state["class_counts"], list)
        ):
            raise BaselineError("Unsupported majority model schema.")
        model = cls(tuple(state["label_ids"]), tuple(state["class_counts"]))
        if (
            type(state["majority_label"]) is not int
            or state["majority_label"] != model.majority_label
        ):
            raise BaselineError(
                "Saved majority label does not match the training counts."
            )
        return model
