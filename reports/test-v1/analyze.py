"""Reproduce post-test diagnostics from saved artifacts; never load or fit models.

Run from any directory with the project's data/train extras installed. Derived
JSON and the figure go to --output-dir; sampled source text stays under artifacts/.
The human review is documented separately in analysis.md.
"""

import argparse
import hashlib
import json
import platform
import statistics
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from filing_sentence_classifier.evaluation.metrics import classification_metrics
from filing_sentence_classifier.text.encoding import TextEncoder

NEURAL = "mean-pool-mlp-regularized-v1"
TFIDF = "tfidf-bigram-c10-v1"
SALT = "test-errors-v1|2026|"


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def plot_confusions(models, destination):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), layout="constrained")
    for ax, name, title in zip(
        axes,
        (TFIDF, NEURAL),
        ("TF-IDF + logistic regression", "MeanPoolMLP · seed 17"),
        strict=True,
    ):
        counts = np.array(models[name]["metrics"]["confusion_matrix"])
        percentages = counts / counts.sum(axis=1, keepdims=True) * 100
        heatmap = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
        for row in range(3):
            for column in range(3):
                ax.text(
                    column,
                    row,
                    f"{counts[row, column]}\n{percentages[row, column]:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white" if percentages[row, column] > 55 else "#172b4d",
                )
        ax.set(
            xticks=range(3),
            yticks=range(3),
            xticklabels=("Specific FLS", "Not-FLS", "Non-specific FLS"),
            yticklabels=("Specific FLS", "Not-FLS", "Non-specific FLS"),
            xlabel="Predicted label",
            ylabel="Published label",
            title=title,
        )
        ax.tick_params(axis="both", length=0, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        "Published test · 1,000 sentences\nCounts and percentage of each true class",
        fontsize=13,
    )
    fig.colorbar(heatmap, ax=axes, shrink=0.8, label="% of true class")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def analyze(root, output):
    inputs = {}

    def read(path, expected=None):
        path = root / path
        content = path.read_bytes()
        actual = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
        if expected is not None and actual != {key: expected[key] for key in actual}:
            raise ValueError(f"Artifact integrity mismatch: {path}")
        inputs[path.relative_to(root).as_posix()] = actual
        return content

    def load(path, expected=None):
        return json.loads(read(path, expected))

    results = load("reports/test-v1/results.json")
    execution = load(results["execution"]["path"], results["execution"])
    assert execution["status"] == "completed"
    base = Path(results["execution"]["path"]).parent
    selection_bytes = read(
        base / "selection.json", execution["files"]["selection.json"]
    )
    assert hashlib.sha256(selection_bytes).hexdigest() == results["selection_sha256"]
    selection = json.loads(selection_bytes)
    freeze = load(selection["freeze"]["path"], selection["freeze"])
    majority_run = load(
        selection["majority_manifest"]["path"], selection["majority_manifest"]
    )
    majority_validation = load(
        "reports/majority-v1/metrics.val.json",
        majority_run["files"]["metrics.val.json"],
    )["metrics"]
    read(base / "data/manifest.json", results["data_manifest"])
    data_bytes = read(base / "data/test.jsonl", execution["files"]["data/test.jsonl"])
    records = [json.loads(line) for line in data_bytes.splitlines()]
    ids = [row["sample_id"] for row in records]
    targets = [row["label"] for row in records]
    assert len(set(ids)) == len(records) == selection["expected_rows"] == 1000
    assert [row["source_row"] for row in records] == list(range(len(records)))
    models, predicted, scores = {}, {}, {}
    for entry in results["models"]:
        name = entry["model_id"]
        artifacts = entry["artifacts"]
        saved = load(artifacts["metrics"]["path"], artifacts["metrics"])
        assert saved == {
            key: value for key, value in entry.items() if key != "artifacts"
        }
        rows = [
            json.loads(line)
            for line in read(
                artifacts["predictions"]["path"], artifacts["predictions"]
            ).splitlines()
        ]
        assert [row["sample_id"] for row in rows] == ids
        predicted[name] = [row["predicted_label"] for row in rows]
        actual = asdict(
            classification_metrics(targets, predicted[name], label_ids=(0, 1, 2))
        )
        assert json.loads(json.dumps(actual)) == saved["metrics"]
        models[name] = saved
        if "scores" in artifacts:
            rows = [
                json.loads(line)
                for line in read(
                    artifacts["scores"]["path"], artifacts["scores"]
                ).splitlines()
            ]
            assert [row["sample_id"] for row in rows] == ids
            assert [row["label_id"] for row in rows] == predicted[name]
            scores[name] = rows
    neural = sorted(
        (model for model in models.values() if model["family"] == "mean_pool_mlp"),
        key=lambda model: model["seed"],
    )
    assert [model["seed"] for model in neural] == [17, 29, 43]
    assert len(models) == 5 and results["delivery_run_id"] == NEURAL

    # Pin the original encoder and the code used for these diagnostics.
    encoder_ref = freeze["neural_model"]["artifacts"]["encoder.json"]
    encoder = TextEncoder.from_dict(load(encoder_ref["path"], encoder_ref))
    for name in (
        "evaluation/metrics.py",
        "text/encoding.py",
        "text/tokenization.py",
        "text/vocabulary.py",
    ):
        content = read(Path("src/filing_sentence_classifier") / name)
        assert (
            hashlib.sha256(content).hexdigest()
            == execution["source"]["code_sha256"][name]
        )
    encoded = [encoder.encode(row["text"]) for row in records]
    assert [item.truncated for item in encoded] == [
        row["truncated"] for row in scores[NEURAL]
    ]
    aggregate = {
        "n": 3,
        "seeds": [17, 29, 43],
        "standard_deviation_ddof": 1,
        "interpretation": "Training-seed variation on one fixed test split, not a confidence interval or significance test. No seed reselection or ensemble.",
        "metrics": {
            key: summarize([model["metrics"][key] for model in neural])
            for key in ("macro_f1", "accuracy")
        },
        "per_class": [
            {
                "label": label,
                **{
                    key: summarize(
                        [model["metrics"]["per_class"][label][key] for model in neural]
                    )
                    for key in ("precision", "recall", "f1")
                },
            }
            for label in range(3)
        ],
    }
    mlp, tfidf = predicted[NEURAL], predicted[TFIDF]
    paired = Counter()
    for target, a, b in zip(targets, mlp, tfidf, strict=True):
        paired[
            "both_correct"
            if a == b == target
            else "only_mlp_correct"
            if a == target
            else "only_tfidf_correct"
            if b == target
            else "both_wrong"
        ] += 1
        if a != target and b != target:
            paired[
                "both_wrong_same_prediction"
                if a == b
                else "both_wrong_different_predictions"
            ] += 1

    def diagnostics(indices):
        items = [encoded[index] for index in indices]
        return {
            "rows": len(items),
            "median_original_tokens": statistics.median(
                item.original_length for item in items
            ),
            "retained_tokens": sum(len(item.input_ids) for item in items),
            "retained_unknown_tokens": sum(item.unknown_count for item in items),
            "retained_unknown_rate": sum(item.unknown_count for item in items)
            / sum(len(item.input_ids) for item in items),
            "truncated_rows": sum(item.truncated for item in items),
            "rows_without_unknown_tokens": sum(
                item.unknown_count == 0 for item in items
            ),
        }

    # Fix IDs before viewing text: two MLP errors per direction, plus three errors
    # unique to TF-IDF. The extra stratum is disjoint from all MLP-error strata.
    review = []
    for gold in range(3):
        for label in range(3):
            if gold == label:
                continue
            candidates = [
                i for i in range(len(records)) if targets[i] == gold and mlp[i] == label
            ]
            review.extend(
                (i, f"mlp_{gold}_to_{label}")
                for i in sorted(
                    candidates,
                    key=lambda i: hashlib.sha256((SALT + ids[i]).encode()).hexdigest(),
                )[:2]
            )
    candidates = [
        i
        for i in range(len(records))
        if mlp[i] == targets[i] and tfidf[i] != targets[i]
    ]
    review.extend(
        (i, "tfidf_only_error")
        for i in sorted(
            candidates,
            key=lambda i: hashlib.sha256((SALT + ids[i]).encode()).hexdigest(),
        )[:3]
    )
    assert len(review) == len({i for i, _ in review}) == 15
    cases = [
        {
            "review_id": f"T{position:02d}",
            "stratum": stratum,
            "sample_id": ids[i],
            "source_row": records[i]["source_row"],
            "gold": targets[i],
            "predictions": {name: labels[i] for name, labels in predicted.items()},
            "original_tokens": encoded[i].original_length,
            "unknown_tokens": encoded[i].unknown_count,
            "truncated": encoded[i].truncated,
        }
        for position, (i, stratum) in enumerate(review, 1)
    ]

    delivery = load("reports/inference-v1/delivery.json")
    costs = {}
    for ref in delivery["retained_benchmark_reports"]:
        benchmark = load(ref["path"], ref)
        name = benchmark["bundle"]["model_id"]
        bundle_ref = next(
            item["bundle_manifest"]
            for item in selection["models"]
            if item["run_id"] == name
        )
        assert bundle_ref["sha256"] == benchmark["bundle"]["manifest_sha256"]
        costs[name] = {
            "bundle_bytes": benchmark["bundle"]["total_bytes"],
            "summary": benchmark["summary"],
            "hardware": benchmark["hardware"],
            "workload": benchmark["workload"],
            "config": benchmark["config"],
        }
    assert costs[NEURAL]["workload"] == costs[TFIDF]["workload"]
    analysis = {
        "schema_version": 1,
        "stage": "post_test_analysis",
        "delivery_run_id": NEURAL,
        "neural_seed_summary": aggregate,
        "paired_outcomes_mlp17_vs_tfidf": dict(paired),
        "neural_seed_unanimous_predictions": sum(
            len({predicted[m["model_id"]][i] for m in neural}) == 1
            for i in range(len(records))
        ),
        "models": {
            name: {"seed": model["seed"], "metrics": model["metrics"]}
            for name, model in models.items()
        },
        "development_comparison": {
            "majority": {
                key: majority_validation[key] for key in ("macro_f1", "accuracy")
            },
            "neural_seed_summary": freeze["seed_summary"]["metrics"],
            "tfidf": freeze["classical_reference"]["validation_metrics"],
            "neural_delivery": freeze["neural_model"]["validation_metrics"],
        },
        "neural_encoding_diagnostics": {
            group: diagnostics(
                [i for i in range(len(records)) if (mlp[i] == targets[i]) == correct]
            )
            for group, correct in (("correct", True), ("incorrect", False))
        },
        "error_boundaries": {
            name: {
                "fls_vs_not_fls": sum(
                    (gold == 1) != (label == 1)
                    for gold, label in zip(targets, predicted[name], strict=True)
                ),
                "specific_vs_non_specific": sum(
                    {gold, label} == {0, 2}
                    for gold, label in zip(targets, predicted[name], strict=True)
                ),
            }
            for name in (NEURAL, TFIDF)
        },
        "review_selection": {
            "salt": SALT,
            "sort": "ascending SHA-256(salt + sample_id), UTF-8",
            "rule": "First two MLP-17 errors per true-to-predicted direction; first three TF-IDF-only errors.",
            "selection_before_manual_text_review": True,
            "population_estimate": False,
            "cases": cases,
        },
        "retained_inference_costs": costs,
        "inputs": inputs,
        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "environment": {
            name: version(name)
            for name in (
                "filing-sentence-classifier",
                "scikit-learn",
                "numpy",
                "matplotlib",
            )
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    plot_confusions(models, output / "confusion-matrices.png")
    local = root / "artifacts/analysis/test-v1"
    local.mkdir(parents=True, exist_ok=True)
    (local / "review-sentences.json").write_text(
        json.dumps(
            [
                {**case, "text": records[i]["text"]}
                for case, (i, _) in zip(cases, review, strict=True)
            ],
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"Verified five saved evaluations. Analysis written to {output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent
    )
    args = parser.parse_args()
    analyze(args.project_dir.resolve(), args.output_dir.resolve())
