"""Benchmark accounting without timing thresholds or model dependencies."""

from types import SimpleNamespace

import pytest

from filing_sentence_classifier.inference import benchmark
from filing_sentence_classifier.inference.benchmark import (
    BenchmarkConfig,
    BenchmarkError,
    latency_summary,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_sizes": ()},
        {"batch_sizes": (1, 1)},
        {"batch_sizes": (0,)},
        {"batch_sizes": (True,)},
        {"batch_sizes": [1]},
        {"trials": 0},
        {"measured_passes": False},
        {"warmup_passes": -1},
    ],
)
def test_invalid_protocol_is_rejected(kwargs):
    with pytest.raises(BenchmarkError):
        BenchmarkConfig(**kwargs)


def test_latency_summary_uses_nearest_rank_and_milliseconds():
    assert latency_summary([4_000_000, 1_000_000, 3_000_000, 2_000_000]) == {
        "samples": 4,
        "min_ms": 1.0,
        "median_ms": 2.5,
        "p95_ms": 4.0,
        "max_ms": 4.0,
    }
    assert latency_summary(list(range(1, 101)))["p95_ms"] == 95 / 1e6


@pytest.mark.parametrize("samples", [[], [0], [-1], [True], [1.5]])
def test_invalid_timings_are_not_silently_summarized(samples):
    with pytest.raises(BenchmarkError):
        latency_summary(samples)


def test_warmup_is_excluded_and_the_partial_batch_is_preserved(monkeypatch):
    calls = []

    def predict(texts, *, batch_size):
        calls.append((texts, batch_size))
        return tuple(SimpleNamespace(truncated=text == "long") for text in texts)

    # Only measured calls consume the clock, not the two warmup passes.
    times = iter([100, 110, 200, 205, 300, 320, 400, 407])
    monkeypatch.setattr(benchmark, "perf_counter_ns", lambda: next(times))
    result = benchmark._measure_batches(
        SimpleNamespace(predict=predict),
        ("one", "long", "last"),
        2,
        warmup_passes=2,
        measured_passes=2,
    )
    assert result == {
        "batch_size": 2,
        "request_sizes": [2, 1],
        "measured_passes_ns": [[10, 5], [20, 7]],
        "truncated_sentences_per_pass": 1,
    }
    assert calls == [(("one", "long"), 2), (("last",), 2)] * 4
