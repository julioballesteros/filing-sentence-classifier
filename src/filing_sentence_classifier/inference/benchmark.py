"""CPU benchmarks of the public Predictor in isolated, sequential processes.

The module entry point is the internal worker used by benchmark_bundle. No model
runtime is imported before the first-load timer starts in each fresh process.
"""

import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from importlib.resources import files
from importlib.resources.abc import Traversable
from io import BytesIO
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from filing_sentence_classifier.artifacts import _DEPENDENCIES, verify_bundle
from filing_sentence_classifier.inference.contracts import prepare_texts
from filing_sentence_classifier.inference.jsonl import read_prediction_rows
from filing_sentence_classifier.inference.predictor import Predictor

THREAD_ENVIRONMENT = {
    name: "1"
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
}


class BenchmarkError(ValueError):
    """Invalid benchmark settings or a failed measurement process."""


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_sizes: tuple[int, ...] = (1, 32, 128)
    warmup_passes: int = 2
    measured_passes: int = 5
    trials: int = 3

    def __post_init__(self) -> None:
        if (
            not isinstance(self.batch_sizes, tuple)
            or not self.batch_sizes
            or any(
                type(n) is not int or n < 1
                for n in (
                    *self.batch_sizes,
                    self.warmup_passes,
                    self.measured_passes,
                    self.trials,
                )
            )
            or len(set(self.batch_sizes)) != len(self.batch_sizes)
        ):
            raise BenchmarkError(
                "Use unique positive batch sizes and positive pass/trial counts."
            )


DEFAULT_BENCHMARK_CONFIG = BenchmarkConfig()


def latency_summary(samples_ns: Sequence[int]) -> dict[str, float | int]:
    """Describe measured calls; p95 uses the nearest-rank convention."""
    if not samples_ns or any(type(n) is not int or n <= 0 for n in samples_ns):
        raise BenchmarkError("Timing samples must be positive integer nanoseconds.")
    ordered = sorted(samples_ns)
    return {
        "samples": len(ordered),
        "min_ms": ordered[0] / 1e6,
        "median_ms": statistics.median(ordered) / 1e6,
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1] / 1e6,
        "max_ms": ordered[-1] / 1e6,
    }


def _measure_batches(
    predictor: Predictor,
    texts: tuple[str, ...],
    batch_size: int,
    *,
    warmup_passes: int,
    measured_passes: int,
) -> dict[str, Any]:
    batches = [
        texts[start : start + batch_size] for start in range(0, len(texts), batch_size)
    ]
    truncated = 0
    for _ in range(warmup_passes):
        truncated = sum(
            row.truncated
            for batch in batches
            for row in predictor.predict(batch, batch_size=batch_size)
        )
    passes = []
    for _ in range(measured_passes):
        timings = []
        for batch in batches:
            start = perf_counter_ns()
            predictions = predictor.predict(batch, batch_size=batch_size)
            timings.append(perf_counter_ns() - start)
            # Release results outside the timer, before measuring the next call.
            del predictions
        passes.append(timings)
    return {
        "batch_size": batch_size,
        "request_sizes": [len(batch) for batch in batches],
        "measured_passes_ns": passes,
        "truncated_sentences_per_pass": truncated,
    }


def _runtime_and_model(predictor: Predictor) -> dict[str, Any]:
    """Inspect learned state and configure only this disposable worker's runtime."""
    if predictor.manifest.family == "mean_pool_mlp":
        import torch

        from filing_sentence_classifier.inference.pytorch import PyTorchBackend

        backend = predictor._backend
        assert isinstance(backend, PyTorchBackend)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        parameters = tuple(backend._model.parameters())
        return {
            "device": "cpu",
            "dtype": "float32",
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "parameter_count": sum(p.numel() for p in parameters),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
            "vocabulary_size": len(backend._encoder.vocabulary),
        }
    from threadpoolctl import threadpool_info  # type: ignore[import-untyped]

    from filing_sentence_classifier.inference.tfidf import TfidfBackend

    tfidf = predictor._backend
    assert isinstance(tfidf, TfidfBackend)
    vectorizer, classifier = (
        tfidf._model.named_steps["tfidf"],
        tfidf._model.named_steps["logreg"],
    )
    return {
        "device": "cpu",
        "dtype": "float64",
        "native_pools": [
            {
                key: pool.get(key)
                for key in ("internal_api", "prefix", "version", "num_threads")
            }
            for pool in threadpool_info()
        ],
        "classifier_parameter_count": int(
            classifier.coef_.size + classifier.intercept_.size
        ),
        "idf_values": int(vectorizer.idf_.size),
        "learned_numeric_bytes": int(
            classifier.coef_.nbytes
            + classifier.intercept_.nbytes
            + vectorizer.idf_.nbytes
        ),
        "vocabulary_size": len(vectorizer.vocabulary_),
    }


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    start = perf_counter_ns()
    predictor = Predictor.from_bundle(
        Path(payload["bundle"]), expected_manifest_sha256=payload["manifest_sha256"]
    )
    load_ns = perf_counter_ns() - start
    runtime = _runtime_and_model(predictor)
    texts = tuple(payload["texts"])
    start = perf_counter_ns()
    first = predictor.predict(texts[:1], batch_size=1)
    first_prediction_ns = perf_counter_ns() - start
    del first
    config = BenchmarkConfig(
        **{**payload["config"], "batch_sizes": tuple(payload["config"]["batch_sizes"])}
    )
    offset = payload["trial"] % len(config.batch_sizes)
    order = config.batch_sizes[offset:] + config.batch_sizes[:offset]
    return {
        "trial": payload["trial"],
        "pid": os.getpid(),
        "first_load_ns": load_ns,
        "first_single_prediction_ns": first_prediction_ns,
        "runtime_and_model": runtime,
        "batch_order": order,
        "measurements": [
            _measure_batches(
                predictor,
                texts,
                size,
                warmup_passes=config.warmup_passes,
                measured_passes=config.measured_passes,
            )
            for size in order
        ],
    }


def _hardware() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "processor": platform.processor(),
    }
    if sys.platform == "darwin":
        for field, key in (
            ("processor", "machdep.cpu.brand_string"),
            ("physical_cpus", "hw.physicalcpu"),
            ("memory_bytes", "hw.memsize"),
        ):
            try:
                value = subprocess.check_output(
                    ["/usr/sbin/sysctl", "-n", key],
                    text=True,
                    timeout=5,
                    stderr=subprocess.DEVNULL,
                ).strip()
                result[field] = value if field == "processor" else int(value)
            except (OSError, ValueError, subprocess.SubprocessError):
                result[field] = None
    elif sys.platform.startswith("linux"):
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.is_file():
            for line in cpuinfo.read_text().splitlines():
                if line.startswith("model name"):
                    result["processor"] = line.split(":", 1)[1].strip()
                    break
    return result


def _source_hashes() -> dict[str, str]:
    hashes = {}

    def visit(node: Traversable, prefix: str) -> None:
        for item in node.iterdir():
            if item.name == "__pycache__":
                continue
            name = prefix + item.name
            if item.is_dir():
                visit(item, name + "/")
            elif item.name.endswith(".py"):
                hashes[name] = hashlib.sha256(item.read_bytes()).hexdigest()

    visit(files("filing_sentence_classifier"), "")
    return dict(sorted(hashes.items()))


def benchmark_bundle(
    bundle: Path,
    input_path: Path,
    *,
    config: BenchmarkConfig = DEFAULT_BENCHMARK_CONFIG,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Benchmark one trusted bundle against a fixed JSONL workload without labels.

    Load timings include lazy runtime imports and bundle verification, but not
    interpreter startup. OS file caches are not cleared. Prediction timers cover
    the public Python API, excluding input/output and benchmark bookkeeping.
    """
    manifest = verify_bundle(bundle, expected_manifest_sha256=expected_manifest_sha256)
    manifest_bytes = (bundle / "manifest.json").read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    # Pin the same metadata used for the measured child loads.
    if (
        expected_manifest_sha256 is not None
        and manifest_sha != expected_manifest_sha256
    ):
        raise BenchmarkError("Bundle manifest changed during verification.")
    manifest = verify_bundle(bundle, expected_manifest_sha256=manifest_sha)
    source = _source_hashes()
    content = input_path.read_bytes()
    rows = tuple(read_prediction_rows(BytesIO(content)))
    texts = tuple(row.text for row in rows)
    if not texts:
        raise BenchmarkError("Benchmark input must contain at least one sentence.")
    if max(config.batch_sizes) > min(len(texts), manifest.limits.max_batch_size):
        raise BenchmarkError(
            "Batch sizes must not exceed the workload or bundle request limit."
        )
    for start in range(0, len(texts), manifest.limits.max_batch_size):
        prepare_texts(
            texts[start : start + manifest.limits.max_batch_size],
            limits=manifest.limits,
        )
    trials = []
    started = datetime.now(UTC).isoformat()
    for trial in range(config.trials):
        payload = {
            "bundle": str(bundle.resolve()),
            "manifest_sha256": manifest_sha,
            "config": asdict(config),
            "texts": texts,
            "trial": trial,
        }
        try:
            child = subprocess.run(
                [sys.executable, "-m", __name__],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                check=False,
                timeout=600,
                env={**os.environ, **THREAD_ENVIRONMENT, "PYTHONHASHSEED": "0"},
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise BenchmarkError(f"Benchmark worker failed: {exc}") from exc
        if child.returncode:
            raise BenchmarkError(f"Benchmark worker failed: {child.stderr.strip()}")
        trials.append(json.loads(child.stdout))
    if source != _source_hashes():
        raise BenchmarkError("Package sources changed while benchmarking.")
    verify_bundle(bundle, expected_manifest_sha256=manifest_sha)
    summaries = []
    for size in config.batch_sizes:
        measured = [
            next(row for row in trial["measurements"] if row["batch_size"] == size)
            for trial in trials
        ]
        full: list[int] = []
        tail: list[int] = []
        elapsed: list[int] = []
        for row in measured:
            for timings in row["measured_passes_ns"]:
                elapsed.append(sum(timings))
                for request_size, duration in zip(
                    row["request_sizes"], timings, strict=True
                ):
                    (full if request_size == size else tail).append(duration)
        summaries.append(
            {
                "batch_size": size,
                "full_request_latency": latency_summary(full),
                "tail_request_size": len(texts) % size,
                "tail_request_latency": latency_summary(tail) if tail else None,
                "sentences_per_second": len(texts) * len(elapsed) * 1e9 / sum(elapsed),
                "pass_api_time": latency_summary(elapsed),
            }
        )
    return {
        "schema_version": 1,
        "stage": "inference_benchmark",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "bundle": {
            "model_id": manifest.model_id,
            "family": manifest.family,
            "manifest_sha256": manifest_sha,
            "files": {
                **{name: entry.to_dict() for name, entry in manifest.files.items()},
                "manifest.json": {
                    "size_bytes": len(manifest_bytes),
                    "sha256": manifest_sha,
                },
            },
            "total_bytes": len(manifest_bytes)
            + sum(entry.size_bytes for entry in manifest.files.values()),
        },
        "workload": {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
            "sentences": len(texts),
            "order": "input file order; no shuffling",
            "characters": {
                "min": min(map(len, texts)),
                "median": statistics.median(map(len, texts)),
                "max": max(map(len, texts)),
            },
        },
        "config": asdict(config),
        "hardware": _hardware(),
        "environment": {
            "python": platform.python_version(),
            "packages": {
                name: version(name)
                for name in sorted(_DEPENDENCIES[manifest.family] - {"python"})
            },
            "worker_environment": {**THREAD_ENVIRONMENT, "PYTHONHASHSEED": "0"},
        },
        "code_sha256": source,
        "measurement_policy": {
            "clock": "time.perf_counter_ns",
            "device": "cpu",
            "concurrent_requests": 1,
            "load": "first Predictor.from_bundle call per fresh process, including runtime imports and integrity checks; interpreter startup excluded; OS cache uncontrolled",
            "inference": "Predictor.predict including cleaning, encoding/vectorization, batching, model, probability and result construction; JSON and file I/O excluded",
            "warmup": "complete corpus passes for each batch size, excluded from measurements",
            "throughput": "total sentences / sum of measured API call times, including tail requests",
            "p95": "nearest rank; full and tail requests reported separately",
            "batch_order": "rotated by trial index",
            "garbage_collection": "enabled",
        },
        "summary": {
            "first_load": latency_summary([trial["first_load_ns"] for trial in trials]),
            "first_single_prediction": latency_summary(
                [trial["first_single_prediction_ns"] for trial in trials]
            ),
            "batches": summaries,
        },
        "trials": trials,
    }


if __name__ == "__main__":
    try:
        print(json.dumps(_worker(json.load(sys.stdin)), allow_nan=False))
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
