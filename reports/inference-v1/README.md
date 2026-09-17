# Inference cost of the frozen delivery bundles

Measured on **2026-09-17**, using an **Apple M3 Max**, 14 physical/logical CPUs, 36 GiB memory, macOS 26.6.2 arm64, and Python 3.13.9. Each worker uses **one CPU computation thread** and serves one request at a time. The GPU is not used.

The [protocol](protocol.json) pins the selected bundles and workload. Full reports retain every measured call, process identity, dependency version, source hash, and aggregation rule: [PyTorch](mean-pool-mlp.json) and [TF-IDF](tfidf.json). Each model's first completed run is retained; all three trials are included. The complete project test suite finished before measurement began.

## Size and startup

| Model | Bundle bytes | Bundle MiB | Learned state | First load, median ms | First prediction, median ms |
| --- | ---: | ---: | --- | ---: | ---: |
| MeanPoolMLP | 1,637,623 | 1.562 | 388,227 parameters, float32 | 741.165 | 0.385 |
| TF-IDF + logistic regression | 303,884 | 0.290 | 28,380 classifier parameters + 9,459 IDF values, float64 | 723.482 | 5.656 |

Bundle sizes include the manifest and all payloads, excluding the Python environment. Neural parameters occupy 1,552,908 bytes and include the saved PAD row. The classical model's coefficient, intercept, and IDF arrays occupy 302,712 bytes before serialization; its vocabulary has 9,459 features. These learned-state counts describe different structures and are not interchangeable complexity measures. No process-memory measurement is claimed.

First load is one `Predictor.from_bundle` call in each new process, including lazy model-runtime imports, integrity checks, and restoration. It excludes interpreter startup. The OS file cache is not cleared, and parent-side verification reads bundle files before the workers start; these are **fresh-process loads, not cold-disk measurements**. Load ranges were 736.214–743.316 ms for PyTorch and 720.708–730.199 ms for TF-IDF (three samples each). First prediction is a separate request containing the first workload sentence, after runtime configuration and before corpus warmup.

## Warm inference

Latency describes a complete `Predictor.predict` request with the stated number of sentences. Throughput includes every sentence, including the final partial request.

| Model | Batch size | Median request ms | p95 request ms | Sentences/s |
| --- | ---: | ---: | ---: | ---: |
| MeanPoolMLP | 1 | 0.082 | 0.115 | 11,651 |
| MeanPoolMLP | 32 | 1.237 | 1.554 | 25,578 |
| MeanPoolMLP | 128 | 5.358 | 5.735 | 24,511 |
| TF-IDF + logistic regression | 1 | 0.916 | 0.960 | 1,085 |
| TF-IDF + logistic regression | 32 | 1.747 | 2.003 | 17,637 |
| TF-IDF + logistic regression | 128 | 4.412 | 4.542 | 28,199 |

Each model uses the same **519 validation sentences**, in saved order, without targets. Sentence lengths range from 38 to 1,341 characters (median 172). PyTorch truncates three sentences under its existing 128-token policy; TF-IDF truncates none. The published test is not accessed, and neither model is fitted or changed.

The protocol uses three sequential fresh-process trials per model, two untimed full-workload warmup passes for each batch size, then five measured passes. Batch order rotates across trials: `[1, 32, 128]`, `[32, 128, 1]`, `[128, 1, 32]`. PyTorch was measured before TF-IDF. There are 7,785 full-request samples at batch 1, 240 at batch 32, and 60 at batch 128. Batches 32 and 128 each leave seven sentences at the end of a pass; their 15 tail timings are reported separately in JSON and excluded from the full-request latency table.

`time.perf_counter_ns` surrounds only the public prediction call, including validation, cleaning, encoding/vectorization, padding, model computation, probabilities, and result construction. JSON parsing, serialization, input/output, and benchmark bookkeeping are excluded. Garbage collection stays enabled. Throughput is total measured sentences divided by the sum of all measured API call times; p95 uses nearest rank. Raw timings allow both calculations to be reproduced.

## Interpretation and limits

For this workload and implementation, PyTorch has lower single-sentence latency and higher throughput at batch 32. TF-IDF has a smaller bundle and higher throughput at batch 128. Increasing the neural batch from 32 to 128 does not improve its measured throughput. Both models benefit from loading once and reusing the predictor.

These are complete API measurements, including the TF-IDF backend's per-call thread-limit context. They do not isolate matrix operations or identify a performance bottleneck. Repeated passes over a fixed small corpus, OS caching, heterogeneous CPU scheduling, background activity, and model execution order limit generalization. The samples are descriptive, correlated measurements, not independent evidence for a production latency guarantee. No network, concurrent traffic, CLI serialization, or service overhead is measured. Quality and model selection remain governed by the existing [freeze](../mlp-selection-v1/freeze.json).

## Reproduce

Use the recorded dependencies (`uv sync --locked --extra data --extra train`) and the already exported bundles. Recreate the local workload from the verified validation partition; sentence text is kept outside Git:

```python
import hashlib
import json
from pathlib import Path

from filing_sentence_classifier.data.loading import load_split

protocol = json.loads(Path("reports/inference-v1/protocol.json").read_text())
workload = protocol["workload"]
validation = load_split(
    Path(workload["source_directory"]),
    "val",
    expected_manifest_sha256=workload["source_manifest_sha256"],
)
content = "".join(
    json.dumps({"sample_id": sample_id, "text": text}, sort_keys=True) + "\n"
    for sample_id, text in zip(validation.sample_ids, validation.texts, strict=True)
).encode("utf-8")
assert hashlib.sha256(content).hexdigest() == workload["sha256"]
path = Path(workload["path"])
path.parent.mkdir(parents=True, exist_ok=True)
if path.exists():
    assert path.read_bytes() == content
else:
    with path.open("xb") as stream:
        stream.write(content)
```

Run the commands sequentially, with no training or test suite running at the same time. Use new output paths to preserve prior results:

```bash
mkdir -p artifacts/benchmarks/inference-reproduction
uv run --no-sync filing-sentence-classifier benchmark \
  --bundle artifacts/bundles/mean-pool-mlp-regularized-v1 \
  --manifest-sha256 77ced5004162a6a85f173ca1ada2649fed61891e92e20d08e896fac6c6596f2b \
  --input artifacts/benchmarks/inference-v1/validation.jsonl \
  --output artifacts/benchmarks/inference-reproduction/mean-pool-mlp.json
uv run --no-sync filing-sentence-classifier benchmark \
  --bundle artifacts/bundles/tfidf-bigram-c10-v1 \
  --manifest-sha256 6c6be38bd1a9e28fcf6e3fabbd0f514a283bad2555242a300cafccb4a357a962 \
  --input artifacts/benchmarks/inference-v1/validation.jsonl \
  --output artifacts/benchmarks/inference-reproduction/tfidf.json
```

Defaults reproduce the registered settings. `--batch-size` can be repeated; `--warmup-passes`, `--passes`, and `--trials` control the protocol. Batch sizes must fit both the workload and the bundle's request limit. The benchmark pins bundle contents across workers and refuses publication if files or package sources change. It never overwrites an existing report. Repeated runs reproduce the procedure and identities, not exact wall-clock timings.

## Delivery verification

The [delivery record](delivery.json) adds wheel installation and end-to-end verification to these measurements. Both model families pass synthetic training → export → relocated-bundle prediction checks with their original data and runs removed, networking disabled, and MLflow unavailable. Incompatible bundle versions are rejected. The installed wheel also reproduces the selected bundles' example probabilities exactly.

The record identifies the wheel, local validation environment, full-suite result, and CI configuration. Its package sources match the benchmarked code, so the existing timings are retained. The selected model files, benchmark JSON reports, and model-selection freeze remain unchanged. Test is still reserved for final evaluation.
