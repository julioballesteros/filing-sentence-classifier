"""JSONL framing, bounded requests, diagnostics, and complete file publication."""

import json
from io import BytesIO, StringIO
from types import SimpleNamespace

import pytest

from filing_sentence_classifier.inference.contracts import (
    InputLimits,
    LabelSet,
    Prediction,
    prepare_texts,
)
from filing_sentence_classifier.inference.jsonl import (
    MAX_JSONL_LINE_BYTES,
    PredictionIOError,
    new_prediction_file,
    predict_jsonl,
)


@pytest.fixture
def predictor():
    limits = InputLimits(max_characters=20, max_batch_size=2)
    calls = []

    def predict(texts, *, batch_size):
        prepared = prepare_texts(texts, limits=limits)
        calls.append((prepared, batch_size))
        return tuple(
            Prediction("synthetic-v1", LabelSet(("yes", "no")), (0.25, 0.75))
            for _ in prepared
        )

    return SimpleNamespace(
        manifest=SimpleNamespace(limits=limits), predict=predict, calls=calls
    )


def test_many_rows_keep_order_and_ids_with_bounded_model_requests(predictor):
    rows = [
        {"text": "  growth\n", "sample_id": "café"},
        {"text": "growth", "sample_id": "café"},
        {"text": "unknown"},
        {"text": "!!!", "sample_id": "last"},
        {"text": "end"},
    ]
    source = BytesIO("\r\n".join(json.dumps(row) for row in rows).encode())
    destination = StringIO()
    assert predict_jsonl(predictor, source, destination, batch_size=10) == 5
    assert predictor.calls == [
        (("growth", "growth"), 10),
        (("unknown", "!!!"), 10),
        (("end",), 10),
    ]
    output = [json.loads(line) for line in destination.getvalue().splitlines()]
    assert [row.get("sample_id") for row in output] == [
        "café",
        "café",
        None,
        "last",
        None,
    ]
    assert output[0] == output[1]
    assert "text" not in output[0]
    assert output[0]["label_id"] == 1
    assert output[0]["probabilities"] == {"0": 0.25, "1": 0.75}


@pytest.mark.parametrize(
    "raw",
    [
        b"\n",
        b"not-json\n",
        b"[]\n",
        b"{}\n",
        b'{"text": 1}\n',
        b'{"text":"secret", "extra":true}\n',
        b'{"text":"secret", "text":"second"}\n',
        b'{"text":"secret", "sample_id":null}\n',
        b'{"text":"secret", "sample_id":NaN}\n',
        b'{"text":"secret", "sample_id":" "}\n',
        b'{"text":"secret", "sample_id":"\\ud800"}\n',
        b'{"text":"\xff"}\n',
        b'\xef\xbb\xbf{"text":"secret"}\n',
        json.dumps({"text": "secret", "sample_id": "x" * 257}).encode(),
        b'{"text":"secret"}' + b" " * MAX_JSONL_LINE_BYTES,
        b"[" * 2000,
    ],
)
def test_invalid_rows_report_the_line_without_echoing_text(predictor, raw):
    source = BytesIO(b'{"text":"valid"}\n' + raw)
    destination = StringIO()
    with pytest.raises(PredictionIOError, match="Line 2") as error:
        predict_jsonl(predictor, source, destination)
    assert "secret" not in str(error.value)
    assert destination.getvalue() == ""
    assert predictor.calls == []


@pytest.mark.parametrize("text", [" ", "x" * 21, "\ud800"])
def test_shared_text_limits_fail_before_the_current_batch_is_written(predictor, text):
    source = BytesIO((json.dumps({"text": text}) + "\n").encode())
    destination = StringIO()
    with pytest.raises(PredictionIOError, match="Lines 1-1"):
        predict_jsonl(predictor, source, destination)
    assert destination.getvalue() == ""
    assert predictor.calls == []


def test_empty_input_is_a_successful_empty_stream(predictor):
    destination = StringIO()
    assert predict_jsonl(predictor, BytesIO(), destination) == 0
    assert destination.getvalue() == ""
    assert predictor.calls == []


def test_late_error_keeps_completed_batches_in_stream_only(predictor):
    destination = StringIO()
    with pytest.raises(PredictionIOError, match="Line 3"):
        predict_jsonl(
            predictor,
            BytesIO(b'{"text":"one"}\n{"text":"two"}\ninvalid\n'),
            destination,
        )
    assert len(destination.getvalue().splitlines()) == 2


def test_atomic_file_publication_cleans_up_after_late_error(predictor, tmp_path):
    path = tmp_path / "predictions.jsonl"
    with pytest.raises(PredictionIOError):
        with new_prediction_file(path) as destination:
            predict_jsonl(
                predictor,
                BytesIO(b'{"text":"one"}\n{"text":"two"}\ninvalid\n'),
                destination,
            )
    assert list(tmp_path.iterdir()) == []
    with new_prediction_file(path) as destination:
        predict_jsonl(predictor, BytesIO(b'{"text":"one"}'), destination)
        assert not path.exists()
    assert len(path.read_text().splitlines()) == 1
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "concurrent"])
def test_output_never_replaces_existing_or_concurrently_created_paths(tmp_path, kind):
    path = tmp_path / "output.jsonl"
    if kind == "file":
        path.write_bytes(b"existing")
    elif kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        path.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        with new_prediction_file(path) as destination:
            destination.write("replacement")
            path.write_bytes(b"existing")
    assert list(tmp_path.iterdir()) == [path]
    if kind in {"file", "concurrent"}:
        assert path.read_bytes() == b"existing"
    elif kind == "symlink":
        assert path.is_symlink() and not path.exists()
    else:
        assert path.is_dir()
