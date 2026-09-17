"""Published test keeps its population and labels, with explicit text adaptation."""

import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from filing_sentence_classifier.data.heldout import prepare_published_test
from filing_sentence_classifier.data.spec import DatasetSource, SourceFile
from filing_sentence_classifier.evaluation.metrics import EvaluationError

LABELS = {0: "specific", 1: "historical", 2: "generic"}


def parquet(rows):
    stream = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows), stream)
    content = stream.getvalue().to_pybytes()
    file = SourceFile("test.parquet", len(content), hashlib.sha256(content).hexdigest())
    return content, DatasetSource("tests/heldout", "a" * 40, (file,)), file


def test_all_rows_labels_and_source_order_survive_cleaning():
    rows = [
        {
            "text": "  ACME\u0099 expects\n growth. ",
            "label": 0,
            "label_text": LABELS[0],
        },
        {"text": "ACME\u2122 expects growth.", "label": 2, "label_text": LABELS[2]},
        {"text": "ACME\u2122 expects growth.", "label": 2, "label_text": LABELS[2]},
    ]
    prepared = prepare_published_test(*parquet(rows), LABELS, expected_rows=3)
    assert prepared.targets == (0, 2, 2)
    assert prepared.texts == ("ACME\u2122 expects growth.",) * 3
    assert len(set(prepared.sample_ids)) == 3
    assert [row.source_row for row in prepared.records] == [0, 1, 2]
    assert prepared.changes[0]["operations"] == (
        "repair_c1_trademark",
        "collapse_whitespace",
    )
    assert prepared.manifest["counts"]["excluded"] == 0
    assert prepared.manifest["counts"]["duplicate_extra_rows"] == 2
    assert prepared.manifest["counts"]["conflicting_label_groups"] == 1
    assert prepared == prepare_published_test(*parquet(rows), LABELS, expected_rows=3)


@pytest.mark.parametrize(
    "change",
    [
        {"text": None},
        {"text": "  "},
        {"text": "Unresolved\u0081"},
        {"label": 9},
        {"label": True},
        {"label_text": "wrong"},
    ],
)
def test_invalid_row_stops_preparation_without_silent_filtering(change):
    row = {"text": "Valid sentence.", "label": 0, "label_text": LABELS[0], **change}
    with pytest.raises(EvaluationError, match="row 0"):
        prepare_published_test(*parquet([row]), LABELS, expected_rows=1)


def test_source_integrity_and_full_row_count_are_required():
    content, dataset, file = parquet(
        [{"text": "Valid sentence.", "label": 0, "label_text": LABELS[0]}]
    )
    with pytest.raises(EvaluationError, match="integrity"):
        prepare_published_test(content + b"x", dataset, file, LABELS, expected_rows=1)
    with pytest.raises(EvaluationError, match="row count"):
        prepare_published_test(content, dataset, file, LABELS, expected_rows=2)
