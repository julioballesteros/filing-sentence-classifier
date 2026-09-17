"""Exercise the complete final evaluation offline with synthetic model artifacts."""

import json
from dataclasses import asdict, replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from typer.testing import CliRunner

from filing_sentence_classifier import cli
from filing_sentence_classifier.baselines.run import run_majority_baseline
from filing_sentence_classifier.baselines.tfidf_run import run_tfidf_baseline
from filing_sentence_classifier.data.heldout import (
    test_preparation_recipe as preparation_recipe,
)
from filing_sentence_classifier.data.loading import DataLoadError, load_split
from filing_sentence_classifier.data.spec import DatasetSource, SourceFile
from filing_sentence_classifier.evaluation import campaign
from filing_sentence_classifier.evaluation.metrics import EvaluationError
from filing_sentence_classifier.exporting import export_bundle
from filing_sentence_classifier.models.mean_pool_mlp import MeanPoolMLP
from filing_sentence_classifier.text.encoding import TextEncoder
from filing_sentence_classifier.text.tokenization import tokenize
from filing_sentence_classifier.text.vocabulary import Vocabulary
from filing_sentence_classifier.training.artifacts import (
    environment,
    file_inventory,
    json_bytes,
    sha256,
)
from filing_sentence_classifier.training.checkpoints import save_checkpoint
from filing_sentence_classifier.training.config import TrainingConfig


def reference(path, root):
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(content),
        "size_bytes": len(content),
    }


@pytest.fixture
def final_campaign(tmp_path, development_artifact):
    """Small real serialized pipelines/checkpoints; no project data or network."""
    labels = {0: "specific", 1: "historical", 2: "generic"}
    train = load_split(development_artifact, "train")
    majority = run_majority_baseline(development_artifact, tmp_path / "majority")
    data = json.loads((majority / "manifest.json").read_bytes())["data"]
    config_path = tmp_path / "tfidf.toml"
    config_path.write_text("""[tfidf]
ngram_range = [1, 2]
min_df = 1
[logistic_regression]
c = 1.0
max_iter = 1000
tol = 0.0001
""")
    classical = run_tfidf_baseline(
        development_artifact, tmp_path / "tfidf", config_path
    )
    models = []
    seed_runs = []
    encoder = TextEncoder(
        Vocabulary.fit(map(tokenize, train.texts), min_frequency=1), 8
    )
    for seed in (17, 29, 43):
        run = tmp_path / f"mlp-{seed}"
        run.mkdir()
        config_bytes = f"""schema_version = 2
[model]
embedding_dim = 4
hidden_dim = 3
dropout = 0.2
[training]
batch_size = 32
max_epochs = 1
patience = 1
min_delta = 0.0
learning_rate = 0.001
weight_decay = 0.0
[runtime]
seed = {seed}
device = "cpu"
num_workers = 0
num_threads = 1
""".encode()
        config = TrainingConfig.from_toml(config_bytes)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = MeanPoolMLP(
                len(encoder.vocabulary), embedding_dim=4, hidden_dim=3, dropout=0.2
            )
        save_checkpoint(
            run / "checkpoints/best.pt", model, config=config, epoch=1, macro_f1=0.5
        )
        (run / "config.toml").write_bytes(config_bytes)
        (run / "encoder.json").write_bytes(json_bytes(encoder.to_dict()))
        (run / "data-manifest.json").write_bytes(
            (development_artifact / "manifest.json").read_bytes()
        )
        state = {
            "schema_version": 1,
            "stage": "neural_training_run",
            "status": "completed",
            "run_id": run.name,
            "data": data,
            "config": config.to_dict(),
            "recipe": {
                "family": "mean_pool_mlp",
                "fit_split": "train",
                "evaluation_split": "val",
                "selection_metric": "macro_f1",
                "checkpoint_tie_break": "earliest_epoch",
            },
            "model": {"num_classes": 3},
            "preprocessing": {
                "vocabulary_size": len(encoder.vocabulary),
                "max_length": 8,
            },
            "environment": environment(),
            "files": file_inventory(run),
        }
        (run / "manifest.json").write_bytes(json_bytes(state))
        seed_runs.append(
            {
                "run_id": run.name,
                "seed": seed,
                "provenance": {
                    "manifest_sha256": sha256((run / "manifest.json").read_bytes()),
                    "files": state["files"],
                },
            }
        )
    for run in [classical, *(tmp_path / row["run_id"] for row in seed_runs)]:
        bundle = export_bundle(
            run,
            tmp_path / "bundles" / run.name,
            model_id=run.name,
            data_manifest=development_artifact / "manifest.json",
        )
        models.append(
            {
                "run_id": run.name,
                "bundle_manifest": reference(bundle / "manifest.json", tmp_path),
            }
        )
    seeds = tmp_path / "seeds.json"
    seeds.write_bytes(
        json_bytes({"seeds": [17, 29, 43], "data": data, "runs": seed_runs})
    )
    freeze = tmp_path / "freeze.json"
    classical_state = json.loads((classical / "manifest.json").read_bytes())
    freeze.write_bytes(
        json_bytes(
            {
                "schema_version": 1,
                "stage": "delivery_freeze",
                "status": "frozen",
                "refit_train_and_validation": False,
                "reserved_test_evaluation": False,
                "seed_summary": {"seeds": [17, 29, 43]},
                "data": data,
                "evidence": {"seeds.json": reference(seeds, tmp_path)},
                "neural_model": {"seed": 17, "run_id": "mlp-17"},
                "classical_reference": {
                    "run_id": "tfidf",
                    "artifacts": {
                        **classical_state["files"],
                        "manifest.json": reference(
                            classical / "manifest.json", tmp_path
                        ),
                    },
                },
            }
        )
    )
    # Conflicting duplicate labels and the known encoding exception must survive.
    rows = [
        {
            "text": f"Future company {index} expects growth.",
            "label": index % 3,
            "label_text": labels[index % 3],
        }
        for index in range(35)
    ]
    rows[1]["text"] = rows[0]["text"]
    rows[2]["text"] = "ACME\u0099 expects growth."
    raw = tmp_path / "raw-test"
    revision = "f" * 40
    snapshot = raw / revision
    snapshot.mkdir(parents=True)
    test_path = snapshot / "test.parquet"
    pq.write_table(pa.Table.from_pylist(rows), test_path)
    content = test_path.read_bytes()
    test_file = SourceFile(test_path.name, len(content), sha256(content))
    dataset = DatasetSource("tests/final", revision, (test_file,))
    selection = tmp_path / "selection.json"
    selection.write_bytes(
        json_bytes(
            {
                "schema_version": 1,
                "stage": "final_test_selection",
                "preparation": preparation_recipe(),
                "freeze": reference(freeze, tmp_path),
                "models": models,
                "majority_manifest": reference(majority / "manifest.json", tmp_path),
                "development_manifest": reference(
                    development_artifact / "manifest.json", tmp_path
                ),
                "source": {
                    "repo_id": dataset.repo_id,
                    "revision": revision,
                    "split": "test",
                    "file": asdict(test_file),
                },
                "expected_rows": 35,
                "policy": {
                    "fit": False,
                    "test_driven_selection": False,
                    "delivery_seed": 17,
                    "primary_metric": "macro_f1",
                    "zero_division": 0,
                    "batch_size": 32,
                    "device": "cpu",
                    "num_threads": 1,
                },
            }
        )
    )
    return selection, dict(
        project_dir=tmp_path,
        raw_dir=raw,
        dataset=dataset,
        test_file=test_file,
        label_names=labels,
    )


def execute(final_campaign, output):
    selection, kwargs = final_campaign
    return campaign.run_test_campaign(
        selection,
        output,
        expected_selection_sha256=sha256(selection.read_bytes()),
        **kwargs,
    )


def test_complete_campaign_cli_metrics_and_reproduction(
    final_campaign, tmp_path, monkeypatch
):
    selection, kwargs = final_campaign
    monkeypatch.setattr(cli, "FLS_SOURCE", kwargs["dataset"])
    monkeypatch.setattr(cli, "FLS_TEST_FILE", kwargs["test_file"])
    monkeypatch.setattr(cli, "LABEL_NAMES", kwargs["label_names"])
    # The evaluation path must only restore fitted state, even for baselines.
    from sklearn.pipeline import Pipeline

    monkeypatch.setattr(
        Pipeline, "fit", lambda *args, **kw: pytest.fail("No refitting during test")
    )
    output = tmp_path / "evaluation"
    result = CliRunner().invoke(
        cli.app,
        [
            "evaluate-test",
            "--selection",
            str(selection),
            "--selection-sha256",
            sha256(selection.read_bytes()),
            "--output-dir",
            str(output),
            "--project-dir",
            str(tmp_path),
            "--raw-dir",
            str(kwargs["raw_dir"]),
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["status"] == "completed"
    assert manifest["test_access_started"] is True
    assert manifest["new_training_runs"] == 0
    assert manifest["delivery_run_id"] == "mlp-17"
    for name, metadata in manifest["files"].items():
        content = (output / name).read_bytes()
        assert (sha256(content), len(content)) == (
            metadata["sha256"],
            metadata["size_bytes"],
        )
    truth = [
        json.loads(line)
        for line in (output / "data/test.jsonl").read_bytes().splitlines()
    ]
    assert len(truth) == 35
    assert truth[0]["group_id"] == truth[1]["group_id"]
    assert truth[0]["label"] != truth[1]["label"]
    metrics_files = sorted(output.glob("*/metrics.test.json"))
    assert len(metrics_files) == 5
    reproduction = execute(final_campaign, tmp_path / "reproduction")
    for path in metrics_files:
        predictions = [
            json.loads(line)
            for line in path.with_name("predictions.test.jsonl")
            .read_bytes()
            .splitlines()
        ]
        assert [p["sample_id"] for p in predictions] == [
            row["sample_id"] for row in truth
        ]
        targets = [row["label"] for row in truth]
        predicted = [row["predicted_label"] for row in predictions]
        metrics = json.loads(path.read_bytes())["metrics"]
        assert metrics["accuracy"] == accuracy_score(targets, predicted)
        assert metrics["macro_f1"] == pytest.approx(
            f1_score(
                targets, predicted, labels=[0, 1, 2], average="macro", zero_division=0
            )
        )
        assert (
            metrics["confusion_matrix"]
            == confusion_matrix(targets, predicted, labels=[0, 1, 2]).tolist()
        )
        assert (
            path.read_bytes() == (reproduction / path.relative_to(output)).read_bytes()
        )
        for name in ("predictions.test.jsonl", "scores.test.jsonl"):
            if (path.parent / name).exists():
                assert (path.parent / name).read_bytes() == (
                    reproduction / path.parent.name / name
                ).read_bytes()
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(EvaluationError, match="destination exists"):
        execute(final_campaign, output)
    assert (output / "manifest.json").read_bytes() == before


@pytest.mark.parametrize(
    "tamper", ["missing_seed", "freeze", "bundle", "policy", "wrong_run"]
)
def test_invalid_selection_never_reads_test(
    final_campaign, tmp_path, monkeypatch, tamper
):
    selection, _ = final_campaign
    value = json.loads(selection.read_bytes())
    if tamper == "missing_seed":
        value["models"].pop()
    elif tamper == "freeze":
        value["freeze"]["sha256"] = "0" * 64
    elif tamper == "bundle":
        payload = tmp_path / "bundles/mlp-29/model.pt"
        payload.write_bytes(payload.read_bytes() + b"corrupt")
    elif tamper == "policy":
        value["preparation"]["before_clean_v1"] = {}
    else:
        value["models"][1]["bundle_manifest"] = value["models"][2]["bundle_manifest"]
    selection.write_bytes(json_bytes(value))
    original_read = campaign._read

    def guarded_read(path):
        assert path.suffix != ".parquet", "Preflight must not open test"
        return original_read(path)

    monkeypatch.setattr(campaign, "_read", guarded_read)
    with pytest.raises(EvaluationError):
        execute(final_campaign, tmp_path / "evaluation")
    assert not (tmp_path / "evaluation").exists()


def test_failure_after_test_access_is_recorded(final_campaign, tmp_path):
    _, kwargs = final_campaign
    source = kwargs["raw_dir"] / kwargs["dataset"].revision / kwargs["test_file"].path
    source.write_bytes(source.read_bytes() + b"corrupt")
    output = tmp_path / "failed"
    with pytest.raises(EvaluationError, match="integrity"):
        execute(final_campaign, output)
    state = json.loads((output / "manifest.json").read_bytes())
    assert state["status"] == "failed"
    assert state["test_access_started"] is True
    assert not list(output.glob("*/metrics.test.json"))


def test_general_loader_still_rejects_test(development_artifact):
    with pytest.raises(DataLoadError, match="Only saved train and val"):
        load_split(development_artifact, "test")


def test_new_overlap_stops_campaign_without_removing_test_rows(
    final_campaign, tmp_path
):
    selection, kwargs = final_campaign
    value = json.loads(selection.read_bytes())
    development = (tmp_path / value["development_manifest"]["path"]).parent
    text = load_split(development, "train").texts[0]
    source = kwargs["raw_dir"] / kwargs["dataset"].revision / kwargs["test_file"].path
    rows = pq.read_table(source).to_pylist()
    rows[0]["text"] = text
    pq.write_table(pa.Table.from_pylist(rows), source)
    content = source.read_bytes()
    file = replace(kwargs["test_file"], sha256=sha256(content), size_bytes=len(content))
    kwargs["test_file"] = file
    kwargs["dataset"] = replace(kwargs["dataset"], files=(file,))
    value["source"]["file"] = asdict(file)
    selection.write_bytes(json_bytes(value))
    output = tmp_path / "overlap"
    with pytest.raises(EvaluationError, match="overlap"):
        execute(final_campaign, output)
    state = json.loads((output / "manifest.json").read_bytes())
    assert state["status"] == "failed"
    assert state["test_access_started"] is True
    assert not list(output.glob("*/metrics.test.json"))


def test_failed_model_keeps_partial_evidence_and_restores_runtime(
    final_campaign, tmp_path, monkeypatch
):
    def fail(*args, **kwargs):
        raise RuntimeError("Synthetic inference failure")

    monkeypatch.setattr(campaign.Predictor, "predict", fail)
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    output = tmp_path / "failed-model"
    with pytest.raises(RuntimeError, match="Synthetic inference"):
        execute(final_campaign, output)
    state = json.loads((output / "manifest.json").read_bytes())
    assert state["status"] == "failed"
    assert "majority/metrics.test.json" in state["files"]
    assert (output / "data/test.jsonl").is_file()
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() is deterministic


def test_wrong_selection_checksum_fails_before_preflight(tmp_path, monkeypatch):
    selection = tmp_path / "selection.json"
    selection.write_bytes(b"{}")
    monkeypatch.setattr(
        campaign,
        "_preflight",
        lambda *args: pytest.fail("Must check selection hash first"),
    )
    with pytest.raises(EvaluationError, match="Selection checksum"):
        campaign.run_test_campaign(
            selection,
            tmp_path / "output",
            expected_selection_sha256="0" * 64,
            project_dir=tmp_path,
            raw_dir=tmp_path,
            dataset=DatasetSource("tests/invalid", "a" * 40, ()),
            test_file=SourceFile("test.parquet", 1, "0" * 64),
            label_names={},
        )
    assert not (tmp_path / "output").exists()
