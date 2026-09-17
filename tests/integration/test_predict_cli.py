"""Public prediction CLI parity and file handling for both model families."""

import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version

import pytest
import torch
from typer.testing import CliRunner

from filing_sentence_classifier.artifacts import BundleFile, BundleManifest
from filing_sentence_classifier.baselines.tfidf import (
    TfidfConfig,
    fit_tfidf,
    save_tfidf_model,
)
from filing_sentence_classifier.cli import app
from filing_sentence_classifier.inference.contracts import InputLimits, LabelSet
from filing_sentence_classifier.inference.predictor import Predictor
from filing_sentence_classifier.models.mean_pool_mlp import MeanPoolMLP
from filing_sentence_classifier.text.encoding import TextEncoder
from filing_sentence_classifier.text.tokenization import tokenize
from filing_sentence_classifier.text.vocabulary import Vocabulary
from filing_sentence_classifier.training.checkpoints import save_checkpoint
from filing_sentence_classifier.training.config import ModelConfig, TrainingConfig

TEXTS = (
    "We expect growth",
    "We expect investment",
    "Profit increased yesterday",
    "Revenue increased yesterday",
    "Results may differ",
    "Outcomes may differ",
)


@pytest.fixture(params=["mean_pool_mlp", "tfidf_logreg"])
def bundle(tmp_path, request):
    directory = tmp_path / "bundle"
    directory.mkdir()
    family = request.param
    if family == "mean_pool_mlp":
        config = TrainingConfig(model=ModelConfig(4, 3, 0.2))
        encoder = TextEncoder(Vocabulary.fit(map(tokenize, TEXTS), min_frequency=1), 3)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = MeanPoolMLP(
                len(encoder.vocabulary), embedding_dim=4, hidden_dim=3, dropout=0.2
            )
        save_checkpoint(
            directory / "model.pt", model, config=config, epoch=1, macro_f1=0.5
        )
        (directory / "encoder.json").write_text(json.dumps(encoder.to_dict()))
        lines = ["schema_version = 2"]
        for section, values in config.to_dict().items():
            if isinstance(values, dict):
                lines += [
                    f"[{section}]",
                    *(f"{key} = {json.dumps(value)}" for key, value in values.items()),
                ]
        (directory / "config.toml").write_text("\n".join(lines) + "\n")
        dependencies = ["torch", "filing-sentence-classifier"]
    else:
        config_bytes = b"""[tfidf]
ngram_range = [1, 2]
min_df = 1
[logistic_regression]
c = 1.0
max_iter = 1000
tol = 0.0001
"""
        pipeline = fit_tfidf(
            TEXTS,
            (0, 0, 1, 1, 2, 2),
            label_ids=(0, 1, 2),
            config=TfidfConfig.from_toml(config_bytes),
        )
        save_tfidf_model(pipeline, directory / "model.joblib")
        (directory / "config.toml").write_bytes(config_bytes)
        dependencies = [
            "filing-sentence-classifier",
            "scikit-learn",
            "numpy",
            "scipy",
            "joblib",
            "threadpoolctl",
        ]
    manifest = BundleManifest(
        model_id="synthetic-cli-v1",
        family=family,
        labels=LabelSet(("specific", "historical", "generic")),
        limits=InputLimits(max_characters=80, max_batch_size=2),
        environment={
            "python": platform.python_version(),
            **{name: version(name) for name in dependencies},
        },
        source_run_id="synthetic-run",
        source_manifest_sha256="a" * 64,
        files={
            path.name: BundleFile(
                hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size
            )
            for path in directory.iterdir()
        },
    )
    (directory / "manifest.json").write_text(json.dumps(manifest.to_dict()))
    return directory


def assert_matches(actual, expected):
    assert len(actual) == len(expected)
    for row, prediction in zip(actual, expected, strict=True):
        reference = prediction.to_dict()
        assert row.pop("probabilities") == pytest.approx(
            reference.pop("probabilities"), abs=1e-7
        )
        assert row == reference


def test_text_stdout_is_the_shared_prediction_schema(bundle):
    text = "  We expect growth\nnext year. "
    result = CliRunner().invoke(
        app, ["predict", "--bundle", str(bundle), "--text", text]
    )
    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert_matches(
        [json.loads(result.stdout)], Predictor.from_bundle(bundle).predict([text])
    )
    assert json.loads(result.stdout)["truncated"] == (
        "model.pt" in {p.name for p in bundle.iterdir()}
    )


def test_file_batches_reuse_one_model_and_match_the_python_api(
    bundle, tmp_path, monkeypatch
):
    texts = (*TEXTS, "unknown", "!!!", TEXTS[0])
    source = tmp_path / "sentences.jsonl"
    source.write_text(
        "".join(
            json.dumps({"text": text, "sample_id": str(index)}) + "\n"
            for index, text in enumerate(texts)
        )
    )
    before = source.read_bytes()
    output = tmp_path / "predictions.jsonl"
    expected_predictor = Predictor.from_bundle(bundle)
    expected = tuple(
        row
        for start in range(0, len(texts), 2)
        for row in expected_predictor.predict(texts[start : start + 2], batch_size=2)
    )
    load = Predictor.from_bundle
    calls = []

    def observe(*args, **kwargs):
        calls.append(args)
        return load(*args, **kwargs)

    monkeypatch.setattr(Predictor, "from_bundle", observe)
    result = CliRunner().invoke(
        app,
        [
            "predict",
            "--bundle",
            str(bundle),
            "--input",
            str(source),
            "--output",
            str(output),
            "--batch-size",
            "10",
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == result.stderr == ""
    assert len(calls) == 1
    actual = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row.pop("sample_id") for row in actual] == list(map(str, range(len(texts))))
    assert_matches(actual, expected)
    assert source.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))


def test_stdin_and_explicit_stdout_with_pinned_manifest(bundle):
    digest = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    result = CliRunner().invoke(
        app,
        [
            "predict",
            "--bundle",
            str(bundle),
            "--input",
            "-",
            "--output",
            "-",
            "--manifest-sha256",
            digest,
        ],
        input='{"text":"We expect growth", "sample_id":"café"}\n',
    )
    assert result.exit_code == 0, result.output
    row = json.loads(result.stdout)
    assert row.pop("sample_id") == "café"
    assert_matches([row], Predictor.from_bundle(bundle).predict(["We expect growth"]))


@pytest.mark.parametrize(
    "case",
    [
        "input_missing",
        "output_exists",
        "same_file",
        "parent_missing",
        "late_invalid",
        "checksum",
        "incompatible",
        "blank_text",
    ],
)
def test_failures_leave_no_published_output_and_report_to_stderr(
    bundle, tmp_path, case
):
    source = tmp_path / "input.jsonl"
    source.write_text('{"text":"one"}\n{"text":"two"}\n')
    output = tmp_path / "output.jsonl"
    extras = []
    if case == "input_missing":
        source.unlink()
    elif case == "output_exists":
        output.write_text("keep")
    elif case == "same_file":
        output = source
    elif case == "parent_missing":
        output = tmp_path / "missing" / "output.jsonl"
    elif case == "late_invalid":
        with source.open("a") as stream:
            stream.write("secret invalid data\n")
    elif case == "checksum":
        extras = ["--manifest-sha256", "0" * 64]
    elif case == "incompatible":
        path = bundle / "manifest.json"
        state = json.loads(path.read_text())
        state["environment"]["python"] = "0.0.0"
        path.write_text(json.dumps(state))
    before = output.read_bytes() if output.exists() else None
    inputs = ["--text", " "] if case == "blank_text" else ["--input", str(source)]
    result = CliRunner().invoke(
        app,
        ["predict", "--bundle", str(bundle), *inputs, "--output", str(output), *extras],
    )
    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    assert "Error:" in result.stderr
    assert "secret" not in result.stderr
    assert "Traceback" not in result.stderr
    if before is None:
        assert not output.exists()
    else:
        assert output.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))


def test_fresh_process_runs_from_an_unrelated_directory_without_network(
    bundle, tmp_path
):
    moved = tmp_path / "delivery"
    bundle.rename(moved)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    script = """import sys
def offline(event, args):
    assert event != 'socket.connect', 'Prediction must be offline.'
sys.addaudithook(offline)
from filing_sentence_classifier.cli import app
app()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "predict",
            "--bundle",
            str(moved),
            "--input",
            "-",
        ],
        input='{"text":"We expect growth"}\n',
        capture_output=True,
        text=True,
        cwd=elsewhere,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert_matches(
        [json.loads(result.stdout)],
        Predictor.from_bundle(moved).predict(["We expect growth"]),
    )


@pytest.mark.parametrize(
    "args",
    [[], ["--text", "one", "--input", "-"], ["--text", "one", "--batch-size", "0"]],
)
def test_invalid_options_fail_before_loading_a_bundle(args):
    result = CliRunner().invoke(app, ["predict", "--bundle", "missing", *args])
    assert result.exit_code == 2
    assert result.stdout == ""


def test_prediction_help_and_import_need_no_model_runtime(tmp_path):
    script = """import builtins, sys
original = builtins.__import__
blocked = {'torch', 'sklearn', 'numpy', 'scipy', 'joblib', 'mlflow', 'matplotlib', 'pyarrow', 'huggingface_hub'}
def guarded(name, *args, **kwargs):
    assert name.split('.')[0] not in blocked, name
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from filing_sentence_classifier.cli import app
app()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, "predict", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--bundle" in result.stdout


def test_stdout_keeps_complete_batches_and_reports_late_errors(bundle):
    result = CliRunner().invoke(
        app,
        ["predict", "--bundle", str(bundle), "--input", "-"],
        input='{"text":"one"}\n{"text":"two"}\ninvalid\n',
    )
    assert result.exit_code == 1
    assert len([json.loads(line) for line in result.stdout.splitlines()]) == 2
    assert "Line 3" in result.stderr


def test_missing_runtime_is_an_actionable_cli_error(bundle, tmp_path):
    dependency, extra = (
        ("torch", "train") if (bundle / "model.pt").exists() else ("sklearn", "data")
    )
    script = f"""import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] == {dependency!r}:
        raise ModuleNotFoundError('Runtime not installed', name=name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from filing_sentence_classifier.cli import app
app()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "predict",
            "--bundle",
            str(bundle),
            "--text",
            "growth",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert f"[{extra}] extra" in result.stderr
    assert "Traceback" not in result.stderr
