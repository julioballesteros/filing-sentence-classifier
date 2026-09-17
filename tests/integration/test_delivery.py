"""Train, export, relocate, and predict through the installed command entry point."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from importlib.metadata import version
from pathlib import Path

import pytest
import torch

from filing_sentence_classifier.baselines.tfidf import load_tfidf_model
from filing_sentence_classifier.data.loading import load_split
from filing_sentence_classifier.models.mean_pool_mlp import MeanPoolMLP
from filing_sentence_classifier.text.encoding import TextEncoder
from filing_sentence_classifier.training.checkpoints import load_checkpoint
from filing_sentence_classifier.training.config import TrainingConfig

NEURAL_CONFIG = """schema_version = 2
[model]
embedding_dim = 8
hidden_dim = 4
dropout = 0.2
[training]
batch_size = 16
max_epochs = 2
patience = 2
min_delta = 0.0
learning_rate = 0.03
weight_decay = 0.0
[runtime]
seed = 17
device = "cpu"
num_workers = 0
num_threads = 1
"""
TFIDF_CONFIG = """[tfidf]
ngram_range = [1, 2]
min_df = 1
[logistic_regression]
c = 1.0
max_iter = 1000
tol = 0.0001
"""

# -I excludes the checkout and PYTHONPATH. CI additionally requires imports from
# this interpreter's installation directory, rejecting editable/source fallback.
LAUNCHER = """import builtins, os, sys, sysconfig
from importlib.metadata import distribution
from pathlib import Path

mode = sys.argv.pop(1)
forbidden = {'mlflow'}
if mode == 'inference':
    forbidden |= {'huggingface_hub', 'pyarrow', 'matplotlib'}
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in forbidden:
        raise ModuleNotFoundError('Unavailable in the delivery process', name=name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
def offline(event, args):
    assert event != 'socket.connect', 'Delivery checks must run offline.'
sys.addaudithook(offline)

import filing_sentence_classifier
if os.environ.get('FSC_REQUIRE_WHEEL') == '1':
    actual = Path(filing_sentence_classifier.__file__).resolve().parent
    expected = Path(sysconfig.get_path('purelib')).resolve() / 'filing_sentence_classifier'
    assert actual == expected, f'Expected an installed wheel, got {actual}'
entry = next(entry for entry in distribution('filing-sentence-classifier').entry_points
             if entry.group == 'console_scripts' and entry.name == 'filing-sentence-classifier')
sys.argv[0] = entry.name
entry.load()()
"""


def invoke(launcher, cwd, args, *, inference=False, input_text=None):
    return subprocess.run(
        [
            sys.executable,
            "-I",
            str(launcher),
            "inference" if inference else "training",
            *map(str, args),
        ],
        cwd=cwd,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=60,
        env={
            **os.environ,
            "MPLCONFIGDIR": str(cwd / "mpl-cache"),
            "PYTHONHASHSEED": "7",
        },
    )


def reference_probabilities(run, family, texts):
    """Read the actual trained state directly, independently of bundle/Predictor."""
    if family == "tfidf_logreg":
        path = run / "model.joblib"
        model = load_tfidf_model(
            path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
        return model.predict_proba(texts).tolist()
    config = TrainingConfig.from_toml((run / "config.toml").read_bytes())
    encoder = TextEncoder.from_dict(json.loads((run / "encoder.json").read_bytes()))
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        model = MeanPoolMLP(
            len(encoder.vocabulary),
            embedding_dim=config.model.embedding_dim,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
        )
        load_checkpoint(run / "checkpoints/best.pt", model, expected_config=config)
        return [
            model(ids := torch.tensor([encoder.encode(text).input_ids]), ids != 0)
            .softmax(dim=1)
            .tolist()[0]
            for text in texts
        ]


@pytest.mark.training
@pytest.mark.parametrize("family", ["mean_pool_mlp", "tfidf_logreg"])
def test_training_to_portable_inference_without_sources_or_network(
    development_artifact,
    tmp_path,
    family,
):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    launcher = tmp_path / "entry.py"
    launcher.write_text(LAUNCHER)
    source = tmp_path / "source"
    source.mkdir()
    config = source / "config.toml"
    config.write_text(NEURAL_CONFIG if family == "mean_pool_mlp" else TFIDF_CONFIG)
    run = source / "run"
    if family == "mean_pool_mlp":
        vocabulary = source / "vocabulary"
        built = invoke(
            launcher,
            elsewhere,
            [
                "build-vocabulary",
                "--data-dir",
                development_artifact,
                "--output-dir",
                vocabulary,
                "--min-frequency",
                "1",
            ],
        )
        assert built.returncode == 0, built.stderr
        args = ["train", "--vocabulary-dir", vocabulary, "--max-length", "8"]
    else:
        args = ["baseline", "tfidf"]
    trained = invoke(
        launcher,
        elsewhere,
        [
            *args,
            "--data-dir",
            development_artifact,
            "--config",
            config,
            "--output-dir",
            run,
        ],
    )
    assert trained.returncode == 0, trained.stderr

    validation = load_split(development_artifact, "val")
    texts = (*validation.texts, "zzunknown", "Plan " * 20)
    expected = reference_probabilities(run, family, texts)
    saved_labels = {
        row["sample_id"]: row["predicted_label"]
        for line in (run / "predictions.val.jsonl").read_text().splitlines()
        if (row := json.loads(line))
    }
    source_payload = run / (
        "checkpoints/best.pt" if family == "mean_pool_mlp" else "model.joblib"
    )
    payload_bytes = source_payload.read_bytes()
    bundle = tmp_path / "bundle"
    exported = invoke(
        launcher,
        elsewhere,
        [
            "export",
            "--run-dir",
            run,
            "--output-dir",
            bundle,
            "--model-id",
            "delivery-v1",
            "--data-manifest",
            development_artifact / "manifest.json",
            "--max-batch-size",
            "4",
            "--manifest-sha256",
            hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest(),
        ],
    )
    assert exported.returncode == 0, exported.stderr
    assert (
        bundle / ("model.pt" if family == "mean_pool_mlp" else "model.joblib")
    ).read_bytes() == payload_bytes

    delivered = tmp_path / "delivery"
    bundle.rename(delivered)
    # Every synthetic source, vocabulary, configuration, and training artifact is
    # removed. The child receives just the bundle and raw text over stdin.
    for path in tmp_path.iterdir():
        if path not in {delivered, launcher, elsewhere}:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    assert not development_artifact.exists() and not run.exists()
    manifest_bytes = (delivered / "manifest.json").read_bytes()
    before = {path.name: path.read_bytes() for path in delivered.iterdir()}
    predicted = invoke(
        launcher,
        elsewhere,
        [
            "predict",
            "--bundle",
            delivered,
            "--input",
            "-",
            "--batch-size",
            "2",
            "--manifest-sha256",
            hashlib.sha256(manifest_bytes).hexdigest(),
        ],
        inference=True,
        input_text="".join(
            json.dumps({"sample_id": str(index), "text": text}) + "\n"
            for index, text in enumerate(texts)
        ),
    )
    assert predicted.returncode == 0, predicted.stderr
    assert predicted.stderr == ""
    rows = [json.loads(line) for line in predicted.stdout.splitlines()]
    assert len(rows) == len(texts)
    for index, (row, probabilities) in enumerate(zip(rows, expected, strict=True)):
        assert row["sample_id"] == str(index)
        assert row["model_id"] == "delivery-v1"
        assert row["probabilities"] == pytest.approx(
            {str(label): value for label, value in enumerate(probabilities)},
            rel=1e-6,
            abs=1e-7,
        )
    assert [row["label_id"] for row in rows[: len(validation.texts)]] == [
        saved_labels[sample_id] for sample_id in validation.sample_ids
    ]
    assert rows[-1]["truncated"] == (family == "mean_pool_mlp")
    assert {path.name: path.read_bytes() for path in delivered.iterdir()} == before

    # Check rejection through the same isolated, installed entry point.
    state = json.loads(manifest_bytes)
    state["schema_version"] = 999
    (delivered / "manifest.json").write_text(json.dumps(state))
    rejected = invoke(
        launcher,
        elsewhere,
        ["predict", "--bundle", delivered, "--text", "growth"],
        inference=True,
    )
    assert rejected.returncode == 1
    assert rejected.stdout == ""
    assert "Error:" in rejected.stderr and "schema" in rejected.stderr
    assert "Traceback" not in rejected.stderr


def test_installed_console_script_is_available_outside_the_checkout(tmp_path):
    executable = Path(sysconfig.get_path("scripts")) / (
        "filing-sentence-classifier.exe"
        if os.name == "nt"
        else "filing-sentence-classifier"
    )
    result = subprocess.run(
        [str(executable), "--version"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == version("filing-sentence-classifier")
