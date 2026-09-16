"""Verify portable snapshots, strict restoration, and failed-write behavior."""

from dataclasses import replace

import pytest
import torch

from filing_sentence_classifier.models.mean_pool_mlp import MeanPoolMLP
from filing_sentence_classifier.training import checkpoints
from filing_sentence_classifier.training.checkpoints import (
    CheckpointError,
    load_checkpoint,
    save_checkpoint,
)
from filing_sentence_classifier.training.config import ModelConfig, TrainingConfig


@pytest.fixture
def config():
    return TrainingConfig(model=ModelConfig(4, 3, 0.2))


@pytest.fixture
def model(config):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        return MeanPoolMLP(
            7, embedding_dim=4, hidden_dim=3, dropout=config.model.dropout
        )


def test_snapshot_does_not_follow_later_updates_and_restores_buffers_and_mode(
    tmp_path, model, config
):
    model.register_buffer("counter", torch.tensor(3))
    original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    path = tmp_path / "checkpoints" / "best.pt"
    save_checkpoint(path, model, config=config, epoch=2, macro_f1=0.65)
    payload = torch.load(path, weights_only=True, map_location="cpu")
    assert payload["config"] == config.to_dict()
    assert payload["model_state_dict"]._metadata == model.state_dict()._metadata
    assert all(
        tensor.device.type == "cpu" for tensor in payload["model_state_dict"].values()
    )
    assert "optimizer_state_dict" not in payload
    with torch.no_grad():
        for value in model.state_dict().values():
            value.add_(10)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before_rng = torch.get_rng_state().clone()
    metadata = load_checkpoint(path, model, expected_config=config)
    assert (metadata.epoch, metadata.macro_f1, metadata.config) == (2, 0.65, config)
    assert not model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    assert torch.equal(before_rng, torch.get_rng_state())
    assert all(
        torch.equal(tensor, original[name])
        for name, tensor in model.state_dict().items()
    )


def test_new_checkpoints_do_not_overwrite_and_failed_replacements_preserve_previous_bytes(
    tmp_path, model, config, monkeypatch
):
    path = tmp_path / "best.pt"
    save_checkpoint(path, model, config=config, epoch=1, macro_f1=0.2)
    original = path.read_bytes()
    with pytest.raises(CheckpointError, match="Could not save"):
        save_checkpoint(path, model, config=config, epoch=2, macro_f1=0.3)
    assert path.read_bytes() == original

    def failed_save(payload, stream):
        stream.write(b"partial file")
        raise OSError("disk write failed")

    with monkeypatch.context() as context:
        context.setattr(checkpoints.torch, "save", failed_save)
        with pytest.raises(CheckpointError, match="disk write failed"):
            save_checkpoint(
                path, model, config=config, epoch=2, macro_f1=0.3, overwrite=True
            )
    assert path.read_bytes() == original
    assert set(tmp_path.iterdir()) == {path}
    save_checkpoint(path, model, config=config, epoch=2, macro_f1=0.3, overwrite=True)
    assert load_checkpoint(path, model).epoch == 2


@pytest.mark.parametrize(
    "case",
    [
        "schema",
        "extra",
        "config",
        "epoch",
        "score",
        "missing_key",
        "shape",
        "dtype",
        "nonfinite",
    ],
)
def test_invalid_checkpoints_are_rejected_before_model_mutation(
    tmp_path, model, config, case
):
    path = tmp_path / "best.pt"
    save_checkpoint(path, model, config=config, epoch=1, macro_f1=0.5)
    payload = torch.load(path, weights_only=True)
    state = payload["model_state_dict"]
    if case == "schema":
        payload["schema_version"] = True
    elif case == "extra":
        payload["unsupported"] = 1
    elif case == "config":
        payload["config"]["training"]["patience"] = 0
    elif case == "epoch":
        payload["epoch"] = 0
    elif case == "score":
        payload["macro_f1"] = float("nan")
    elif case == "missing_key":
        state.pop("classifier.3.bias")
    elif case == "shape":
        state["classifier.3.bias"] = torch.zeros(10)
    elif case == "dtype":
        state["classifier.3.bias"] = state["classifier.3.bias"].double()
    else:
        state["classifier.3.bias"].fill_(float("inf"))
    # A valid early key must not be copied before detecting a later invalid key.
    state["embedding.weight"].fill_(99)
    torch.save(payload, path)
    original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with pytest.raises(CheckpointError):
        load_checkpoint(path, model)
    assert model.training
    assert all(
        torch.equal(tensor, original[name])
        for name, tensor in model.state_dict().items()
    )


def test_wrong_run_or_model_is_rejected(tmp_path, model, config):
    path = tmp_path / "best.pt"
    save_checkpoint(path, model, config=config, epoch=1, macro_f1=0.5)
    with pytest.raises(CheckpointError, match="configuration"):
        load_checkpoint(path, model, expected_config=replace(config, learning_rate=0.1))
    with pytest.raises(CheckpointError, match="model type"):
        load_checkpoint(path, torch.nn.Linear(4, 3))


@pytest.mark.parametrize("content", [b"", b"invalid checkpoint"])
def test_unreadable_files_raise_checkpoint_errors(tmp_path, model, content):
    path = tmp_path / "best.pt"
    path.write_bytes(content)
    with pytest.raises(CheckpointError, match="Could not load"):
        load_checkpoint(path, model)


def test_nonfinite_model_cannot_replace_a_valid_checkpoint(tmp_path, model, config):
    path = tmp_path / "best.pt"
    save_checkpoint(path, model, config=config, epoch=1, macro_f1=0.5)
    original = path.read_bytes()
    with torch.no_grad():
        model.embedding.weight.fill_(float("nan"))
    with pytest.raises(CheckpointError, match="finite tensors"):
        save_checkpoint(
            path, model, config=config, epoch=2, macro_f1=0.6, overwrite=True
        )
    assert path.read_bytes() == original
