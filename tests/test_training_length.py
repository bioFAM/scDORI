"""Training length from optimizer steps: "auto" epoch limits and patience, explicit overrides, old configs."""

import json
import math

import anndata as ad
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scdori import resolve_training_length
from scdori._core.config import TrainConfig
from scdori._core.models import scDoRI
from scdori._core.train_scdori import train_scdori_phases
from scdori.configuration import load_notebook_config


# training cells (batch 128): a few hundred metacells, the small and large benchmark ends, the gastrulation example
@pytest.mark.parametrize(
    "cells,steps,phase1_epochs,phase1_patience,grn_epochs,grn_patience",
    [
        (400, 4, 6250, 250, 6250, 75),
        (100, 1, 25000, 1000, 25000, 300),
        (4224, 33, 1000, 31, 1000, 10),
        (11008, 86, 1000, 30, 1000, 5),
        (45488, 356, 1000, 30, 1000, 5),
    ],
)
def test_auto_values(cells, steps, phase1_epochs, phase1_patience, grn_epochs, grn_patience):
    assert math.ceil(cells / 128) == steps
    one = resolve_training_length(TrainConfig(), steps, "phase1")
    two = resolve_training_length(TrainConfig(), steps, "grn")
    assert (one["max_epochs"], one["patience"]) == (phase1_epochs, phase1_patience)
    assert (two["max_epochs"], two["patience"]) == (grn_epochs, grn_patience)
    assert one["max_epochs_source"] == one["patience_source"] == "auto"
    assert one["max_steps_resolved"] == phase1_epochs * steps


def test_patience_counts_validation_checks():
    config = TrainConfig(eval_frequency=4)
    assert resolve_training_length(config, 4, "phase1")["patience"] == max(30, math.ceil(1000 / 16))
    assert resolve_training_length(config, 4, "grn")["patience"] == max(5, math.ceil(300 / 16))


def test_explicit_values_and_step_budgets_override_auto():
    config = TrainConfig(max_scdori_epochs=200, phase1_patience=7, grn_max_steps=5000, grn_patience_steps=40)
    one = resolve_training_length(config, 4, "phase1")
    assert (one["max_epochs"], one["patience"], one["max_epochs_source"]) == (200, 7, "config")
    two = resolve_training_length(config, 4, "grn")
    assert (two["max_epochs"], two["patience"]) == (max(1000, 1250), max(5, 10))
    assert resolve_training_length(TrainConfig(phase1_patience=0), 4, "phase1")["patience"] == 0


def test_old_configs_with_integers_are_unchanged(tmp_path):
    old = {"max_scdori_epochs": 1000, "phase1_patience": 50, "max_grn_epochs": 1000, "grn_val_patience": 5}
    path = tmp_path / "scdori_config.json"
    path.write_text(json.dumps({"training": old}))
    config = load_notebook_config(path).training
    for steps in (1, 4, 86, 356):
        one = resolve_training_length(config, steps, "phase1")
        two = resolve_training_length(config, steps, "grn")
        assert (one["max_epochs"], one["patience"], two["max_epochs"], two["patience"]) == (1000, 50, 1000, 5)
    path.write_text(json.dumps({"training": {"max_scdori_epochs": "auto", "grn_val_patience": "auto"}}))
    assert load_notebook_config(path).training.max_scdori_epochs == "auto"
    for bad in ("forever", -1, 2.5, True):
        path.write_text(json.dumps({"training": {"phase1_patience": bad}}))
        with pytest.raises(ValueError):
            load_notebook_config(path)


def test_invalid_values_are_rejected():
    for bad in ("forever", -1, 2.5, True, None):
        with pytest.raises(ValueError):
            resolve_training_length(TrainConfig(max_grn_epochs=bad), 4, "grn")
    with pytest.raises(ValueError):
        resolve_training_length(TrainConfig(), 0, "phase1")
    with pytest.raises(ValueError):
        resolve_training_length(TrainConfig(phase1_max_steps=0), 4, "phase1")


def test_template_uses_auto():
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "configs/scdori.example.json"
    training = load_notebook_config(example).training
    assert training.max_scdori_epochs == training.max_grn_epochs == "auto"
    assert training.phase1_patience == training.grn_val_patience == "auto"


def test_phase1_uses_and_records_the_resolved_values(tmp_path, monkeypatch):
    torch.manual_seed(3)
    rng = np.random.default_rng(3)
    model = scDoRI(torch.device("cpu"), 7, 11, 3, 4, 2, 9, 5)
    rna = ad.AnnData(rng.poisson(3, (12, 7)).astype(np.float32))
    atac = ad.AnnData(rng.poisson(3, (12, 11)).astype(np.float32))
    cells, tfs = np.ones((12, 1), dtype=np.float32), np.array([1, 3, 5])
    batches = np.eye(2, dtype=np.float32)[np.arange(12) % 2]
    loader = DataLoader(TensorDataset(torch.arange(8)), batch_size=3, shuffle=True)  # 3 steps per epoch
    config = TrainConfig(weights_folder_scdori=str(tmp_path), phase1_max_steps=9, phase1_patience_steps=3)
    epochs = []

    def phase(epoch, cfg):
        epochs.append(epoch)
        return "warmup_1" if epoch < cfg.epoch_warmup_1 else "warmup_2"

    monkeypatch.setattr("scdori._core.train_scdori.get_phase_scdori", phase)
    config.max_scdori_epochs = 4  # explicit integer wins over "auto" (which would give 1000)
    train_scdori_phases(model, "cpu", loader, DataLoader(TensorDataset(torch.arange(8, 12)), batch_size=4),
                        rna, atac, cells, tfs, batches, config)
    assert epochs == [0, 1, 2, 3]
    record = json.loads((tmp_path / "training_length_phase1.json").read_text())
    assert record["steps_per_epoch"] == 3 and record["max_epochs"] == 4 and record["max_epochs_source"] == "config"
    assert record["patience"] == 30 and record["patience_source"] == "auto"  # max(30, ceil(3 / 3))
    assert model.phase1_training_length == record
