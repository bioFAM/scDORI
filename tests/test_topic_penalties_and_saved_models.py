"""Topic penalties and loading saved fits for downstream analysis."""

import copy
import json
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scdori._core.config import TrainConfig
from scdori._core.downstream_inputs import load_saved_tf_expression
from scdori._core.models import scDoRI
from scdori._core.topic_penalties import topic_usage_penalties, validate_topic_penalties
from scdori.configuration import check_checkpoint_model_config, load_notebook_config


def test_penalty_formula_gradients_and_batch_reduction():
    logits = torch.tensor([[1.0, -1.0, 0.0], [0.0, 2.0, 1.0]], dtype=torch.float64, requires_grad=True)
    theta = logits.softmax(-1)
    cfg = TrainConfig(theta_entropy_weight=10, usage_balance_weight=3)
    actual = topic_usage_penalties(theta, cfg)
    usage = theta.mean(0)
    reference = -10 * (theta * (theta + 1e-12).log()).sum() + 3 * len(theta) * (usage * (usage + 1e-12).log()).sum()
    torch.testing.assert_close(sum(actual.values()), reference, rtol=0, atol=0)
    g = torch.autograd.grad(sum(actual.values()), logits, retain_graph=True)[0]
    torch.testing.assert_close(g, torch.autograd.grad(reference, logits)[0])
    repeated = topic_usage_penalties(theta.repeat(2, 1), cfg)
    for name in actual:
        torch.testing.assert_close(repeated[name], 2 * actual[name])
    assert topic_usage_penalties(theta, TrainConfig()) == {}
    assert topic_usage_penalties(theta, SimpleNamespace()) == {}  # legacy API objects


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "10"])
def test_invalid_topic_weights(value):
    with pytest.raises(ValueError, match="finite nonnegative"):
        validate_topic_penalties(SimpleNamespace(theta_entropy_weight=value))


def test_zero_default_training_is_bitwise_unchanged_and_nonzero_weights_work(tmp_path, monkeypatch):
    from scdori._core import train_scdori

    torch.manual_seed(29)
    rng = np.random.default_rng(29)
    initial = scDoRI(torch.device("cpu"), 7, 11, 3, 4, 2, 9, 5)
    rna = ad.AnnData(rng.poisson(3, (12, 7)).astype(np.float32))
    atac = ad.AnnData(rng.poisson(3, (12, 11)).astype(np.float32))
    train = DataLoader(TensorDataset(torch.arange(8)), batch_size=4, shuffle=True)
    val = DataLoader(TensorDataset(torch.arange(8, 12)), batch_size=4)
    inputs = (rna, atac, np.ones((12, 1)), np.array([1, 3, 5]), np.eye(2)[np.arange(12) % 2])
    results = []
    for mode in ("reference", "zero", "positive"):
        cfg = TrainConfig(
            max_scdori_epochs=2,
            epoch_warmup_1=1,
            eval_frequency=1,
            weights_folder_scdori=str(tmp_path / mode),
            theta_entropy_weight=10 if mode == "positive" else 0,
            usage_balance_weight=10 if mode == "positive" else 0,
        )
        model = copy.deepcopy(initial)
        monkeypatch.setattr(
            train_scdori, "topic_usage_penalties", (lambda *_: {}) if mode == "reference" else topic_usage_penalties
        )
        torch.manual_seed(83)
        train_scdori.train_scdori_phases(model, "cpu", train, val, *inputs, cfg)
        results.append(model)
    assert results[0].phase1_history == results[1].phase1_history
    for key, tensor in results[0].state_dict().items():
        torch.testing.assert_close(tensor, results[1].state_dict()[key], rtol=0, atol=0)
    for p, q in zip(results[0].parameters(), results[1].parameters(), strict=True):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    assert any(not torch.equal(p, q) for p, q in zip(results[1].parameters(), results[2].parameters(), strict=True))
    assert results[2].phase1_history[-1]["train_theta_entropy"] > 0
    assert results[2].phase1_history[-1]["train_usage_balance"] < 0
    # Penalties change fitting, but not validation's definition on a fixed model.
    zero = TrainConfig()
    positive = TrainConfig(theta_entropy_weight=100, usage_balance_weight=100)
    assert train_scdori.compute_eval_loss_scdori(
        results[2], "cpu", val, *inputs, zero
    ) == train_scdori.compute_eval_loss_scdori(results[2], "cpu", val, *inputs, positive)


@pytest.mark.parametrize("shape", [(3, 2), (3, 1, 2)])
def test_saved_expression_is_authoritative(tmp_path, shape):
    values = np.array([[0.3, 0.7], [0, 0], [1, 0.2]], dtype=np.float32)
    np.save(tmp_path / "gate_used.npy", values.reshape(shape))
    (tmp_path / "heldout_summary.json").write_text(json.dumps({"gate": {"policy": "mean_both_axes", "clamp": 0.2}}))
    actual, record = load_saved_tf_expression(tmp_path, 3, 2, excluded_topics=[1])
    np.testing.assert_array_equal(actual, values)
    assert actual.dtype == values.dtype and record["recomputed"] is False
    assert record["recorded_expression_settings"]["clamp"] == 0.2
    assert len(record["sha256"]) == 64
    np.save(tmp_path / "topic_tf_expression.npy", values)
    np.testing.assert_array_equal(load_saved_tf_expression(tmp_path, 3, 2)[0], values)
    np.save(tmp_path / "topic_tf_expression.npy", values * 0.5)
    with pytest.raises(ValueError, match="Conflicting"):
        load_saved_tf_expression(tmp_path, 3, 2)
    np.testing.assert_array_equal(
        load_saved_tf_expression(tmp_path, 3, 2, filename=tmp_path / "gate_used.npy")[0], values
    )


@pytest.mark.parametrize(
    "values,excluded",
    [([[2, 0]], []), ([[np.nan, 0]], []), ([[0.1]], []), ([[0.1, 0.2]], [0]), ([[0, 0]], [2]), ([[0, 0]], [0.1])],
)
def test_reject_invalid_saved_expression(tmp_path, values, excluded):
    np.save(tmp_path / "gate_used.npy", np.array(values))
    with pytest.raises(ValueError):
        load_saved_tf_expression(tmp_path, 1, 2, excluded_topics=excluded)


def test_missing_expression_and_explicit_missing_path_never_fall_back(tmp_path):
    with pytest.raises(FileNotFoundError, match="original training split"):
        load_saved_tf_expression(tmp_path, 1, 2)
    np.save(tmp_path / "gate_used.npy", np.zeros((1, 2)))
    with pytest.raises(FileNotFoundError):
        load_saved_tf_expression(tmp_path, 1, 2, filename=tmp_path / "missing.npy")


@pytest.mark.parametrize(
    "change",
    [
        {"theta_activation": "relu"},
        {"theta_temperature": 0.5},
        {"decoder_norm": "none"},
        {"decoder_norm_atac": "layernorm"},
        {"label_weight": 10},
        {"theta_temperature_final": 0.25},
    ],
)
def test_checkpoint_with_unsupported_settings_is_rejected(tmp_path, change):
    (tmp_path / "config.json").write_text(json.dumps({"phase1_run": "/base", "training": change}))
    folder = tmp_path / "phase2"
    folder.mkdir()
    with pytest.raises(ValueError, match="scDORI cannot reproduce"):
        check_checkpoint_model_config(folder / "best.pth", TrainConfig())


def test_supported_config_roundtrip_and_checkpoint(tmp_path):
    payload = {
        "training": {"theta_entropy_weight": 10, "usage_balance_weight": 10},
        "downstream": {
            "tf_expression_file": "gate_used.npy",
            "permutation_seed": 42,
            "activator_alpha": 0.025,
            "repressor_alpha": 0.05,
        },
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(payload))
    cfg = load_notebook_config(p)
    assert cfg.training.theta_entropy_weight == 10
    assert cfg.downstream.tf_expression_file == str(tmp_path / "gate_used.npy")
    assert cfg.downstream.permutation_seed == 42
    assert check_checkpoint_model_config(tmp_path / "best.pth", cfg.training) == payload["training"]
    payload["downstream"]["permutation_seed"] = "42"
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Invalid type"):
        load_notebook_config(p)


def test_pruned_checkpoint_without_selection_cannot_silently_use_all_topics(tmp_path):
    from scdori.configuration import check_notebook_inputs

    folder = tmp_path / "phase2"
    folder.mkdir()
    # Even if the current analysis config says "all", the saved fit controls this.
    (folder / "notebook_config.json").write_text(json.dumps({"training": {"grn_topic_selection": "live"}}))
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"training": {"weights_folder_grn": "phase2"}}))
    cfg = load_notebook_config(p)
    with pytest.raises(FileNotFoundError, match="copy its grn_topics.json"):
        check_notebook_inputs(cfg, "downstream")


@pytest.mark.parametrize("activator_alpha,repressor_alpha", [(0.05, 0.05), (0.025, 0.1)])
def test_notebook_reloads_networks_with_the_configured_cutoffs(tmp_path, activator_alpha, repressor_alpha):
    from pathlib import Path

    notebook = json.loads((Path(__file__).resolve().parents[1] / "docs/notebooks/downstream.ipynb").read_text())
    cell = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code" and "grn_act = np.load" in "".join(c["source"])
    )
    act, rep = np.full((2, 3, 4), 7), np.full((2, 3, 4), -2)
    np.save(tmp_path / f"grn_activator__{activator_alpha}.npy", act)
    np.save(tmp_path / f"grn_repressor__{repressor_alpha}.npy", rep)
    namespace = {
        "np": np,
        "GRN_OUTPUT": tmp_path,
        "ACTIVATOR_ALPHA": activator_alpha,
        "REPRESSOR_ALPHA": repressor_alpha,
    }
    exec(compile(cell, "downstream GRN reload cell", "exec"), namespace)
    np.testing.assert_array_equal(namespace["grn_act"], act)
    np.testing.assert_array_equal(namespace["grn_rep"], rep)
