"""A configuration object that lacks a setting must behave like ``TrainConfig()`` for that setting."""

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scdori._core import train_grn
from scdori._core.config import TrainConfig, train_default
from scdori._core.models import scDoRI
from scdori._core.topic_selection import apply_grn_topic_selection, resolve_grn_topics
from scdori._core.train_grn import get_tf_expression, topic_tf_detection_filter
from scdori.pp.config import PreprocessingConfig

FALLBACK_KEYS = [
    "tf_expression_scaling",
    "tf_expression_min_detection",
    "freeze_phase1_in_grn",
    "grn_topic_selection",
    "grn_dead_topic_max_theta",
    "grn_dead_topic_max_mean_fraction",
    "cache_frozen_grn",
    "phase2_train_nuisance",
    "update_encoder_in_grn",
    "update_peak_gene_in_grn",
    "update_topic_peak_in_grn",
    "update_topic_tf_in_grn",
]


def full_config():
    default = TrainConfig()
    return SimpleNamespace(**{f.name: getattr(default, f.name) for f in fields(TrainConfig)})


def without(*keys):
    config = full_config()
    for key in keys:
        delattr(config, key)
    return config


def test_train_default_is_the_dataclass_default():
    default = TrainConfig()
    for item in fields(TrainConfig):
        assert train_default(item.name) == getattr(default, item.name), item.name
    with pytest.raises(KeyError):
        train_default("not_a_setting")


def test_current_defaults_that_fallbacks_now_follow():
    assert train_default("freeze_phase1_in_grn") is True
    assert train_default("grn_topic_selection") == "live"
    assert train_default("tf_expression_scaling") == "mean_within_across"
    assert train_default("tf_expression_min_detection") == 0.02
    assert train_default("grn_dead_topic_max_theta") == 0.05
    assert train_default("grn_dead_topic_max_mean_fraction") == 0.1
    assert train_default("phase2_train_nuisance") is False


def test_no_gene_or_tf_defaults_in_preprocessing():
    config = PreprocessingConfig()
    assert config.genes_user == [] and config.tfs_user == [] and config.peaks_user == []
    config.genes_user.append("X")
    assert PreprocessingConfig().genes_user == []  # not shared between instances


def theta_with_collapsed_topics(k=6):
    theta = np.full((40, k), 1.0 / k)
    theta[:, 4] = 1e-9  # collapsed under any threshold
    theta[:2, 5] = 0.07  # max 0.07 in 2 of 40 cells, mean below 0.1 / K: collapsed with max 0.1, kept with 0.05
    theta[2:, 5] = 1e-6
    theta /= theta.sum(1, keepdims=True)
    return theta


def live(config):
    config.grn_topic_selection = "live"
    return config


@pytest.mark.parametrize("key", [None, "grn_dead_topic_max_theta", "grn_dead_topic_max_mean_fraction"])
def test_live_rule_without_a_key_matches_trainconfig(key):
    model = scDoRI(torch.device("cpu"), 5, 7, 3, 6, 1, 8, 4)
    theta = theta_with_collapsed_topics()
    expected = resolve_grn_topics(model, live(TrainConfig()), theta)
    np.testing.assert_array_equal(expected, [0, 1, 2, 3, 5])
    config = live(without(key) if key else full_config())
    np.testing.assert_array_equal(resolve_grn_topics(model, config, theta), expected)
    np.testing.assert_array_equal(
        resolve_grn_topics(model, live(TrainConfig(grn_dead_topic_max_theta=0.1)), theta), [0, 1, 2, 3])


def test_missing_selection_key_uses_the_live_default():
    model = scDoRI(torch.device("cpu"), 5, 7, 3, 6, 1, 8, 4)
    null = {"keep": np.array([True, False, True, True, False, True])}
    for config in (TrainConfig(), without("grn_topic_selection")):
        np.testing.assert_array_equal(resolve_grn_topics(model, config, theta_with_collapsed_topics()), [0, 1, 2, 3, 5])
        with pytest.raises(ValueError, match="live"):  # the default needs topic weights; a shuffle result is not used
            resolve_grn_topics(model, config, None, null=null)


@pytest.mark.parametrize("key", ["freeze_phase1_in_grn", "update_encoder_in_grn", "grn_dead_topic_max_theta"])
def test_apply_selection_without_a_key_matches_trainconfig(key):
    theta = torch.as_tensor(theta_with_collapsed_topics())
    records = []
    for config in (live(TrainConfig()), live(without(key))):
        model = scDoRI(torch.device("cpu"), 5, 7, 3, 6, 1, 8, 4)
        records.append(apply_grn_topic_selection(model, config, lambda: theta))
    assert records[0] == records[1]
    assert records[1]["excluded_topics"] == [4]
    assert records[1]["dead_max_theta"] == 0.05


def test_detection_floor_without_the_key_matches_trainconfig():
    rng = np.random.default_rng(0)
    import anndata as ad

    x = np.zeros((30, 2), dtype=np.float32)
    x[:, 0] = rng.poisson(5, 30) + 1  # TF 0 seen in every cell, TF 1 never
    rna = ad.AnnData(x)
    selected = np.arange(30)[:, None]
    _, expected = topic_tf_detection_filter(selected, rna, [0, 1], full_config())
    _, missing = topic_tf_detection_filter(selected, rna, [0, 1], without("tf_expression_min_detection"))
    np.testing.assert_array_equal(missing, expected)
    assert not missing[0, 1]


def test_tf_expression_scaling_without_the_key_matches_trainconfig(monkeypatch):
    rng = np.random.default_rng(1)
    import anndata as ad

    rna = ad.AnnData(rng.poisson(3, (60, 4)).astype(np.float32))
    theta = rng.random((60, 3))
    ids = np.arange(60)
    monkeypatch.setattr("scdori._core.train_grn.get_latent_topics", lambda *a, **k: (theta, ids))
    monkeypatch.setattr("scdori._core.topic_selection.excluded_grn_topics", lambda model: [])
    model = SimpleNamespace(topic_tf_decoder=torch.zeros(1, dtype=torch.float32))
    args = (model, "cpu", None, rna, None, None, [0, 1, 2, 3], None)
    expected = get_tf_expression("True", *args, full_config())
    missing = get_tf_expression("True", *args, without("tf_expression_scaling"))
    torch.testing.assert_close(missing, expected, rtol=0, atol=0)
    within = get_tf_expression("True", *args, SimpleNamespace(**{**vars(full_config()),
                                                                 "tf_expression_scaling": "within_topic"}))
    assert not torch.equal(within, expected)  # the old fallback would have given a different gate


def test_train_model_grn_without_freeze_key_takes_the_frozen_path(monkeypatch):
    called = []
    monkeypatch.setattr("scdori._core.frozen_grn.train_frozen_grn", lambda *a, **k: called.append(True) or "frozen")
    out = train_grn.train_model_grn(None, "cpu", None, None, None, None, None, None, None,
                                    without("freeze_phase1_in_grn"))
    assert out == "frozen" and called
