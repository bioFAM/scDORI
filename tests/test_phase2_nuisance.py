"""phase2_train_nuisance: phase 2 also trains the RNA dispersion and the RNA library-size network, with the cache."""

import copy

import anndata as ad
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scdori._core.config import TrainConfig
from scdori._core.frozen_grn import NUISANCE_PARAMETER_PREFIXES
from scdori._core.models import scDoRI
from scdori._core.train_grn import train_model_grn

GRN = ("tf_gene_topic_", "rna_grn_batch_")
EXPECTED_NUISANCE = {"rna_alpha_nb",
                     *(f"rna_library_factor.{i}.{p}" for i in (0, 1, 4, 5, 7) for p in ("weight", "bias"))}


@pytest.fixture
def case(tmp_path):
    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    model = scDoRI(torch.device("cpu"), 7, 11, 3, 4, 2, 9, 5)
    with torch.no_grad():
        model.tf_binding_matrix_repressor.mul_(-1)
    rna = ad.AnnData(rng.poisson(3, (12, 7)).astype(np.float32))
    atac = ad.AnnData(rng.poisson(3, (12, 11)).astype(np.float32))
    cells = np.ones((12, 1), dtype=np.float32)
    tfs = np.array([1, 3, 5])
    batches = np.eye(2, dtype=np.float32)[np.arange(12) % 2]
    config = TrainConfig(freeze_phase1_in_grn=True, grn_topic_selection="all", max_grn_epochs=3, cells_per_topic=3,
                         batch_size_cell_prediction=5, weights_folder_grn=str(tmp_path))
    return model, rna, atac, cells, tfs, batches, config


def loader(ids, shuffle=False):
    return DataLoader(TensorDataset(torch.tensor(ids)), batch_size=4, shuffle=shuffle)


def train(case, model=None, **settings):
    m, rna, atac, cells, tfs, batches, config = case
    model = model or m
    config = copy.deepcopy(config)
    for key, value in settings.items():
        setattr(config, key, value)
    torch.manual_seed(99)
    train_model_grn(model, "cpu", loader([8, 2, 10, 0, 1, 3, 7, 9], True), loader([4, 5, 6, 11]),
                    rna, atac, cells, tfs, batches, config)
    return model


def test_parameter_list_is_the_rna_dispersion_and_library_network(case):
    names = {n for n, _ in case[0].named_parameters() if n.startswith(NUISANCE_PARAMETER_PREFIXES)}
    assert names == EXPECTED_NUISANCE


def test_default_is_unchanged_and_equals_a_config_without_the_key(case):
    reference = copy.deepcopy(case[0])
    before = copy.deepcopy(case[0].state_dict())
    explicit = train(case, phase2_train_nuisance=False)
    m, rna, atac, cells, tfs, batches, config = case
    config_without = copy.deepcopy(config)
    del config_without.phase2_train_nuisance
    torch.manual_seed(99)
    train_model_grn(reference, "cpu", loader([8, 2, 10, 0, 1, 3, 7, 9], True), loader([4, 5, 6, 11]),
                    rna, atac, cells, tfs, batches, config_without)
    for name, value in explicit.state_dict().items():
        assert torch.equal(value, reference.state_dict()[name]), name
        if not name.startswith(GRN):
            assert torch.equal(value, before[name]), name


@pytest.mark.parametrize("cached", [True, False])
def test_option_trains_exactly_the_listed_parameters(case, cached):
    before = copy.deepcopy(case[0].state_dict())
    model = train(case, phase2_train_nuisance=True, cache_frozen_grn=cached)
    after = model.state_dict()
    changed = {name for name, _ in model.named_parameters() if not torch.equal(after[name], before[name])}
    assert changed == EXPECTED_NUISANCE | {n for n, _ in model.named_parameters() if n.startswith(GRN)}
    for name, value in after.items():
        if "running_" in name or "num_batches_tracked" in name:
            moved = not torch.equal(value, before[name])
            assert moved == name.startswith(("rna_library_factor", "rna_grn_batch_norm")), name


def count_calls(module):
    calls = []
    module.register_forward_hook(lambda m, i, o: calls.append(m.training))
    return calls


def test_cache_is_used_and_library_network_runs_in_training_mode(case):
    default = copy.deepcopy(case[0])
    calls_default = count_calls(default.encoder_rna)
    train(case, default)
    model = case[0]
    calls_option = count_calls(model.encoder_rna)
    library = count_calls(model.rna_library_factor)
    dropout = {"library": [], "other": []}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            key = "library" if name.startswith("rna_library_factor") else "other"
            module.register_forward_pre_hook(lambda m, _, k=key: dropout[k].append(m.training))
    train(case, model, phase2_train_nuisance=True, cache_frozen_grn=True)
    assert len(calls_option) == len(calls_default)  # the encoder only runs to build the cache, as by default
    assert set(calls_option) == {False}
    assert True in library and False in library  # training epochs in train mode, validation in eval
    assert True in dropout["library"] and set(dropout["other"]) <= {False}


# Linear biases directly before a BatchNorm in training mode have a true gradient of zero; Adam turns their
# rounding-level gradients into learning-rate-sized steps, so these two biases (and the running means that absorb
# them) differ between any two computation orders. They are compared through the network's output instead.
BN_PRECEDED = ("rna_library_factor.0.bias", "rna_library_factor.4.bias")


def test_deterministic_with_seed_and_cache_matches_full_forward(case):
    first = train(case, copy.deepcopy(case[0]), phase2_train_nuisance=True)
    second = train(case, copy.deepcopy(case[0]), phase2_train_nuisance=True)
    full = train(case, copy.deepcopy(case[0]), phase2_train_nuisance=True, cache_frozen_grn=False)
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name]), name
        if name not in BN_PRECEDED and "running_mean" not in name:
            torch.testing.assert_close(value, full.state_dict()[name], atol=3e-5, rtol=3e-4)
    x = torch.as_tensor(case[1].X)
    with torch.no_grad():
        cached_out = first.eval().rna_library_factor(x)
        full_out = full.eval().rna_library_factor(x)
    torch.testing.assert_close(cached_out, full_out, atol=0, rtol=1e-2)


def test_own_learning_rate(case):
    before = copy.deepcopy(case[0].state_dict())
    model = train(case, phase2_train_nuisance=True, learning_rate_nuisance=0.0)
    after = model.state_dict()
    for name in EXPECTED_NUISANCE:
        assert torch.equal(after[name], before[name]), name  # learning rate 0: parameters stay
    assert not torch.equal(after["tf_gene_topic_activator_grn"], before["tf_gene_topic_activator_grn"])
    assert not torch.equal(after["rna_library_factor.1.running_mean"], before["rna_library_factor.1.running_mean"])


def test_legacy_phase2_rejects_the_option(case):
    with pytest.raises(ValueError, match="phase2_train_nuisance requires freeze_phase1_in_grn"):
        train(case, phase2_train_nuisance=True, freeze_phase1_in_grn=False)


def test_config_accepts_the_learning_rate(tmp_path):
    import json

    from scdori.configuration import load_notebook_config

    path = tmp_path / "c.json"
    for value in (None, 0.0005, 1):
        path.write_text(json.dumps({"training": {"learning_rate_nuisance": value}}))
        assert load_notebook_config(path).training.learning_rate_nuisance == value
    path.write_text(json.dumps({"training": {"learning_rate_nuisance": "fast"}}))
    with pytest.raises(ValueError):
        load_notebook_config(path)
