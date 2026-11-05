import copy

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, TensorDataset

from scdori._core.config import TrainConfig
from scdori._core.dataloader import create_minibatch
from scdori._core.evaluation import get_latent_topics
from scdori._core.frozen_grn import (
    _regularization,
    build_cell_cache,
    build_grn_factors,
    estimate_frozen_cache_bytes,
    forward_from_topics,
    freeze_phase1,
)
from scdori._core.models import initialize_scdori_parameters, scDoRI
from scdori._core.train_grn import get_tf_expression, train_model_grn
from scdori._core.train_scdori import train_scdori_phases
from scdori._core.utils import log_nb_positive


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
    config = TrainConfig(
        freeze_phase1_in_grn=True,
        grn_topic_selection="all",
        max_grn_epochs=2,
        cells_per_topic=3,
        batch_size_cell_prediction=5,
        weights_folder_grn=str(tmp_path),
    )
    return model, rna, atac, cells, tfs, batches, config


def loader(ids, shuffle=False):
    return DataLoader(TensorDataset(torch.tensor(ids)), batch_size=4, shuffle=shuffle)


def test_phase1_checkpoint_and_patience_reset_when_objective_changes(case, monkeypatch):
    from scdori._core import train_scdori

    model, rna, atac, cells, tfs, batches, config = case
    config.max_scdori_epochs = 3
    config.epoch_warmup_1 = 1
    config.phase1_patience = 0
    evaluated, saved = [], []
    losses = iter([1.0, 100.0, 90.0])

    def evaluate(*args):
        evaluated.append(next(losses))
        return evaluated[-1], 0.0, 0.0, 0.0

    monkeypatch.setattr(train_scdori, "compute_eval_loss_scdori", evaluate)
    monkeypatch.setattr(train_scdori, "save_model_weights", lambda *args: saved.append(evaluated[-1]))
    train_scdori_phases(model, "cpu", loader(range(8)), loader(range(8, 12)), rna, atac, cells, tfs, batches, config)
    assert evaluated == [1.0, 100.0, 90.0]
    assert saved == evaluated


def inputs(case, ids):
    model, rna, atac, cells, tfs, batches, _ = case
    x, tf, lib, nc, batch = create_minibatch("cpu", torch.tensor(ids), rna, atac, cells, tfs, batches)
    return x[:, : model.num_genes], x[:, model.num_genes :], tf, lib, nc, batch


def test_fixed_decoder_cache_matches_all_auxiliary_losses(case, monkeypatch):
    from scdori._core.frozen_grn import _auxiliary_losses

    model, rna, atac, cells, tfs, batches, _ = case
    freeze_phase1(model)
    expected = []
    with torch.no_grad():
        for start in range(0, 12, 4):
            x, a, tf, lib, nc, batch = inputs(case, list(range(start, start + 4)))
            out = model(x, a, tf, None, lib[:, :1], lib[:, 1:], nc, batch)
            expected.append(_auxiliary_losses(model, out, x, a, tf, lib))
    calls = []
    original = model.topic_gene_matrix

    def counted():
        calls.append(1)
        return original()

    monkeypatch.setattr(model, "topic_gene_matrix", counted)
    cache = build_cell_cache(model, "cpu", rna, atac, cells, tfs, batches, 4)
    assert len(calls) == 1
    torch.testing.assert_close(cache.auxiliary, torch.cat(expected), rtol=0, atol=0)
    estimate = estimate_frozen_cache_bytes(model, rna.n_obs)
    assert estimate["cpu_cell_cache"] == sum(
        t.numel() * t.element_size() for t in (cache.theta, cache.library, cache.auxiliary)
    )
    factors = build_grn_factors(model)
    assert estimate["device_grn_factors"] == sum(t.numel() * t.element_size() for t in factors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_cached_forward_gradients_and_update(case):
    model, rna, atac, cells, tfs, batches, _ = case
    model = model.to("cuda")
    model.device = torch.device("cuda")
    freeze_phase1(model)
    reference = copy.deepcopy(model)
    cache = build_cell_cache(model, "cuda", rna, atac, cells, tfs, batches, 4)
    factors = build_grn_factors(model)
    topic_tf = torch.rand(model.num_topics, model.num_tfs, device="cuda")
    model.rna_grn_batch_norm.train()
    reference.rna_grn_batch_norm.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
    ref_opt = torch.optim.Adam([p for p in reference.parameters() if p.requires_grad], lr=0.001)
    for ids in ([8, 3, 10, 1], [11, 2, 7, 0]):
        inp, tf, lib, nc, batch = create_minibatch("cuda", torch.tensor(ids), rna, atac, cells, tfs, batches)
        x, a = inp[:, : model.num_genes], inp[:, model.num_genes :]
        expected = reference(x, a, tf, topic_tf, lib[:, :1], lib[:, 1:], nc, batch, phase="grn")["mu_nb_rna_grn"]
        actual = forward_from_topics(
            model, cache.theta[ids].cuda(), cache.library[ids].cuda(), batch, topic_tf, factors
        )
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        opt.zero_grad()
        ref_opt.zero_grad()
        (-log_nb_positive(x, actual, torch.nn.functional.softplus(model.rna_alpha_nb)).mean()).backward()
        (-log_nb_positive(x, expected, torch.nn.functional.softplus(reference.rna_alpha_nb)).mean()).backward()
        for p, q in zip(model.parameters(), reference.parameters(), strict=True):
            if p.requires_grad:
                torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4)
        opt.step()
        ref_opt.step()
        for p, q in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(p, q, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("training", [False, True])
def test_cached_forward_gradients_and_update_match_full(case, training):
    model, rna, atac, cells, tfs, batches, config = case
    freeze_phase1(model)
    reference = copy.deepcopy(model)
    cache = build_cell_cache(model, "cpu", rna, atac, cells, tfs, batches, 5)
    factors = build_grn_factors(model)
    topic_tf = torch.rand(4, 3)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
    ref_opt = torch.optim.Adam([p for p in reference.parameters() if p.requires_grad], lr=0.001)
    model.rna_grn_batch_norm.train(training)
    reference.rna_grn_batch_norm.train(training)
    # Same-sized batches of different cells must never share cell features.
    for ids in ([8, 3, 10, 1], [11, 2, 7, 0]):
        x, a, tf, lib, nc, batch = inputs(case, ids)
        expected = reference(x, a, tf, topic_tf, lib[:, :1], lib[:, 1:], nc, batch, phase="grn")
        actual = forward_from_topics(model, cache.theta[ids], cache.library[ids], batch, topic_tf, factors)
        torch.testing.assert_close(actual, expected["mu_nb_rna_grn"], atol=2e-6, rtol=2e-5)
        loss = -log_nb_positive(x, actual, torch.nn.functional.softplus(model.rna_alpha_nb)).mean()
        ref_loss = -log_nb_positive(
            x, expected["mu_nb_rna_grn"], torch.nn.functional.softplus(reference.rna_alpha_nb)
        ).mean()
        opt.zero_grad()
        ref_opt.zero_grad()
        loss.backward()
        ref_loss.backward()
        for (_name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
            if p.requires_grad:
                torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4)
        opt.step()
        ref_opt.step()
        for p, q in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(p, q, atol=2e-5, rtol=2e-4)
    assert not cache.theta.requires_grad
    assert not factors[0].requires_grad
    assert not any("cache" in key for key in model.state_dict())


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("policy", ["within_topic", "mean_within_across"])
def test_tf_expression_shuffled_subset_zero_counts_and_dtype(case, sparse, policy):
    model, rna, atac, cells, tfs, batches, config = case
    config.tf_expression_scaling = policy
    rna.X = rna.X.astype(np.float64)
    rna.X[8, tfs] = 0
    if sparse:
        rna.X = sp.csr_matrix(rna.X)
    ids = [8, 2, 10, 0, 6]
    model.train()
    args = (rna, atac, cells, tfs, batches, config)
    actual = get_tf_expression("True", model, "cpu", loader(ids, True), *args)
    other = get_tf_expression("actual", model, "cpu", loader(list(reversed(ids))), *args)
    torch.testing.assert_close(actual, other)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert model.training and model.encoder_rna.training
    # Independent top-cell selection using absolute IDs.
    ids = np.sort(ids)
    x, a, _, lib, nc, _ = inputs(case, ids)
    model.eval()
    with torch.no_grad():
        theta = model.encode(x, a, lib[:, :1], lib[:, 1:], nc)[0].numpy()
    values = rna.X[ids][:, tfs]
    values = values.toarray() if sp.issparse(values) else values
    totals = values.sum(axis=1, keepdims=True)
    values = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0) * np.median(totals)
    top = np.argsort(theta, axis=0, kind="stable")[-config.cells_per_topic :]
    expected = np.stack([values[top[:, t]].mean(axis=0) for t in range(model.num_topics)])
    within = (expected - expected.min(axis=1, keepdims=True)) / (np.ptp(expected, axis=1, keepdims=True) + 1e-9)
    across = (expected - expected.min(axis=0, keepdims=True)) / (np.ptp(expected, axis=0, keepdims=True) + 1e-9)
    expected = (within + across) / 2 if policy == "mean_within_across" else within
    expected[expected < config.tf_expression_clamp] = 0
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6)


def test_latent_mode_and_invalid_mode(case):
    model, rna, atac, cells, tfs, batches, config = case
    args = (model, "cpu", loader([0, 1]), rna, atac, cells, tfs, batches, config)
    tf = get_tf_expression("latent", *args)
    assert tf.shape == (4, 3) and torch.isfinite(tf).all()
    with pytest.raises(ValueError, match="Unknown"):
        get_tf_expression("typo", *args)


def test_encoder_only_extraction_restores_mixed_modes(case, monkeypatch):
    model, rna, atac, cells, tfs, batches, _ = case
    model.train()
    model.encoder_rna.eval()
    modes = [m.training for m in model.modules()]

    def forbidden(*args, **kwargs):
        raise AssertionError("Latent extraction must not decode")

    monkeypatch.setattr(model, "forward", forbidden)
    theta, ids = get_latent_topics(
        model, "cpu", loader([8, 1, 10]), rna, atac, cells, tfs, batches, return_indices=True
    )
    assert ids.tolist() == [8, 1, 10]
    assert theta.shape == (3, 4)
    assert modes == [m.training for m in model.modules()]


def test_checkpoint_phase2_uses_the_current_binding(case, caplog):
    model = case[0]
    saved_act = model.tf_binding_matrix_activator.detach().clone()
    saved_rep = model.tf_binding_matrix_repressor.detach().clone()
    initialize_scdori_parameters(model, None, None, None, None, phase="grn")  # nothing given: keep the checkpoint's copy
    assert torch.equal(model.tf_binding_matrix_activator, saved_act)
    with caplog.at_level("WARNING"):
        initialize_scdori_parameters(model, None, None, saved_act.clone(), saved_rep.clone(), phase="grn")
    assert "differs from the loaded checkpoint" not in caplog.text
    new_act, new_rep = saved_act + 0.5, saved_rep - 0.5
    with caplog.at_level("WARNING"):
        initialize_scdori_parameters(model, None, None, new_act, new_rep, phase="grn")
    assert "differs from the loaded checkpoint" in caplog.text
    assert torch.equal(model.tf_binding_matrix_activator, new_act)
    assert torch.equal(model.tf_binding_matrix_repressor, new_rep)
    assert not model.tf_binding_matrix_activator.requires_grad
    with pytest.raises(ValueError, match="shape"):
        initialize_scdori_parameters(model, None, None, new_act[:-1], new_rep, phase="grn")


def test_checkpoint_phase2_freezes_fixed_priors(case):
    model = case[0]
    initialize_scdori_parameters(model, None, None, None, None, phase="grn")
    assert not model.tf_binding_matrix_activator.requires_grad
    assert not model.tf_binding_matrix_repressor.requires_grad
    assert not model.gene_peak_factor_fixed.requires_grad
    assert model.tf_gene_topic_activator_grn.requires_grad


def test_factor_cache_rejects_trainable_dependencies(case):
    with pytest.raises(ValueError, match="frozen"):
        build_grn_factors(case[0])


def test_regularization_has_gradient(case):
    model, *_, config = case
    loss = _regularization(model, config, grn=True) + _regularization(model, config, grn=False)
    loss.backward()
    assert torch.count_nonzero(model.tf_gene_topic_activator_grn.grad)
    assert torch.count_nonzero(model.gene_peak_factor_learnt.grad)


@pytest.mark.parametrize("mode", ["True", "latent"])
@pytest.mark.parametrize("count_mode", ["insertions", "fragments"])
@pytest.mark.parametrize("policy", ["within_topic", "mean_within_across"])
def test_training_equivalence_and_frozen_state(case, mode, count_mode, policy):
    model, rna, atac, cells, tfs, batches, config = case
    reference = copy.deepcopy(model)
    before = copy.deepcopy(model.state_dict())
    config.tf_expression_mode = mode
    config.tf_expression_scaling = policy
    atac.uns["scdori_atac_count_mode"] = count_mode
    if count_mode == "fragments":
        atac.X += 0.25  # Fractional metacell means must survive preparation.
    for cached, candidate in [(True, model), (False, reference)]:
        torch.manual_seed(99)
        config.cache_frozen_grn = cached
        train_model_grn(
            candidate,
            "cpu",
            loader([8, 2, 10, 0, 1, 3, 7, 9], True),
            loader([4, 5, 6, 11]),
            rna,
            atac,
            cells,
            tfs,
            batches,
            config,
        )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], atol=3e-5, rtol=3e-4)
        if not name.startswith(("tf_gene_topic_", "rna_grn_batch_")):
            assert torch.equal(value, before[name]), name
    assert not torch.equal(model.tf_gene_topic_activator_grn, before["tf_gene_topic_activator_grn"])


def test_incompatible_flags_rejected(case):
    model, rna, atac, cells, tfs, batches, config = case
    config.update_encoder_in_grn = True
    with pytest.raises(ValueError, match="cannot be combined"):
        train_model_grn(model, "cpu", loader([0, 1]), loader([2, 3]), rna, atac, cells, tfs, batches, config)


@pytest.mark.parametrize("phase,penalty", [("phase1", "l1"), ("phase1", "l2"), ("grn", "l1")])
def test_training_regularization_updates_weights_without_reconstruction(case, phase, penalty):
    model, rna, atac, cells, tfs, batches, config = case
    for name in vars(config):
        if name.startswith(("weight_", "l1_penalty_", "l2_penalty_")):
            setattr(config, name, 0.0)
    config.freeze_phase1_in_grn = False
    config.max_grn_epochs = config.max_scdori_epochs = 1
    config.weights_folder_scdori = config.weights_folder_grn
    config.tf_expression_mode = "latent"
    if phase == "phase1":
        setattr(config, penalty + "_penalty_topic_tf", 0.1)
        parameter = model.topic_tf_decoder
        train = train_scdori_phases
    else:
        config.l1_penalty_grn_activator = 0.1
        parameter = model.tf_gene_topic_activator_grn
        train = train_model_grn
    before = parameter.detach().clone()
    train(model, "cpu", loader([0, 1, 2, 3]), loader([4, 5, 6, 7]), rna, atac, cells, tfs, batches, config)
    assert parameter.abs().sum() < before.abs().sum()


def test_single_technical_batch_prediction_and_gradient_roundoff(case):
    # A constant batch offset is removed by training BatchNorm. Its exact
    # gradient is zero; Adam can magnify float32 noise in this null direction.
    # Compare the predictions and gradients themselves for this boundary case.
    model, rna, atac, cells, tfs, batches, _ = case
    freeze_phase1(model)
    cache = build_cell_cache(model, "cpu", rna, atac, cells, tfs, batches, 5)
    factors = build_grn_factors(model)
    model.rna_grn_batch_norm.train()
    ids = [8, 2, 10, 0]
    x, a, tf, lib, nc, batch = inputs(case, ids)
    topic_tf = torch.rand(4, 3)
    full = model(x, a, tf, topic_tf, lib[:, :1], lib[:, 1:], nc, batch, phase="grn")["mu_nb_rna_grn"]
    cached = forward_from_topics(model, cache.theta[ids], cache.library[ids], batch, topic_tf, factors)
    torch.testing.assert_close(full, cached, atol=2e-6, rtol=2e-5)
    parameters = [p for p in model.parameters() if p.requires_grad]
    a_grads = torch.autograd.grad(full.square().sum(), parameters)
    b_grads = torch.autograd.grad(cached.square().sum(), parameters)
    for a_grad, b_grad in zip(a_grads, b_grads, strict=True):
        torch.testing.assert_close(a_grad, b_grad, atol=2e-5, rtol=2e-3)


def test_sparse_cache_matches_dense_and_rebuilds_for_new_model(case):
    model, rna, atac, cells, tfs, batches, _ = case
    freeze_phase1(model)
    dense = build_cell_cache(model, "cpu", rna, atac, cells, tfs, batches, 4)
    rna.X, atac.X = sp.csr_matrix(rna.X), sp.csr_matrix(atac.X)
    sparse = build_cell_cache(model, "cpu", rna, atac, cells, tfs, batches, 4)
    torch.testing.assert_close(dense.theta, sparse.theta)
    torch.testing.assert_close(dense.auxiliary, sparse.auxiliary)
    with torch.no_grad():
        model.mu_theta.bias[0].add_(1)
    rebuilt = build_cell_cache(model, "cpu", rna, atac, cells, tfs, batches, 4)
    assert not torch.allclose(rebuilt.theta, dense.theta)


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("policy", ["within_topic", "mean_within_across"])
def test_zero_tf_float64_data_trains_without_nan(case, frozen, policy):
    model, rna, atac, cells, tfs, batches, config = case
    config.freeze_phase1_in_grn = frozen
    config.tf_expression_scaling = policy
    rna.X = rna.X.astype(np.float64)
    rna.X[0, tfs] = 0
    rna.X = sp.csr_matrix(rna.X)
    train_model_grn(
        model, "cpu", loader([0, 1, 2, 3, 4, 5, 6, 7]), loader([8, 9, 10, 11]), rna, atac, cells, tfs, batches, config
    )
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_resumed_checkpoint_priors_receive_no_gradients(case, tmp_path):
    model = case[0]
    initialize_scdori_parameters(
        model,
        model.gene_peak_factor_learnt.detach().clone(),
        model.gene_peak_factor_fixed.detach().clone(),
        model.tf_binding_matrix_activator.detach().clone(),
        model.tf_binding_matrix_repressor.detach().clone(),
    )
    path = tmp_path / "phase1.pth"
    torch.save(model.state_dict(), path)
    resumed = scDoRI(torch.device("cpu"), 7, 11, 3, 4, 2, 9, 5)
    resumed.load_state_dict(torch.load(path, weights_only=True))
    assert resumed.tf_binding_matrix_activator.requires_grad
    initialize_scdori_parameters(resumed, None, None, None, None, phase="grn")
    freeze_phase1(resumed)
    x, a, tf, lib, nc, batch = inputs(case, [0, 1, 2, 3])
    out = resumed(x, a, tf, torch.rand(4, 3, dtype=torch.float64), lib[:, :1], lib[:, 1:], nc, batch, phase="grn")
    assert out["mu_nb_rna_grn"].dtype == torch.float32
    out["mu_nb_rna_grn"].square().sum().backward()
    for name in ["gene_peak_factor_fixed", "tf_binding_matrix_activator", "tf_binding_matrix_repressor"]:
        assert getattr(resumed, name).grad is None
    assert resumed.tf_gene_topic_activator_grn.grad is not None


@pytest.mark.parametrize("policy, peak_score", [("within_topic", 1.0), ("mean_within_across", 0.5)])
def test_tf_expression_recomputed_when_same_model_receives_new_data(case, policy, peak_score):
    model, rna, atac, cells, tfs, batches, config = case
    config.tf_expression_scaling = policy
    freeze_phase1(model)
    args = ("True", model, "cpu", loader(list(range(12))), rna, atac, cells, tfs, batches, config)
    before = get_tf_expression(*args)
    rna.X[:, tfs] = [1, 100, 3]
    after = get_tf_expression(*args)
    # Identical expression across topics gives zero across-topic contrast.
    expected = torch.tensor([[0.0, peak_score, 0.0]]).expand(model.num_topics, -1)
    torch.testing.assert_close(after, expected)
    assert not torch.allclose(before, after)
