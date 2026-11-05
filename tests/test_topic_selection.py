import copy
import json

import anndata as ad
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scdori._core.config import TrainConfig
from scdori._core.dataloader import create_minibatch
from scdori._core.downstream import (
    compute_atac_grn_activator_with_significance,
    compute_atac_grn_repressor_with_significance,
    compute_significant_grn,
    get_top_activators_per_topic,
)
from scdori._core.evaluation import get_latent_topics
from scdori._core.frozen_grn import (
    build_cell_cache,
    build_grn_factors,
    estimate_frozen_cache_bytes,
    forward_from_topics,
    freeze_phase1,
)
from scdori._core.models import scDoRI
from scdori._core.topic_selection import (
    GRN_TOPICS_FILE,
    collapsed_topics,
    grn_topic_ids,
    load_grn_topics,
    max_excluded_theta_mass,
    resolve_grn_topics,
    set_grn_topics,
)
from scdori._core.train_grn import get_tf_expression, train_model_grn

DEAD = [1, 4]
LIVE = [0, 2, 3, 5]


@pytest.fixture
def case(tmp_path):
    """Six topics; topics 1 and 4 collapsed by a strongly negative mu_theta bias."""
    torch.manual_seed(11)
    rng = np.random.default_rng(11)
    n = 24
    model = scDoRI(torch.device("cpu"), 7, 11, 3, 6, 2, 9, 5)
    with torch.no_grad():
        model.tf_binding_matrix_repressor.mul_(-1)
        model.mu_theta.bias[DEAD] = -40.0
    rna = ad.AnnData(rng.poisson(3, (n, 7)).astype(np.float32))
    atac = ad.AnnData(rng.poisson(3, (n, 11)).astype(np.float32))
    cells = np.ones((n, 1), dtype=np.float32)
    tfs = np.array([1, 3, 5])
    batches = np.eye(2, dtype=np.float32)[np.arange(n) % 2]
    config = TrainConfig(
        freeze_phase1_in_grn=True,
        grn_topic_selection="all",
        max_grn_epochs=2,
        cells_per_topic=4,
        batch_size_cell_prediction=8,
        weights_folder_grn=str(tmp_path / "grn"),
    )
    return model, rna, atac, cells, tfs, batches, config


def loader(ids, shuffle=False):
    return DataLoader(TensorDataset(torch.tensor(ids)), batch_size=6, shuffle=shuffle)


def gate(case, model=None):
    m, rna, atac, cells, tfs, batches, config = case
    return get_tf_expression("True", model or m, "cpu", loader(range(16)), rna, atac, cells, tfs, batches, config)


def test_selection_helpers_and_config(case):
    model, *_, config = case
    theta = np.full((5, 4), 0.25)
    theta[:, 3] = 1e-9
    np.testing.assert_array_equal(collapsed_topics(theta), [3])
    assert max_excluded_theta_mass(theta, [3]) == pytest.approx(1e-9)
    assert resolve_grn_topics(model, config) is None  # explicit all-topic reference
    config.grn_topic_selection = "explicit"
    config.grn_topics = [5, 0]
    np.testing.assert_array_equal(resolve_grn_topics(model, config), [5, 0])
    set_grn_topics(model, [5, 0])
    assert grn_topic_ids(model) == [0, 5]
    set_grn_topics(model, range(6))
    assert model.grn_topics is None  # all topics: reference path
    with pytest.raises(ValueError, match="unique"):
        set_grn_topics(model, [0, 0])
    config.grn_topic_selection = "bogus"
    with pytest.raises(ValueError, match="Unknown grn_topic_selection"):
        resolve_grn_topics(model, config)


@pytest.mark.parametrize("policy", ["within_topic", "mean_within_across"])
def test_pruned_factors_gate_and_forward(case, policy):
    model, rna, atac, cells, tfs, batches, config = case
    config.tf_expression_scaling = policy
    freeze_phase1(model)
    full_factors = build_grn_factors(model)
    full_gate = gate(case)
    theta = torch.as_tensor(get_latent_topics(model, "cpu", loader(range(24)), rna, atac, cells, tfs, batches))
    assert theta[:, DEAD].max() < 1e-15
    np.testing.assert_array_equal(collapsed_topics(theta.numpy()), DEAD)
    lib = torch.ones((24, 1))
    batch = torch.as_tensor(batches)
    model.rna_grn_batch_norm.eval()
    full = forward_from_topics(model, theta, lib, batch, full_gate, full_factors)

    set_grn_topics(model, LIVE)
    factors = build_grn_factors(model)
    for pruned, reference in zip(factors, full_factors, strict=True):
        torch.testing.assert_close(pruned, reference[LIVE], rtol=0, atol=0)
    pruned_gate = gate(case)
    torch.testing.assert_close(pruned_gate[LIVE], full_gate[LIVE], rtol=0, atol=0)
    assert (pruned_gate[DEAD] == 0).all()
    pruned = forward_from_topics(model, theta, lib, batch, pruned_gate, factors)
    torch.testing.assert_close(pruned, full, rtol=1e-6, atol=0)
    estimate = estimate_frozen_cache_bytes(model, 24)
    assert estimate["device_grn_factors"] == sum(t.numel() * t.element_size() for t in factors)

    # Cached and full-forward GRN predictions agree under the same selection.
    cache = build_cell_cache(model, "cpu", rna, atac, cells, tfs, batches, 8)
    x, tf, libraries, nc, b = create_minibatch("cpu", torch.arange(24), rna, atac, cells, tfs, batches)
    with torch.no_grad():
        out = model(x[:, :7], x[:, 7:], tf, pruned_gate, libraries[:, :1], libraries[:, 1:], nc, b, phase="grn")
        cached = forward_from_topics(model, cache.theta, cache.library, b, pruned_gate, factors)
    torch.testing.assert_close(cached, out["mu_nb_rna_grn"], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("cached", [True, False])
def test_live_training_matches_all_topics_on_live_slices(case, tmp_path, cached):
    model, rna, atac, cells, tfs, batches, config = case
    config.cache_frozen_grn = cached
    runs = {}
    for mode in ("all", "live"):
        torch.manual_seed(3)
        m = copy.deepcopy(model)
        c = copy.deepcopy(config)
        c.grn_topic_selection = mode
        c.weights_folder_grn = str(tmp_path / mode)
        shuffled = DataLoader(
            TensorDataset(torch.arange(16)), batch_size=6, shuffle=True, generator=torch.Generator().manual_seed(1)
        )
        runs[mode] = train_model_grn(m, "cpu", shuffled, loader(range(16, 24)), rna, atac, cells, tfs, batches, c)
    record = json.loads((tmp_path / "live" / GRN_TOPICS_FILE).read_text())
    assert record["excluded_topics"] == DEAD and record["grn_topics"] == LIVE
    assert record["max_excluded_theta_mass_train_cells"] < 1e-15
    assert json.loads((tmp_path / "all" / GRN_TOPICS_FILE).read_text())["excluded_topics"] == []
    live, full = runs["live"], runs["all"]
    for name in ("tf_gene_topic_activator_grn", "tf_gene_topic_repressor_grn"):
        torch.testing.assert_close(getattr(live, name)[LIVE], getattr(full, name)[LIVE], rtol=1e-4, atol=1e-6)
        assert (getattr(live, name)[DEAD] == 0).all()


def test_checkpoint_selection_roundtrip(case, tmp_path):
    model, *_, config = case
    set_grn_topics(model, LIVE)
    folder = tmp_path / "grn"
    folder.mkdir()
    (folder / GRN_TOPICS_FILE).write_text(json.dumps({"format": 1, "num_topics": 6, "grn_topics": LIVE}))
    fresh = copy.deepcopy(model)
    set_grn_topics(fresh, None)
    fresh.load_state_dict(model.state_dict())  # the state_dict format is unchanged
    load_grn_topics(fresh, folder)
    assert grn_topic_ids(fresh) == LIVE
    assert load_grn_topics(fresh, tmp_path / "missing") is None and fresh.grn_topics is None
    (folder / GRN_TOPICS_FILE).write_text(json.dumps({"format": 1, "num_topics": 7, "grn_topics": LIVE}))
    with pytest.raises(ValueError, match="model has 6"):
        load_grn_topics(fresh, folder)


def test_permutations_skip_topics_with_seed_identity(case, tmp_path):
    model = case[0]
    full_act = compute_atac_grn_activator_with_significance(model, "cpu", 0.1, tmp_path / "a", 20, seed=7)
    full_rep = compute_atac_grn_repressor_with_significance(model, "cpu", 0.1, tmp_path / "a", 20, seed=7)
    set_grn_topics(model, LIVE)  # default topics follow the model's GRN selection
    act = compute_atac_grn_activator_with_significance(model, "cpu", 0.1, tmp_path / "b", 20, seed=7)
    rep = compute_atac_grn_repressor_with_significance(model, "cpu", 0.1, tmp_path / "b", 20, seed=7)
    assert act.shape == full_act.shape == (6, 3, 7)
    np.testing.assert_array_equal(act[LIVE], full_act[LIVE])
    np.testing.assert_array_equal(rep[LIVE], full_rep[LIVE])
    assert not act[DEAD].any() and not rep[DEAD].any()
    explicit = compute_atac_grn_activator_with_significance(model, "cpu", 0.1, tmp_path / "c", 20, topics=[3], seed=7)
    np.testing.assert_array_equal(explicit[3], full_act[3])

    # Final GRNs keep shape K; excluded topics are zero and rankings work with gaps in topic IDs.
    with torch.no_grad():
        model.tf_gene_topic_activator_grn[DEAD] = 0
        model.tf_gene_topic_repressor_grn[DEAD] = 0
    topic_gate = gate(case)
    grn_act, grn_rep = compute_significant_grn(model, "cpu", 0.1, 0.1, topic_gate, tmp_path / "b")
    assert not grn_act[DEAD].any() and not grn_rep[DEAD].any()
    import matplotlib

    matplotlib.use("Agg")
    table, _ = get_top_activators_per_topic(grn_act, ["a", "b", "c"], None, selected_topics=LIVE, top_k=2)
    assert table.index.tolist() == [f"Topic_{k}" for k in LIVE]


def test_pruning_rejected_before_mutation_for_nonfrozen_phase1(case):
    from scdori._core.topic_selection import apply_grn_topic_selection

    model, *_, config = case
    config.grn_topic_selection = "live"
    weights = model.tf_gene_topic_activator_grn.detach().clone()
    for frozen, update in [(False, False), (True, True)]:
        config.freeze_phase1_in_grn = frozen
        config.update_encoder_in_grn = update
        with pytest.raises(ValueError, match="fully frozen"):
            apply_grn_topic_selection(model, config)
        assert model.grn_topics is None
        torch.testing.assert_close(model.tf_gene_topic_activator_grn, weights, rtol=0, atol=0)


def test_subset_sampler_population_and_generator_preserved():
    from torch.utils.data import SubsetRandomSampler

    from scdori._core.topic_selection import sequential_loader

    generator = torch.Generator().manual_seed(12)
    state = generator.get_state().clone()
    train = DataLoader(
        TensorDataset(torch.arange(20)), batch_size=3, sampler=SubsetRandomSampler([8, 2, 6], generator=generator)
    )
    ids = torch.cat([batch[0] for batch in sequential_loader(train)]).tolist()
    assert ids == [8, 2, 6]
    assert torch.equal(state, generator.get_state())


@pytest.mark.parametrize("topics", [[0.5, 1.5], [True, False], ["1", "2"]])
def test_topic_ids_cannot_be_silently_truncated(case, topics):
    with pytest.raises(ValueError, match="integer"):
        set_grn_topics(case[0], topics)


def test_live_file_requires_checkpoint_and_training_validation(case, tmp_path):
    from scdori._core.diagnostics import _sha256

    model, *_, config = case
    checkpoint = tmp_path / "phase1.pth"
    torch.save(model.state_dict(), checkpoint)
    record = tmp_path / "live.json"
    record.write_text(
        json.dumps({"format": 1, "num_topics": 6, "live_topics": LIVE, "phase1_checkpoint_sha256": _sha256(checkpoint)})
    )
    config.grn_topic_selection = "live"
    config.grn_live_topics_file = str(record)
    with pytest.raises(ValueError, match="phase1_checkpoint"):
        resolve_grn_topics(model, config)
    config.phase1_checkpoint = str(checkpoint)
    with pytest.raises(ValueError, match="training-cell theta"):
        resolve_grn_topics(model, config)
    with pytest.raises(ValueError, match="not collapsed"):
        resolve_grn_topics(model, config, np.full((10, 6), 1 / 6))
    theta = np.zeros((10, 6))
    theta[:, LIVE] = 0.25
    np.testing.assert_array_equal(resolve_grn_topics(model, config, theta), LIVE)
    checkpoint.write_bytes(b"different")
    with pytest.raises(ValueError, match="different phase-1"):
        resolve_grn_topics(model, config, theta)


def test_checkpoint_specific_selection_wins_over_run_directory(case, tmp_path):
    from scdori._core.data_io import save_model_weights
    from scdori._core.downstream import load_best_model
    from scdori._core.topic_selection import apply_grn_topic_selection, write_grn_topics

    model, *_, config = case
    config.grn_topic_selection = "explicit"
    config.grn_topics = LIVE
    record = apply_grn_topic_selection(model, config)
    save_model_weights(model, tmp_path, "scdori_best_eval")
    write_grn_topics(tmp_path, {**record, "grn_topics": list(range(model.num_topics)), "excluded_topics": []})
    set_grn_topics(model, None)
    load_best_model(model, tmp_path / "best_scdori_best_eval.pth", "cpu")
    assert grn_topic_ids(model) == LIVE
    set_grn_topics(model, None)
    load_grn_topics(model, tmp_path / "best_scdori_best_eval.pth")
    assert grn_topic_ids(model) == LIVE


def test_new_defaults_select_live_topics_after_phase1(case):
    model, _, _, _, _, _, _ = case
    config = TrainConfig()
    assert config.freeze_phase1_in_grn and config.cache_frozen_grn
    assert config.grn_topic_selection == "live"
    theta = np.full((10, 6), .25)
    theta[:, DEAD] = 1e-12
    before = {k: v.clone() for k, v in model.state_dict().items()}
    np.testing.assert_array_equal(resolve_grn_topics(model, config, theta), LIVE)
    for key, value in before.items():
        torch.testing.assert_close(value, model.state_dict()[key])
