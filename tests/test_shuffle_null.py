"""grn_topic_selection="shuffle_null": topics are kept only if real cells give them more weight than shuffled cells."""

import json

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, TensorDataset

from scdori._core.config import TrainConfig
from scdori._core.models import scDoRI
from scdori._core.topic_selection import (
    apply_grn_topic_selection,
    grn_topic_ids,
    shuffle_null_topics,
    shuffle_within_rows,
)
from scdori._core.train_grn import train_model_grn

N, GENES, PEAKS, BLOCK = 300, 20, 12, 4


def reference_shuffle_within_cells(X, rng):
    """Reference shuffle: each cell's stored values get new random positions within the cell, one cell at a time."""
    X = sp.csr_matrix(X, copy=True); p = X.shape[1]  # noqa: E702
    for i in range(X.shape[0]):
        s, e = X.indptr[i], X.indptr[i + 1]
        X.indices[s:e] = rng.choice(p, e - s, replace=False)
    X.sort_indices(); return X  # noqa: E702


@pytest.fixture
def data():
    rng = np.random.default_rng(5)
    x = rng.poisson(1.0, (N, GENES)).astype(np.float32)
    x[:, :BLOCK] = rng.poisson(20.0, (N, BLOCK))  # real cells concentrate counts in the first gene block
    rna = ad.AnnData(sp.csr_matrix(x))
    atac = ad.AnnData(sp.csr_matrix(rng.poisson(2.0, (N, PEAKS)).astype(np.float32)),
                      uns={"scdori_atac_count_mode": "fragments"})
    return rna, atac, np.ones((N, 1), dtype=np.float32), np.array([0, 1]), np.ones((N, 1), dtype=np.float32)


def structured_model():
    """Topic 0 follows the gene block (structure); topic 1 follows counts outside it, which shuffling increases."""
    torch.manual_seed(0)
    model = scDoRI(torch.device("cpu"), GENES, PEAKS, 2, 3, 1, 4, 2, batch_norm=False)
    with torch.no_grad():
        for p in list(model.encoder_rna.parameters()) + list(model.encoder_atac.parameters()) + list(
                model.mu_theta.parameters()):
            p.zero_()
        first, second = model.encoder_rna[0], model.encoder_rna[4]
        first.weight[0, :BLOCK] = 1.0  # unit 0: counts in the block
        first.weight[1, BLOCK:GENES] = 1.0  # unit 1: counts outside the block
        second.weight[0, 0] = second.weight[1, 1] = 1.0
        model.mu_theta.weight[0, 0] = model.mu_theta.weight[1, 1] = 0.1
    return model


def test_shuffle_matches_the_reference_null():
    x = sp.random(40, 30, density=0.3, random_state=1, format="csr", dtype=np.float32)
    ours = shuffle_within_rows(x, np.random.default_rng(7))
    ref = reference_shuffle_within_cells(x, np.random.default_rng(7))
    assert (ours != ref).nnz == 0
    np.testing.assert_allclose(np.asarray(ours.sum(1)).ravel(), np.asarray(x.sum(1)).ravel(), rtol=1e-6)
    for row in range(x.shape[0]):  # same values per cell, other positions
        np.testing.assert_array_equal(np.sort(ours[row].data), np.sort(x[row].data))
    dense = shuffle_within_rows(x.toarray(), np.random.default_rng(7))
    assert (dense != ref).nnz == 0


def test_structured_topic_kept_and_shuffle_topic_dropped(data):
    rna, atac, cells, tfs, batches = data
    model = structured_model()
    ids = np.arange(0, N, 2)  # training cells
    from scdori._core.evaluation import get_latent_topics

    theta = get_latent_topics(model, "cpu", DataLoader(TensorDataset(torch.as_tensor(ids)), batch_size=64),
                              rna, atac, cells, tfs, batches)
    result = shuffle_null_topics(model, "cpu", rna, atac, cells, tfs, batches, ids, theta, TrainConfig())
    assert result["keep"][0] and not result["keep"][1]
    assert result["observed"][0] > result["cutoff"][0] and result["observed"][1] <= result["cutoff"][1]
    np.testing.assert_allclose(result["observed"], np.quantile(theta, 0.99, axis=0))
    assert result["shuffled_cells"] <= 3 * 100
    again = shuffle_null_topics(model, "cpu", rna, atac, cells, tfs, batches, ids, theta, TrainConfig())
    for key in ("cutoff", "observed", "keep"):
        np.testing.assert_array_equal(again[key], result[key])  # deterministic with the seed
    other = shuffle_null_topics(model, "cpu", rna, atac, cells, tfs, batches, ids, theta, TrainConfig(grn_null_seed=7))
    assert not np.array_equal(other["cutoff"], result["cutoff"])
    np.testing.assert_array_equal(other["keep"][:2], result["keep"][:2])


def run_phase2(data, tmp_path, **settings):
    rna, atac, cells, tfs, batches = data
    model = structured_model()
    config = TrainConfig(max_grn_epochs=1, cells_per_topic=10, weights_folder_grn=str(tmp_path), **settings)
    torch.manual_seed(1)
    before = torch.get_rng_state()
    loader = DataLoader(TensorDataset(torch.arange(0, N, 2)), batch_size=50, shuffle=True)
    train_model_grn(model, "cpu", loader, DataLoader(TensorDataset(torch.arange(1, N, 2)), batch_size=50),
                    rna, atac, cells, tfs, batches, config)
    return model, json.loads((tmp_path / "grn_topics.json").read_text()), before


def test_phase2_records_and_applies_the_rule(data, tmp_path, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="scdori._core.topic_selection"):
        model, record, _ = run_phase2(data, tmp_path, grn_topic_selection="shuffle_null")
    assert record["selection"] == "shuffle_null" and record["source"] == "shuffled training cells"
    assert 1 in record["excluded_topics"] and 0 in record["grn_topics"]
    assert grn_topic_ids(model) == record["grn_topics"]
    per_topic = {row["topic"]: row for row in record["per_topic"]}
    assert per_topic[0]["kept"] and not per_topic[1]["kept"]
    assert record["null_rule"]["grn_null_top_cells"] == 100 and record["null_rule"]["grn_null_shuffles"] == 5
    assert record["dead_max_theta"] is None
    assert "max_excluded_theta_mass_train_cells" in record
    assert not model.tf_gene_topic_activator_grn[1].any()  # excluded topic: GRN weights stay zero
    assert any("Topic 1 dropped" in r.getMessage() for r in caplog.records)


def test_live_and_all_do_not_run_the_shuffle(data, tmp_path):
    def forbidden():
        pytest.fail("the shuffled-cell test must not run for this selection")

    theta = np.full((50, 3), 1 / 3)
    for mode, expected in (("all", [0, 1, 2]), ("live", [0, 1, 2])):
        model = structured_model()
        record = apply_grn_topic_selection(model, TrainConfig(grn_topic_selection=mode), lambda: theta, forbidden)
        assert record["grn_topics"] == expected and "per_topic" not in record
    model = structured_model()
    with pytest.raises(ValueError, match="shuffle_null"):
        # "shuffle_null" needs the shuffled-cell test
        apply_grn_topic_selection(model, TrainConfig(grn_topic_selection="shuffle_null"), lambda: theta)


def test_shuffle_null_is_deterministic_and_leaves_the_phase2_random_stream(data, tmp_path):
    model_a, record_a, _ = run_phase2(data, tmp_path / "a", grn_topic_selection="shuffle_null")
    model_b, record_b, _ = run_phase2(data, tmp_path / "b", grn_topic_selection="shuffle_null")
    assert record_a["selection"] == "shuffle_null" and record_a == record_b
    # The same topics chosen explicitly give bitwise the same phase 2: the shuffled-cell test draws no
    # numbers from the random stream that phase 2 uses.
    model_c, record_c, _ = run_phase2(data, tmp_path / "c", grn_topic_selection="explicit",
                                      grn_topics=record_a["grn_topics"])
    assert record_c["grn_topics"] == record_a["grn_topics"]
    for name, value in model_a.state_dict().items():
        assert torch.equal(value, model_b.state_dict()[name]), name
        assert torch.equal(value, model_c.state_dict()[name]), name
