import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scanpy.external as sce
import scipy.sparse as sp
import torch

from scdori._core.dataloader import create_minibatch
from scdori._counts import as_fragment_counts
from scdori.pp.metacells import create_metacells
from scdori.pp.peak_selection import keep_promoters_and_select_hv_peaks, select_highly_variable_peaks_by_std


def make_data(sparse=False):
    obs = pd.DataFrame({"sample": ["s1"] * 4, "leiden": ["a", "a", "b", "b"]}, index=list("wxyz"))
    rna = np.array([[0, 4, 12], [8, 0, 4], [2, 6, 0], [0, 4, 8]], dtype=np.float32)
    atac = np.array([[0, 1, 4], [1, 2, 2], [4, 0, 3], [2, 1, 0]], dtype=np.float32)
    if sparse:
        rna, atac = sp.csr_matrix(rna), sp.csr_matrix(atac)
    return ad.AnnData(rna, obs=obs.copy()), ad.AnnData(atac, obs=obs.copy())


@pytest.fixture
def fixed_clusters(monkeypatch):
    # Exercise real normalization/logging and aggregation; isolate expensive
    # clustering algorithms so expected cluster means are known exactly.
    monkeypatch.setattr(sc.pp, "pca", lambda *a, **k: None)
    monkeypatch.setattr(sce.pp, "harmony_integrate", lambda *a, **k: None)
    monkeypatch.setattr(sc.pp, "neighbors", lambda *a, **k: None)
    monkeypatch.setattr(sc.tl, "leiden", lambda *a, **k: None)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("mode", ["insertions", "fragments"])
def test_metacells_preserve_raw_rna_and_convert_atac_once(sparse, mode, fixed_clusters, tmp_path):
    rna, atac = make_data(sparse)
    original_rna = rna.X.toarray() if sparse else rna.X.copy()
    original_atac = atac.X.toarray() if sparse else atac.X.copy()
    mrna, matac = create_metacells(rna, atac, atac_count_mode=mode)
    np.testing.assert_allclose(mrna.X, original_rna.reshape(2, 2, 3).mean(axis=1))
    expected_atac = (original_atac + 1) // 2 if mode == "insertions" else original_atac
    np.testing.assert_allclose(matac.X, expected_atac.reshape(2, 2, 3).mean(axis=1))
    np.testing.assert_array_equal(rna.X.toarray() if sparse else rna.X, original_rna)
    np.testing.assert_array_equal(atac.X.toarray() if sparse else atac.X, original_atac)
    assert mrna.obs.num_cells.tolist() == [2, 2]
    assert matac.uns["scdori_atac_count_mode"] == "fragments"
    path = tmp_path / "atac.h5ad"
    matac.write_h5ad(path)
    loaded = ad.read_h5ad(path)
    x, _, lib, _, _ = create_minibatch(
        "cpu", torch.tensor([1, 0]), mrna, loaded, np.full((2, 1), 2), [0], np.ones((2, 1))
    )
    np.testing.assert_allclose(x[:, 3:], matac.X[[1, 0]])
    np.testing.assert_allclose(lib[:, 1].exp(), matac.X[[1, 0]].sum(axis=1), rtol=1e-6)


@pytest.mark.parametrize("sparse", [False, True])
def test_fragment_peak_selection_and_deterministic_order(sparse):
    _, atac = make_data(sparse)
    atac.var_names = ["z_promoter", "y", "a"]
    atac.var["is_promoter"] = [True, False, False]
    atac.uns["scdori_atac_count_mode"] = "fragments"
    selected = select_highly_variable_peaks_by_std(atac, 3)
    raw = atac.X.toarray() if sparse else atac.X
    np.testing.assert_allclose(selected.var.std_cluster, raw.reshape(2, 2, 3).mean(axis=1).std(axis=0))
    final = keep_promoters_and_select_hv_peaks(atac, 2)
    expected = [name for name in atac.var_names if name in set(final.var_names)]
    assert final.var_names.tolist() == expected
    assert "z_promoter" in final.var_names
    assert final.uns["scdori_atac_count_mode"] == "fragments"


@pytest.mark.parametrize("function", [select_highly_variable_peaks_by_std, keep_promoters_and_select_hv_peaks])
def test_explicit_fragment_mode_propagates_to_training(function):
    rna, atac = make_data()
    result = function(atac, 3, atac_count_mode="fragments")
    assert result.uns["scdori_atac_count_mode"] == "fragments"
    x, *_ = create_minibatch("cpu", torch.tensor([0, 1]), rna, result, np.ones((4, 1)), [0], np.ones((4, 1)))
    np.testing.assert_array_equal(x[:, 3:], result.X[:2])


def test_untagged_training_preserves_legacy_conversion_and_override():
    rna, atac = make_data()
    args = ("cpu", torch.tensor([0, 1]), rna, atac, np.ones((4, 1)), [0], np.ones((4, 1)))
    x, *_ = create_minibatch(*args)
    np.testing.assert_array_equal(x[:, 3:], (atac.X[:2] + 1) // 2)
    x, *_ = create_minibatch(*args, atac_count_mode="fragments")
    np.testing.assert_array_equal(x[:, 3:], atac.X[:2])
    with pytest.raises(ValueError, match="atac_count_mode"):
        create_minibatch(*args, atac_count_mode="typo")


def test_metacell_failure_does_not_replace_input_counts(monkeypatch, fixed_clusters):
    rna, atac = make_data()
    before = rna.X.copy()

    def fail(*args, **kwargs):
        raise RuntimeError("clustering failed")

    monkeypatch.setattr(sce.pp, "harmony_integrate", fail)
    with pytest.raises(RuntimeError, match="clustering failed"):
        create_metacells(rna, atac)
    np.testing.assert_array_equal(rna.X, before)


def test_missing_atac_cells_rejected_before_clustering(fixed_clusters):
    rna, atac = make_data()
    with pytest.raises(ValueError, match="ATAC"):
        create_metacells(rna, atac[:3].copy())


def test_sparse_conversion_keeps_sparse_and_handles_duplicate_entries():
    values = sp.csr_matrix((np.array([1, 1, 255], dtype=np.uint8), [0, 0, 1], [0, 3]), shape=(1, 3))
    converted = as_fragment_counts(values, "insertions")
    assert sp.issparse(converted)
    np.testing.assert_array_equal(converted.toarray(), [[1, 128, 0]])
    assert values.nnz == 3
    np.testing.assert_array_equal(as_fragment_counts(np.array([[255]], dtype=np.uint8), "insertions"), [[128]])
