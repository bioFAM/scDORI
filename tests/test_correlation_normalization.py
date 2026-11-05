import json

import anndata as ad
import numpy as np
import pytest

from scdori.configuration import load_notebook_config
from scdori.pp.correlation import metacell_correlation_matrices
from scdori.pp.metacells import aggregate_metacells


@pytest.fixture
def metacells():
    rng = np.random.default_rng(0)
    rna = ad.AnnData(rng.poisson(3, size=(6, 8)).astype(np.float32))
    rna.var_names = [f"g{i}" for i in range(8)]
    rna.var["gene_type"] = ["TF", "gene", "TF", "gene", "gene", "TF", "gene", "gene"]
    atac = ad.AnnData(rng.poisson(2, size=(6, 10)).astype(np.float32))
    atac.var_names = [f"chr1:{i}-{i + 1}" for i in range(10)]
    return rna, atac


def test_none_reproduces_the_previous_inputs(metacells):
    rna, atac = metacells
    peaks = list(atac.var_names[[7, 2, 5]])
    rna_x, atac_x = metacell_correlation_matrices(rna, atac, peaks, "none")
    np.testing.assert_array_equal(rna_x, rna.X[:, rna.var["gene_type"] == "TF"])
    np.testing.assert_array_equal(atac_x, atac[:, peaks].X)


def test_log_normalized_removes_metacell_depth(metacells):
    rna, atac = metacells
    deeper_rna, deeper_atac = rna.copy(), atac.copy()
    deeper_rna.X[2] *= 5
    deeper_atac.X[4] *= 3
    peaks = list(atac.var_names[:4])
    a = metacell_correlation_matrices(rna, atac, peaks, "log_normalized")
    b = metacell_correlation_matrices(deeper_rna, deeper_atac, peaks, "log_normalized")
    for x, y in zip(a, b, strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-6)
    # Totals use every gene and every metacell peak, not only TFs or the selected peaks.
    tfs = rna.var["gene_type"].to_numpy() == "TF"
    expected = np.log1p(rna.X[:, tfs] / rna.X.sum(axis=1, keepdims=True) * 1e4)
    np.testing.assert_allclose(a[0], expected, rtol=1e-5)


def test_empty_metacell_stays_zero_and_bad_requests_fail(metacells):
    rna, atac = metacells
    atac.X[1] = 0
    _, atac_x = metacell_correlation_matrices(rna, atac, list(atac.var_names), "log_normalized")
    assert (atac_x[1] == 0).all() and np.isfinite(atac_x).all()
    with pytest.raises(ValueError, match="correlation_normalization"):
        metacell_correlation_matrices(rna, atac, list(atac.var_names), "zscore")
    with pytest.raises(ValueError, match="selected peak"):
        metacell_correlation_matrices(rna, atac, ["chr9:1-2"])


def test_window_size_default_follows_species_and_explicit_values_win(tmp_path):
    from scdori.pp.config import PreprocessingConfig

    assert PreprocessingConfig().window_size == 80_000
    assert PreprocessingConfig(species="mouse").window_size == 80_000
    assert PreprocessingConfig(species="Human").window_size == 150_000
    assert PreprocessingConfig(species="zebrafish").window_size == 80_000
    assert PreprocessingConfig(species="human", window_size=50_000).window_size == 50_000

    def loaded(**section):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"preprocessing": section}))
        return load_notebook_config(path).preprocessing

    assert loaded().window_size == 80_000
    assert loaded(species="mouse", genome_assembly="mm10").window_size == 80_000
    human = loaded(species="human", genome_assembly="hg38")
    assert human.window_size == 150_000
    assert human.peak_distance_min_cutoff == pytest.approx(np.exp(-150_000 / human.peak_distance_scaling_factor))
    assert loaded(species="human", genome_assembly="hg38", window_size=60_000).window_size == 60_000
    assert loaded(species="human", genome_assembly="hg38", window_size=None).window_size == 150_000
    with pytest.raises(ValueError, match="window_size"):
        loaded(window_size="80kb")


def test_config_accepts_the_option(tmp_path):
    path = tmp_path / "scdori_config.json"
    path.write_text(json.dumps({"preprocessing": {"correlation_normalization": "log_normalized"}}))
    assert load_notebook_config(path).preprocessing.correlation_normalization == "log_normalized"
    path.write_text(json.dumps({"preprocessing": {}}))
    assert load_notebook_config(path).preprocessing.correlation_normalization == "mean_log"


def test_log_normalized_is_the_same_from_summed_counts_as_from_means(metacells):
    rna, atac = metacells
    peaks = list(atac.var_names)
    n_cells = np.array([3, 40, 7, 100, 12, 500])[:, None]
    from_means = metacell_correlation_matrices(rna, atac, peaks, "log_normalized")
    summed_rna, summed_atac = rna.copy(), atac.copy()
    summed_rna.X = rna.X.astype(np.float64) * n_cells
    summed_atac.X = atac.X.astype(np.float64) * n_cells
    from_sums = metacell_correlation_matrices(summed_rna, summed_atac, peaks, "log_normalized")
    np.testing.assert_allclose(from_sums[0], from_means[0])
    np.testing.assert_allclose(from_sums[1], from_means[1])


def test_mean_log_is_the_default_and_averages_each_cells_log_normalized_counts():
    counts = np.array([[1, 3, 0], [2, 0, 2], [0, 5, 1], [4, 4, 0]], dtype=np.float32)
    rna = ad.AnnData(counts.copy())
    rna.obs_names = [f"c{i}" for i in range(4)]
    rna.var_names = ["a", "b", "c"]
    rna.var["gene_type"] = ["TF", "gene", "TF"]
    rna.obs["leiden"] = ["x", "x", "y", "y"]
    atac = ad.AnnData(np.array([[1, 0], [3, 2], [0, 4], [2, 2]], dtype=np.float32))
    atac.obs_names = rna.obs_names
    atac.var_names = ["chr1:1-2", "chr1:3-4"]
    rna_metacell, atac_metacell = aggregate_metacells(rna, atac, "leiden", "fragments")
    totals = counts.sum(axis=1, keepdims=True)
    scaled = np.log1p(counts / totals * np.median(totals))
    np.testing.assert_allclose(rna_metacell.layers["mean_log"], [scaled[:2].mean(axis=0), scaled[2:].mean(axis=0)], rtol=1e-5)
    np.testing.assert_allclose(rna_metacell.X, [counts[:2].mean(axis=0), counts[2:].mean(axis=0)])
    np.testing.assert_array_equal(rna.X, counts)  # the input stays raw
    rna_x, atac_x = metacell_correlation_matrices(rna_metacell, atac_metacell, list(atac.var_names))
    np.testing.assert_array_equal(rna_x, rna_metacell.layers["mean_log"][:, [0, 2]])
    np.testing.assert_array_equal(atac_x, atac_metacell.X)
    assert load_notebook_config(None).preprocessing.correlation_normalization == "mean_log"


def test_mean_log_needs_the_layer(metacells):
    rna, atac = metacells
    with pytest.raises(ValueError, match="mean_log layer"):
        metacell_correlation_matrices(rna, atac, list(atac.var_names))
