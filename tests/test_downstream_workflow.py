import numpy as np
import pandas as pd
import pytest
import torch

from scdori._core.downstream import (
    compute_activator_tf_activity_per_cell,
    compute_atac_grn_activator_with_significance,
    compute_atac_grn_repressor_with_significance,
    compute_repressor_tf_activity_per_cell,
    compute_significant_grn,
    compute_topic_gene_matrix,
    save_regulons,
)
from scdori._core.interpretation import gene_importance_scores
from scdori._core.models import scDoRI


@pytest.mark.parametrize(
    "function,sign", [(compute_activator_tf_activity_per_cell, 1), (compute_repressor_tf_activity_per_cell, -1)]
)
def test_cell_activity_honors_selected_topics_and_zero_variance(function, sign):
    grn = sign * np.array([[[4.0], [0.0]], [[0.0], [7.0]]])
    theta = np.array([[0.8, 0.2], [0.1, 0.9]])
    scores = function(grn, ["A", "B"], theta, selected_topics=[1], zscore=False)
    np.testing.assert_allclose(scores[:, 0], 0)
    np.testing.assert_allclose(scores[:, 1], theta[:, 1])
    scaled = function(grn, ["A", "B"], theta, selected_topics=[1], zscore=True)
    assert np.isfinite(scaled).all()
    np.testing.assert_allclose(scaled[:, 0], 0)
    with pytest.raises(ValueError, match="nonempty"):
        function(grn, ["A", "B"], theta, selected_topics=[])


def test_gene_subset_changes_reference_and_preserves_order():
    torch.manual_seed(17)
    model = scDoRI(torch.device("cpu"), 7, 11, 3, 4, 2, 9, 5)
    raw = pd.DataFrame(model.topic_gene_matrix().detach().numpy())
    expected = gene_importance_scores(raw.loc[[3, 1]])["downstream_softmax"]
    actual = compute_topic_gene_matrix(model, "cpu", selected_topics=[3, 1])
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    singleton = compute_topic_gene_matrix(model, "cpu", selected_topics=[2])
    np.testing.assert_allclose(singleton, np.full((1, 7), 1 / 7), atol=1e-7)


def test_permutation_tails_zero_null_and_combined_regulons(tmp_path, monkeypatch):
    model = scDoRI(torch.device("cpu"), 3, 3, 1, 2, 1, 5, 3)
    with torch.no_grad():
        model.topic_peak_decoder.copy_(torch.log(torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])))
        model.gene_peak_factor_fixed.copy_(torch.eye(3))
        model.gene_peak_factor_learnt.fill_(1)
        model.tf_binding_matrix_activator.copy_(torch.tensor([[1.0], [0.0], [0.0]]))
        model.tf_binding_matrix_repressor.copy_(torch.tensor([[-1.0], [0.0], [0.0]]))
        model.tf_gene_topic_activator_grn.fill_(2)
        model.tf_gene_topic_repressor_grn.fill_(0.5)
    count = 0

    def permutation(n, device=None):
        nonlocal count
        count += 1
        return torch.roll(torch.arange(n, device=device), 1 + count % 2)

    monkeypatch.setattr(torch, "randperm", permutation)
    act = compute_atac_grn_activator_with_significance(model, "cpu", 0.05, tmp_path, 2)
    rep = compute_atac_grn_repressor_with_significance(model, "cpu", 0.05, tmp_path, 2)
    np.testing.assert_allclose(act[:, :, 0], 1)
    np.testing.assert_allclose(rep[:, :, 0], 1)
    np.testing.assert_array_equal(act[:, :, 1:], 0)
    np.testing.assert_array_equal(rep[:, :, 1:], 0)
    # Preserve historical net activator-minus-repressor semantics and accept tensors.
    a, r = compute_significant_grn(model, "cpu", 0.05, 0.05, torch.tensor([[1.0], [0.0]]), tmp_path)
    np.testing.assert_allclose(a[0, 0, 0], 1.5)
    np.testing.assert_array_equal(a[1], 0)
    np.testing.assert_array_equal(r, 0)
    save_regulons(a, ["TF"], ["G1", "G2", "G3"], 2, tmp_path)
    saved = pd.read_csv(tmp_path / "regulons_tf/activator/TF_activator.tsv", sep="\t", index_col=0)
    assert saved.index.tolist() == ["Topic_0", "Topic_1"]
    with torch.no_grad():
        model.tf_binding_matrix_activator.zero_()
        model.tf_binding_matrix_repressor.zero_()
    np.testing.assert_array_equal(compute_atac_grn_activator_with_significance(model, "cpu", 0.05, tmp_path, 2), 0)
    np.testing.assert_array_equal(compute_atac_grn_repressor_with_significance(model, "cpu", 0.05, tmp_path, 2), 0)
    with pytest.raises(ValueError, match="positive integer"):
        compute_atac_grn_activator_with_significance(model, "cpu", 0.05, tmp_path, 0)


def test_absent_motifs_never_create_repressor_binding():
    from scdori.pp.correlation import compute_in_silico_chipseq

    x = np.array([[1.0, 5.0, 2.0], [2.0, 3.0, 2.0], [3.0, 1.0, 2.0]])
    y = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    act, rep = compute_in_silico_chipseq(x, y, pd.DataFrame(np.zeros((3, 2))), percentile=90, n_bg=2)
    np.testing.assert_array_equal(act, 0)
    np.testing.assert_array_equal(rep, 0)


def test_gtf_to_valid_bed_and_user_only_gene_selection(tmp_path):
    import anndata as ad

    from scdori.pp.gene_selection import compute_hvgs_and_tfs, load_gtf
    from scdori.pp.utils import create_extended_gene_bed

    gtf = tmp_path / "annotation.gtf"
    gtf.write_text('chr1\tfixture\tgene\t11\t40\t.\t+\t.\tgene_id "id1"; gene_name "TF"; gene_type "protein_coding";\n')
    table = load_gtf(gtf)
    bed = create_extended_gene_bed(table, ["TF"], window_size=0)
    assert bed.iloc[0].tolist() == ["chr1", 10, 40, "TF", 0, "+"]
    data = ad.AnnData(np.ones((5, 2)), var=pd.DataFrame(index=["Gene", "TF"]))
    selected, genes, tfs = compute_hvgs_and_tfs(
        data, ["TF"], user_genes=["Gene"], user_tfs=["TF"], num_genes=2, num_tfs=1
    )
    assert genes == ["Gene"] and tfs == ["TF"]
    np.testing.assert_array_equal(selected.X, data.X)


def test_input_loader_checks_cell_and_binding_order(tmp_path):
    from types import SimpleNamespace

    import anndata as ad

    from scdori._core.data_io import load_scdori_inputs

    rna = ad.AnnData(
        np.ones((3, 2)),
        obs=pd.DataFrame(index=["a", "b", "c"]),
        var=pd.DataFrame({"gene_type": ["TF", "TF"]}, index=["T1", "T2"]),
    )
    atac = ad.AnnData(np.ones((3, 4)), obs=rna.obs.copy())
    rna.write_h5ad(tmp_path / "rna.h5ad")
    atac.write_h5ad(tmp_path / "atac.h5ad")
    np.save(tmp_path / "distance.npy", np.ones((2, 4)))
    np.save(tmp_path / "act.npy", np.ones((4, 2)))
    np.save(tmp_path / "rep.npy", -np.ones((4, 2)))
    config = SimpleNamespace(
        data_dir=str(tmp_path),
        output_subdir="",
        rna_metacell_file="rna.h5ad",
        atac_metacell_file="atac.h5ad",
        gene_peak_distance_file="distance.npy",
        insilico_chipseq_act_file="act.npy",
        insilico_chipseq_rep_file="rep.npy",
    )
    assert load_scdori_inputs(config)[0].shape == (3, 2)
    pd.DataFrame({"tf": ["T2", "T1"]}).to_csv(tmp_path / "tf_order.csv", index=False)
    with pytest.raises(ValueError, match="TF order"):
        load_scdori_inputs(config)
    (tmp_path / "tf_order.csv").unlink()
    atac[[2, 1, 0]].write_h5ad(tmp_path / "atac.h5ad")
    with pytest.raises(ValueError, match="same order"):
        load_scdori_inputs(config)
