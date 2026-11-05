"""Saved-fit loading, non-mutating TF previews, and shared heatmap order."""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scdori import load_downstream_model, preview_phase2_tfs, scDoRI
from scdori.configuration import load_notebook_config, save_notebook_config, training_input_order
from scdori._core.train_grn import get_tf_expression


@pytest.fixture
def project(tmp_path):
    folder = tmp_path / "data/generated"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(3)
    obs = pd.DataFrame({"sample": ["B", "A"] * 6, "celltype": ["type1", "type2"] * 6},
                       index=[f"c{i}" for i in range(12)])
    rna = ad.AnnData(rng.poisson(3, (12, 4)).astype(np.float32), obs=obs.copy(),
        var=pd.DataFrame({"gene_type": ["TF", "gene", "TF", "gene"]}, index=["TFa", "g1", "TFb", "g2"]))
    atac = ad.AnnData(rng.poisson(3, (12, 5)).astype(np.float32), obs=obs.copy(),
                     var=pd.DataFrame(index=[f"p{i}" for i in range(5)]))
    atac.uns["scdori_atac_count_mode"] = "insertions"
    rna.write_h5ad(folder / "rna_processed.h5ad")
    atac.write_h5ad(folder / "atac_processed.h5ad")
    for name, shape in [("gene_peak_distance_exp", (4, 5)), ("insilico_chipseq_act", (5, 2)), ("insilico_chipseq_rep", (5, 2))]:
        np.save(folder / f"{name}.npy", rng.uniform(size=shape).astype(np.float32))
    path = tmp_path / "scdori_config.json"
    path.write_text(json.dumps({"training": {"data_dir": "data", "output_subdir": "generated", "batch_col": "sample",
        "weights_folder_grn": "fit", "num_topics": 3, "dim_encoder1": 7, "dim_encoder2": 5,
        "grn_topic_selection": "all", "batch_size_cell_prediction": 4, "cells_per_topic": 3},
        "downstream": {"grn_output": "downstream"}}))
    settings = load_notebook_config(path)
    fit = Path(settings.training.weights_folder_grn)
    fit.mkdir()
    model = scDoRI(torch.device("cpu"), 4, 5, 2, 3, 2, 7, 5)
    torch.save(model.state_dict(), fit / "best_scdori_best_eval.pth")
    save_notebook_config(settings, "training", fit)
    (fit / "training_input_order.json").write_text(json.dumps(training_input_order(rna, atac, [0, 2], ["A", "B"])))
    expression = np.array([[.25, 0], [0, .8], [.5, .3]], dtype=np.float32)
    np.save(fit / "topic_tf_expression.npy", expression)
    np.savez(fit / "training_split.npz", train=np.arange(8), validation=np.arange(8, 12))
    return settings, expression


def test_loader_uses_saved_settings_and_expression_without_refitting(project, monkeypatch):
    settings, expression = project
    settings.training.num_topics = 99
    settings.training.batch_col = "incorrect_current_setting"
    settings.training.tf_expression_clamp = .99
    monkeypatch.setattr("scdori._core.train_grn.get_tf_expression", lambda *a, **k: pytest.fail("Saved values must be reused"))
    result = load_downstream_model(settings, device="cpu")
    np.testing.assert_array_equal(result.tf_expression, expression)
    np.testing.assert_array_equal(result.onehot_batch[:2], [[0, 1], [1, 0]])
    assert result.config.num_topics == 3 and result.config.tf_expression_clamp == .2
    assert settings.training.num_topics == 99  # the caller's object is not silently modified
    assert result.config.data_dir == settings.training.data_dir
    assert result.tf_expression_record["recomputed"] is False
    assert result.rna.layers["counts"].shape == (12, 4)


def test_loader_rejects_reordered_cells_and_does_not_create_results(project):
    settings, _ = project
    folder = settings.training.data_dir / settings.training.output_subdir
    for filename in ["rna_processed.h5ad", "atac_processed.h5ad"]:
        a = ad.read_h5ad(folder / filename)
        a[::-1].copy().write_h5ad(folder / filename)
    with pytest.raises(ValueError, match="input order"):
        load_downstream_model(settings, device="cpu")
    assert not Path(settings.downstream.grn_output).exists()


def test_missing_expression_requires_explicit_original_split_recomputation(project):
    settings, _ = project
    folder = Path(settings.training.weights_folder_grn)
    (folder / "topic_tf_expression.npy").unlink()
    with pytest.raises(FileNotFoundError, match="saved phase-2"):
        load_downstream_model(settings, device="cpu")
    settings.downstream.allow_tf_expression_recompute = True
    result = load_downstream_model(settings, device="cpu")
    loader = DataLoader(TensorDataset(torch.arange(8)), batch_size=4, shuffle=False)
    expected = get_tf_expression(result.config.tf_expression_mode, result.model, "cpu", loader,
        result.rna, result.atac, result.num_cells, result.tf_indices, result.onehot_batch, result.config)
    torch.testing.assert_close(result.tf_expression, expected, rtol=0, atol=0)
    assert result.tf_expression_record["recomputed"]
    np.savez(folder / "training_split.npz", train=np.array([0, 0]))
    with pytest.raises(ValueError, match="training split"):
        load_downstream_model(settings, device="cpu")


@pytest.mark.parametrize("mode", ["actual", "latent"])
def test_preview_matches_expression_rule_and_preserves_rng_model_and_topic_selection(project, mode):
    settings, _ = project
    fit = load_downstream_model(settings, device="cpu")
    c, model = fit.config, fit.model
    c.tf_expression_mode = mode
    c.grn_topic_selection = "explicit"
    c.grn_topics = [0, 2]
    loader = DataLoader(TensorDataset(torch.arange(8)), batch_size=4, shuffle=True)
    model.train()
    model.encoder_rna.eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    modes = [m.training for m in model.modules()]
    random = torch.get_rng_state().clone()
    preview = preview_phase2_tfs(model, "cpu", loader, fit.rna, fit.atac,
                                 fit.num_cells, fit.tf_indices, fit.onehot_batch, c)
    assert modes == [m.training for m in model.modules()]
    assert getattr(model, "grn_topics", None) is None
    torch.testing.assert_close(torch.get_rng_state(), random, rtol=0, atol=0)
    for key, value in state.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    expected = get_tf_expression(mode, model, "cpu", DataLoader(loader.dataset, batch_size=4),
                                 fit.rna, fit.atac, fit.num_cells, fit.tf_indices, fit.onehot_batch, c)
    expected[1] = 0
    np.testing.assert_array_equal(preview.weights.to_numpy(), expected.numpy())
    assert preview.summary.loc["Topic_1", "Phase 2"] == "Skipped topic"
    assert preview.summary["TFs passing the rule"].tolist() == (preview.weights > 0).sum(axis=1).tolist()
    c.tf_expression_clamp = 0
    relaxed = preview_phase2_tfs(model, "cpu", loader, fit.rna, fit.atac, fit.num_cells, fit.tf_indices, fit.onehot_batch, c)
    assert ((preview.weights > 0) <= (relaxed.weights > 0)).to_numpy().all()


def test_three_heatmaps_share_hierarchical_order_without_rescaling():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scdori.pl import plot_tf_score_views
    frame = pd.DataFrame([[0, .9, .8], [.8, 0, .1], [.1, .7, 1]],
                         index=["Topic_7", "Topic_2", "Topic_20"], columns=["C", "A", "B"])
    scores = {"minmax_consensus": frame, "within_group_minmax": frame * .5, "across_groups_minmax": frame * .8}
    before = {k: v.copy() for k, v in scores.items()}
    fig, order = plot_tf_score_views(scores, list(frame.index), list(frame.columns))
    assert order["topics"] == list(frame.index[leaves_list(linkage(frame, method="average"))])
    assert order["tfs"] == list(frame.columns[leaves_list(linkage(frame.T, method="average"))])
    axes = [a for a in fig.axes if a.get_ylabel() == "Topic"]
    assert len(axes) == 3
    for ax, view in zip(axes, order["views"]):
        assert [t.get_text() for t in ax.get_xticklabels()] == order["tfs"]
        assert [t.get_text() for t in ax.get_yticklabels()] == [t.replace("_", " ") for t in order["topics"]]
        np.testing.assert_array_equal(np.asarray(ax.collections[0].get_array()).reshape(3, 3),
                                      scores[view].loc[order["topics"], order["tfs"]])
    for k, v in scores.items(): pd.testing.assert_frame_equal(v, before[k])
    plt.close(fig)
