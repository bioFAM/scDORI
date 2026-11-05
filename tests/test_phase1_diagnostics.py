import json
import random

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scdori._core.config import TrainConfig
from scdori._core.diagnostics import (
    effective_number_of_topics,
    knn_same_fraction,
    phase1_diagnostics,
    read_live_topics,
    topic_usage_qc,
    zscore_across_topics,
)
from scdori._core.downstream import compute_topic_gene_matrix
from scdori._core.interpretation import gene_importance_scores
from scdori._core.models import scDoRI
from scdori._core.train_grn import get_tf_expression

matplotlib.use("Agg")

DEAD = [1, 3]


@pytest.fixture
def planted(tmp_path):
    """Six topics, two of them dead by a strongly negative mu_theta bias."""
    torch.manual_seed(5)
    rng = np.random.default_rng(5)
    n = 80
    model = scDoRI(torch.device("cpu"), 12, 20, 4, 6, 2, 9, 5)
    with torch.no_grad():
        model.mu_theta.bias[DEAD] = -40.0
    rna = ad.AnnData(rng.poisson(2, (n, 12)).astype(np.float32))
    rna.var_names = [f"g{i}" for i in range(12)]
    rna.obs_names = [f"c{i}" for i in range(n)]
    rna.obs["label"] = pd.Categorical(rng.choice(["a", "b", "c"], n))
    rna.obs["batch"] = pd.Categorical(np.where(np.arange(n) % 2, "s1", "s2"))
    atac = ad.AnnData(rng.poisson(2, (n, 20)).astype(np.float32))
    atac.obs_names = rna.obs_names.copy()
    cells = np.ones((n, 1), dtype=np.float32)
    tfs = np.array([1, 3, 5, 7])
    batches = np.eye(2, dtype=np.float32)[np.arange(n) % 2]
    config = TrainConfig(cells_per_topic=10, batch_size_cell_prediction=16, umap_n_neighbors=10, batch_col="batch")
    ckpt = tmp_path / "phase1.pth"
    torch.save(model.state_dict(), ckpt)
    return model, rna, atac, cells, tfs, batches, config, ckpt


def run(planted, tmp_path, **kw):
    model, rna, atac, cells, tfs, batches, config, ckpt = planted
    return phase1_diagnostics(
        model,
        "cpu",
        rna,
        atac,
        cells,
        tfs,
        batches,
        config,
        groupby="label",
        out=tmp_path / "diag",
        checkpoint=ckpt,
        top_n_genes=5,
        top_n_tfs=3,
        **kw,
    )


def test_usage_qc_flags_and_effective_topics():
    theta = np.array([[0.85, 0.1, 0.0, 0.05], [0.75, 0.2, 0.0, 0.05], [0.35, 0.6, 0.0, 0.05]])
    qc = topic_usage_qc(theta, labels=["x", "x", "y"], dead_max_theta=0.1, weak_min_dominant_fraction=2.0)
    assert qc["dead"].tolist() == [False, False, True, False]
    assert qc["dominant_cells"].tolist() == [2, 1, 0, 0]
    # Topic 3 never reaches 0.1 but carries mass in every cell: diffuse, not dead.
    assert qc["status"].tolist() == ["live", "weak", "dead", "diffuse"]
    assert qc.loc[0, "dominant_purity"] == 1.0
    assert qc.loc[1, "enriched_groups"].startswith("y:")
    assert effective_number_of_topics(np.full((5, 4), 0.25)) == pytest.approx(4)


def test_zscore_reference_matches_existing_score_definitions():
    rng = np.random.default_rng(0)
    raw = rng.random((6, 9))
    live = [0, 2, 4, 5]
    z = zscore_across_topics(raw, live)
    expected = gene_importance_scores(pd.DataFrame(raw[live]))["across_topics_zscore"].to_numpy()
    np.testing.assert_allclose(z[live], expected, rtol=1e-12)
    assert np.isnan(z[[1, 3]]).all()
    # TF variant matches the pandas z-score in get_top_activators_per_topic.
    frame = pd.DataFrame(raw[live])
    expected = frame.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8), axis=0).to_numpy()
    np.testing.assert_allclose(zscore_across_topics(raw, live, ddof=1, eps=1e-8)[live], expected, rtol=1e-12)


def test_knn_same_fraction():
    import scipy.sparse as sp

    graph = sp.csr_matrix(np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float) * 0.3)
    assert knn_same_fraction(graph, ["a", "a", "b"]) == pytest.approx((0.5 + 1 + 0) / 3)


def test_planted_dead_topics_end_to_end(planted, tmp_path):
    diag = run(planted, tmp_path)
    np.testing.assert_array_equal(diag.live_topics, [0, 2, 4, 5])
    assert len(diag.topic_qc) == 6 and diag.topic_qc.loc["Topic_1", "status"] == "dead"
    assert diag.summary["n_dead"] == 2
    assert diag.summary["max_theta_of_dead_topics"] < 1e-10
    assert diag.summary["max_dead_theta_mass_per_cell"] < 1e-10
    # Live-reference gene view matches the downstream reference-subset calculation.
    model = planted[0]
    np.testing.assert_allclose(
        diag.gene_views["downstream_softmax_live_topics"].iloc[diag.live_topics].to_numpy(),
        compute_topic_gene_matrix(model, "cpu", selected_topics=diag.live_topics),
        rtol=1e-6,
    )
    assert diag.gene_views["zscore_live_topics"].loc["Topic_1"].isna().all()
    assert set(diag.gene_top.topic) == {f"Topic_{k}" for k in diag.live_topics}
    assert {"ARI_leiden_vs_label", "knn_label_agreement", "knn_same_batch_fraction"} <= set(diag.structure)
    assert diag.structure["expected_same_batch_fraction_if_mixed"] == pytest.approx(0.5)
    assert any(level == "warn" for level, _ in diag.guidance)  # 2 of 6 dead > 25%
    out = tmp_path / "diag"
    for name in (
        "topic_qc.csv",
        "top_genes.csv",
        "tf_gate_preview.csv",
        "diagnostics_summary.json",
        "live_topics.json",
        "gene_zscore_live_supported.csv.gz",
    ):
        assert (out / name).is_file(), name
    record = json.loads((out / "live_topics.json").read_text())
    assert record["dead_topics"] == DEAD and record["num_topics"] == 6


@pytest.mark.parametrize("policy", ["within_topic", "mean_within_across"])
def test_gate_preview_matches_phase2_gate(planted, tmp_path, policy):
    model, rna, atac, cells, tfs, batches, config, _ = planted
    config.tf_expression_scaling = policy
    diag = run(planted, tmp_path, structure=False)
    loader = DataLoader(TensorDataset(torch.arange(rna.n_obs)), batch_size=config.batch_size_cell_prediction)
    gate = get_tf_expression("True", model, "cpu", loader, rna, atac, cells, tfs, batches, config).numpy()
    np.testing.assert_array_equal(diag.tf_gate.to_numpy(), gate)
    g = diag.tf_gate_topics
    assert (g.loc[["Topic_1", "Topic_3"], "gate_cells_mean_theta"] < 1e-10).all()
    # Default policy: dead topics still receive a gate, built from cells that do not use them.
    assert diag.summary["dead_topics_with_nonzero_gate"] == 2
    counts = g[["gate_nonzero_tfs", "zero_no_rna", "zero_minmax_minimum", "zero_below_cutoff", "zero_below_detection"]].sum(1)
    assert (counts == len(tfs)).all()


def test_simple_review_keeps_all_usage_and_original_program_scores(planted, tmp_path):
    import matplotlib.pyplot as plt

    from scdori.pl import plot_phase1_overview, plot_topic_programs

    diag = run(planted, tmp_path, structure=False)
    before = diag.gene_views["zscore_live_topics"].copy()
    figs = plot_phase1_overview(diag, out=tmp_path / "overview")
    assert set(figs) == {"topic_overview", "celltype_overview"}
    bars = figs["topic_overview"].axes[0].patches
    np.testing.assert_allclose([bar.get_height() for bar in bars], sorted(diag.theta.mean(0), reverse=True), rtol=1e-6)
    # Displaying one topic must retain the scores calculated using all kept topics.
    fig = plot_topic_programs(diag, topic=2, top=3)
    expected = diag.gene_top.query("topic == 'Topic_2' and view == 'zscore_live_supported'").sort_values("rank").head(3)
    np.testing.assert_allclose([bar.get_width() for bar in fig.axes[0].patches], expected.zscore_live_supported)
    pd.testing.assert_frame_equal(before, diag.gene_views["zscore_live_topics"])
    assert (tmp_path / "overview/topic_overview.png").is_file()
    # Explicit original IDs remain valid even when the topic has no ranked entries.
    dead = plot_topic_programs(diag, topic=1)
    assert all(not ax.patches for ax in dead.axes)
    with pytest.raises(ValueError, match="original integer"):
        plot_topic_programs(diag, topic=2.5)
    with pytest.raises(ValueError, match="positive integer"):
        plot_topic_programs(diag, top=0)
    plt.close("all")


def test_live_topic_file_checks(planted, tmp_path):
    diag = run(planted, tmp_path, structure=False)
    path, ckpt = tmp_path / "diag" / "live_topics.json", planted[-1]
    np.testing.assert_array_equal(read_live_topics(path, num_topics=6, checkpoint=ckpt), diag.live_topics)
    with pytest.raises(ValueError, match="model has 7"):
        read_live_topics(path, num_topics=7)
    other = tmp_path / "other.pth"
    other.write_bytes(b"different")
    with pytest.raises(ValueError, match="different phase-1 checkpoint"):
        read_live_topics(path, checkpoint=other)


def test_all_figures_render(planted, tmp_path):
    diag = run(planted, tmp_path)
    figs = diag.plot_all(out=tmp_path / "figs", show=False)
    assert {
        "topic_usage",
        "collapse",
        "umap_structure",
        "umap_live_topics",
        "group_topic_heatmap",
        "gene_programs",
        "reference_effect",
        "tf_programs",
        "tf_gate",
    } <= set(figs)
    assert (tmp_path / "figs" / "tf_gate.png").is_file()


def test_no_live_topic_gives_error_guidance(planted, tmp_path):
    diag = run(planted, tmp_path, structure=False, dead_max_theta=1.01, dead_max_mean_fraction=10.0)
    assert diag.guidance[0][0] == "error" and diag.gene_top is None


def test_diagnostics_preserves_model_modes_weights_and_rng(planted, tmp_path):
    model = planted[0]
    model.train()
    model.encoder_rna.eval()  # preserve mixed module modes, too
    modes = [m.training for m in model.modules()]
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    diag = run(planted, tmp_path, structure=False, cell_indices=np.arange(40), gate_indices=np.arange(30))
    assert [m.training for m in model.modules()] == modes
    assert torch.equal(rng, torch.random.get_rng_state())
    np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
    assert numpy_state[2:] == np.random.get_state()[2:]
    assert python_state == random.getstate()
    for key, value in state.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    order = pd.read_csv(tmp_path / "diag/cell_order.csv")
    assert order.cell_id.tolist() == [f"c{i}" for i in range(40)]
    assert {"spearman_rna_library", "spearman_num_cells"} <= set(diag.topic_qc.columns)


def test_single_reference_topic_and_tiny_input(planted, tmp_path):
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaled = zscore_across_topics(matrix, rows=[1], ddof=1)
    np.testing.assert_array_equal(scaled[1], 0)
    assert np.isnan(scaled[0]).all()
    diag = run(planted, tmp_path, cell_indices=np.array([3, 1, 7]))
    assert "skipped" in diag.structure
    assert diag.adata is None


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [1.5], [999]])
def test_invalid_cell_selection_rejected(planted, tmp_path, indices):
    with pytest.raises(ValueError, match="cell_indices"):
        run(planted, tmp_path, structure=False, cell_indices=np.asarray(indices))


def test_overview_uses_project_palette_in_map_and_heatmap(planted, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scdori.pl import plot_phase1_overview

    diag = run(planted, tmp_path, structure=False)
    labels = planted[1].obs.copy()
    diag.adata = ad.AnnData(np.zeros((len(labels), 1)), obs=labels)
    diag.adata.obsm["X_umap"] = np.random.default_rng(7).normal(size=(len(labels), 2))
    palette = {"a": "#532c8a", "b": "#ef4e22", "c": "#1a1a1a"}
    path = tmp_path / "palette.json"
    path.write_text(json.dumps({"label_colors": palette}))
    recorded = []
    original = sns.clustermap

    def capture(*args, **kwargs):
        recorded.append(kwargs.get("row_colors"))
        return original(*args, **kwargs)

    monkeypatch.setattr(sns, "clustermap", capture)
    qc = diag.topic_qc.copy(deep=True)
    figures = plot_phase1_overview(diag, palette=path)
    assert dict(zip(diag.adata.obs.label.cat.categories, diag.adata.uns["label_colors"], strict=True)) == palette
    assert recorded[0].to_dict() == {label: palette[label] for label in diag.group_topic_mean.index}
    pd.testing.assert_frame_equal(diag.topic_qc, qc)
    for figure in figures.values():
        plt.close(figure)
