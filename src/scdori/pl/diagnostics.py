"""Figures for :func:`scdori.phase1_diagnostics` (matplotlib/seaborn/scanpy)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scdori._core.diagnostics import preserve_random_state

# Status colours: live and weak are categorical slots, dead is a neutral grey.
STATUS_COLORS = {"live": "#2a78d6", "weak": "#eda100", "diffuse": "#1baf7a", "dead": "#9a9a95"}
_TEXT_MUTED = "#52514e"


def _status_legend(ax, statuses):
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATUS_COLORS[s]) for s in STATUS_COLORS if s in set(statuses)]
    ax.legend(handles, [s for s in STATUS_COLORS if s in set(statuses)], frameon=False, loc="upper right")


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=_TEXT_MUTED, labelsize=8)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_topic_usage(diag):
    """Mean theta per topic (sorted) and max theta per topic on a log scale with the dead threshold."""
    qc = diag.topic_qc.sort_values("mean_theta", ascending=False)
    colors = [STATUS_COLORS[s] for s in qc["status"]]
    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(qc) * 0.28), 6), sharex=True, constrained_layout=True)
    x = np.arange(len(qc))
    axes[0].bar(x, qc["mean_theta"], color=colors, width=0.8)
    axes[0].set_ylabel("mean theta")
    axes[0].set_title(
        f"Topic usage: {diag.summary['n_dead']} dead, {diag.summary['n_diffuse']} diffuse, "
        f"{diag.summary['n_weak']} weak of {diag.summary['n_topics']}; "
        f"effective topics {diag.summary['effective_topics']:.1f}",
        loc="left",
        fontsize=10,
    )
    _status_legend(axes[0], qc["status"])
    axes[1].scatter(x, np.clip(qc["max_theta"], 1e-12, None), c=colors, s=28, zorder=3)
    axes[1].axhline(diag.settings["dead_max_theta"], color=_TEXT_MUTED, linestyle="--", linewidth=1)
    axes[1].text(
        len(qc) - 0.5,
        diag.settings["dead_max_theta"] * 1.3,
        "max-theta threshold (also check mean usage)",
        ha="right",
        fontsize=8,
        color=_TEXT_MUTED,
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("max theta (log)")
    axes[1].set_xticks(x, qc.index, rotation=90)
    for ax in axes:
        _style(ax)
    return fig


@preserve_random_state
def plot_phase1_overview(diag, out=None, palette=None):
    """A short notebook review: topic use, cell relationships and annotated groups.

    All topics are included in the usage chart. This changes presentation only;
    it does not select topics for training or modify diagnostic score tables.
    """
    from matplotlib.ticker import PercentFormatter

    qc = diag.topic_qc.sort_values("mean_theta", ascending=False)
    fig, ax = plt.subplots(figsize=(max(8, len(qc) * 0.25), 3.5), constrained_layout=True)
    dead = qc["status"].eq("dead")
    ax.bar(np.arange(len(qc)), qc.mean_theta, color=np.where(dead, STATUS_COLORS["dead"], STATUS_COLORS["live"]))
    ax.set_xticks(np.arange(len(qc)), qc.index, rotation=90)
    ax.set(title="How much is each topic used?", ylabel="Average share of each cell")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(
        [plt.Rectangle((0, 0), 1, 1, color=STATUS_COLORS[k]) for k in ("live", "dead")],
        ["Retained for review", "Almost unused"],
        frameon=False,
    )
    _style(ax)
    figs = {"topic_overview": fig}
    if diag.adata is not None:
        keys = list(
            dict.fromkeys(
                k for k in (diag.settings.get("groupby"), diag.settings.get("batch_key")) if k and k in diag.adata.obs
            )
        )
        figs["cells_overview"] = plot_structure(diag, color=keys or ["dominant_topic"], palette=palette)
    if diag.group_topic_mean is not None and len(diag.live_topics):
        figs["celltype_overview"] = plot_group_topic_heatmap(diag, palette=palette)
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        for name, figure in figs.items():
            figure.savefig(out / f"{name}.png", dpi=130, bbox_inches="tight")
    return figs


@preserve_random_state
def plot_topic_programs(diag, topic=None, top=10):
    """Readable gene/TF lists for one topic, keeping the full kept-topic reference.

    Selecting a display topic never recalculates scores against just that topic.
    With ``topic=None``, show the kept topic with greatest mean usage.
    """
    if not isinstance(top, (int, np.integer)) or isinstance(top, bool) or top < 1:
        raise ValueError("top must be a positive integer")
    if topic is None:
        if not len(diag.live_topics):
            raise ValueError("No retained topics to display; inspect topic usage first")
        topic = max(diag.live_topics, key=lambda k: diag.topic_qc.loc[diag.topic_ids[k], "mean_theta"])
    if not isinstance(topic, (int, np.integer)) or isinstance(topic, bool) or not 0 <= topic < len(diag.topic_ids):
        raise ValueError("topic must be an original integer topic ID within range")
    name = diag.topic_ids[topic]
    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.5, top * 0.27 + 1.4)), constrained_layout=True)
    fig.suptitle(f"Genes and TFs associated with {name.replace('_', ' ')}")
    for ax, table, view, feature, title in (
        (axes[0], diag.gene_top, "zscore_live_supported", "gene", "Genes"),
        (axes[1], diag.tf_top, "share_zscore_live_topics", "tf", "TF expression programs"),
    ):
        selected = table[(table.topic == name) & (table.view == view)].sort_values("rank").head(top)
        if selected.empty:
            ax.text(0.5, 0.5, "No ranked entries for this topic", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        ax.barh(np.arange(len(selected)), selected[view], color=STATUS_COLORS["live"])
        ax.set_yticks(np.arange(len(selected)), selected[feature])
        ax.invert_yaxis()
        ax.set(title=title, xlabel="Relative score across retained topics")
        ax.axvline(0, color=_TEXT_MUTED, linewidth=0.7)
        _style(ax)
    return fig


def plot_collapse(diag):
    """Decoder row spread against max theta, and the mu_theta bias per topic."""
    qc = diag.topic_qc
    colors = [STATUS_COLORS[s] for s in qc["status"]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, col, title in (
        (axes[0], "peak_decoder_row_std", "topic-peak decoder"),
        (axes[1], "tf_decoder_row_std", "topic-TF decoder"),
    ):
        ax.scatter(np.clip(qc["max_theta"], 1e-12, None), qc[col], c=colors, s=28)
        ax.set_xscale("log")
        ax.set_xlabel("max theta (log)")
        ax.set_ylabel("row std")
        ax.set_title(title, loc="left", fontsize=10)
        _style(ax)
    order = qc["mu_theta_bias"].sort_values().index
    axes[2].bar(
        np.arange(len(order)), qc.loc[order, "mu_theta_bias"], color=[STATUS_COLORS[s] for s in qc.loc[order, "status"]]
    )
    axes[2].set_xticks(np.arange(len(order)), order, rotation=90, fontsize=6)
    axes[2].set_ylabel("mu_theta bias")
    axes[2].set_title("topic logit bias", loc="left", fontsize=10)
    _style(axes[2])
    _status_legend(axes[0], qc["status"])
    axes[0].get_legend().set_loc("center")
    return fig


def plot_structure(diag, color=None, size=None, palette=None):
    """UMAP on theta coloured by the label, batch and Leiden clusters."""
    import scanpy as sc

    if diag.adata is None:
        raise ValueError("Structure was not computed (structure=False)")
    groupby = diag.settings.get("groupby")
    if palette is not None and groupby in diag.adata.obs:
        from scdori._core.exploration import apply_label_colors

        apply_label_colors(diag.adata, groupby, palette)
    keys = color or [
        k
        for k in (diag.settings.get("groupby"), diag.settings.get("batch_key"), "leiden", "dominant_topic")
        if k and k in diag.adata.obs
    ]
    size = size or max(1.0, 120000 / diag.adata.n_obs)
    # Stack panels so long cell-type/topic legends cannot overlap a neighbouring
    # plot. This also stays readable at notebook width on smaller screens.
    return sc.pl.umap(diag.adata, color=keys, return_fig=True, show=False, size=size, ncols=1, legend_fontsize=8)


def plot_live_topic_umaps(diag, ncols=6, size=None):
    """Small multiples of theta for every live topic on the UMAP."""
    import scanpy as sc

    if diag.adata is None:
        raise ValueError("Structure was not computed (structure=False)")
    names = [diag.topic_ids[k] for k in diag.live_topics]
    size = size or max(1.0, 60000 / diag.adata.n_obs)
    return sc.pl.umap(
        diag.adata,
        color=names,
        return_fig=True,
        show=False,
        ncols=ncols,
        size=size,
        cmap="viridis",
        vmax="p99",
        frameon=False,
    )


def plot_group_topic_heatmap(diag, palette=None):
    """Mean theta per user group and live topic (clustered)."""
    import seaborn as sns

    if diag.group_topic_mean is None:
        raise ValueError("No groupby label was given")
    data = diag.group_topic_mean[[diag.topic_ids[k] for k in diag.live_topics]]
    row_colors = None
    if palette is not None:
        from scdori._core.exploration import load_palette

        colors = load_palette(palette, key=diag.settings.get("groupby") or "celltype")
        row_colors = pd.Series(data.index.astype(str), index=data.index).map(colors).fillna("#bdbdbd")
        row_colors.name = "Cell type"
    g = sns.clustermap(
        data,
        cmap="Blues",
        figsize=(max(8, data.shape[1] / 2.2), max(6, data.shape[0] / 3)),
        xticklabels=True,
        yticklabels=True,
        row_colors=row_colors,
        row_cluster=len(data) > 1,
        col_cluster=data.shape[1] > 1,
        cbar_kws={"label": "Average topic proportion"},
    )
    g.ax_heatmap.tick_params(labelsize=7)
    return g.figure


def _program_heatmap(z, title, label):
    import seaborn as sns

    fig, ax = plt.subplots(
        figsize=(max(8, z.shape[1] * 0.16 + 2), max(3.5, z.shape[0] * 0.28 + 1.2)), constrained_layout=True
    )
    if not z.size:
        ax.text(0.5, 0.5, "No supported features for this view", ha="center", va="center")
        ax.set_axis_off()
        return fig
    sns.heatmap(
        z,
        cmap="RdBu_r",
        center=0,
        vmin=-4,
        vmax=4,
        xticklabels=True,
        yticklabels=True,
        ax=ax,
        cbar_kws={"label": label, "shrink": 0.6},
    )
    ax.tick_params(labelsize=6)
    ax.set_title(title, loc="left", fontsize=10)
    return fig


def plot_gene_programs(diag, top=5):
    """Top supported genes of each kept topic (live-topic reference), and the reference-effect panels."""
    live_names = [diag.topic_ids[k] for k in diag.live_topics]
    top_tab = diag.gene_top[(diag.gene_top.view == "zscore_live_supported") & (diag.gene_top["rank"] <= top)]
    genes = list(dict.fromkeys(top_tab.gene))
    z = diag.gene_views["zscore_live_topics"].loc[live_names, genes]
    fig = _program_heatmap(z, f"Gene programs: top {top} supported genes per kept topic", "z-score across kept topics")
    return fig, plot_reference_effect(diag)


def plot_reference_effect(diag):
    """Per kept topic: how the scaling reference changes the top lists, and raw support/detection of top genes."""
    m = diag.gene_view_metrics
    tf = diag.topic_qc.loc[m.index, "tf_top_jaccard_all_vs_live"]
    fig, axes = plt.subplots(1, 3, figsize=(max(12, len(m) * 0.75), 3.8), constrained_layout=True)
    x = np.arange(len(m))
    axes[0].bar(x - 0.2, m["jaccard_all_vs_live"], 0.4, color="#2a78d6", label="top genes")
    axes[0].bar(x + 0.2, tf, 0.4, color="#eb6834", label="top TFs (decoder share)")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Jaccard, all vs kept-topic reference")
    axes[0].set_title("Does the scaling reference change the top lists?", loc="left", fontsize=10)
    views = [
        ("zscore_all_topics", "all-topic z", "#9a9a95"),
        ("zscore_live_topics", "kept-topic z", "#2a78d6"),
        ("zscore_live_supported", "kept + raw support", "#1baf7a"),
    ]
    for i, (view, label, color) in enumerate(views[:2]):
        axes[1].bar(x + (i - 0.5) * 0.4, m[f"support_fraction_{view}"], 0.4, color=color, label=label)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("share of top genes with raw support")
    axes[1].set_title("Are top genes backed by raw signal?", loc="left", fontsize=10)
    for i, (view, label, color) in enumerate(views):
        axes[2].bar(x + (i - 1) * 0.28, m[f"detection_ratio_{view}"], 0.28, color=color, label=label)
    axes[2].axhline(1, color=_TEXT_MUTED, linewidth=0.8, linestyle="--")
    axes[2].set_ylabel("detection in dominant cells / all cells")
    axes[2].set_title("Are top genes detected where the topic is active?", loc="left", fontsize=10)
    for ax in axes:
        ax.set_xticks(x, m.index, rotation=90, fontsize=7)
        ax.legend(frameon=False, fontsize=7, loc="lower left")
        _style(ax)
    return fig


def plot_tf_programs(diag, top=5):
    """TF expression programs (topic-TF decoder share z-scored across kept topics). Not GRN edges."""
    live_names = [diag.topic_ids[k] for k in diag.live_topics]
    sel = diag.tf_top[(diag.tf_top.view == "share_zscore_live_topics") & (diag.tf_top["rank"] <= top)]
    tfs = list(dict.fromkeys(sel.tf))
    z = diag.tf_views["share_zscore_live_topics"].loc[live_names, tfs]
    return _program_heatmap(
        z, f"TF expression programs: top {top} TFs per kept topic (not GRN edges)", "z-score across kept topics"
    )


def plot_tf_gate(diag):
    """Phase-2 TF-gate preview: support of the gate cells and why gate entries are zero."""
    g = diag.tf_gate_topics
    has_cells = "gate_cells_mean_theta" in g
    fig, axes = plt.subplots(
        1, 2 if has_cells else 1, figsize=(14 if has_cells else 8, 4), squeeze=False, constrained_layout=True
    )
    x = np.arange(len(g))
    if has_cells:
        ax = axes[0, 0]
        ax.bar(x, g["gate_cells_mean_theta"], color=[STATUS_COLORS[s] for s in g["status"]])
        ax.set_ylabel("mean topic proportion in selected cells")
        ax.set_title(
            f"Cells used for TF expression (top {diag.settings['cells_per_topic']} by topic proportion)",
            loc="left",
            fontsize=10,
        )
        ax.set_xticks(x, g.index, rotation=90, fontsize=6)
        _status_legend(ax, g["status"])
        ax.get_legend().set_loc("upper left")
        _style(ax)
    ax = axes[0, -1]
    bottom = np.zeros(len(g))
    for col, label, color in (
        ("gate_nonzero_tfs", "non-zero", "#2a78d6"),
        ("zero_below_cutoff", "below cutoff", "#eda100"),
        ("zero_below_detection", "too rarely detected", "#8e6bbf"),
        ("zero_minmax_minimum", "zero after scaling", "#e87ba4"),
        ("zero_no_rna", "no RNA", "#9a9a95"),
    ):
        ax.bar(x, g[col], bottom=bottom, color=color, label=label, edgecolor="white", linewidth=0.5)
        bottom += g[col].to_numpy()
    ax.set_ylabel("TFs")
    ax.set_title("TFs retained or excluded by the expression rule", loc="left", fontsize=10)
    ax.set_xticks(x, g.index, rotation=90, fontsize=6)
    ax.legend(frameon=False, fontsize=8, ncols=5, loc="upper left", bbox_to_anchor=(0, -0.25))
    _style(ax)
    return fig


def plot_phase1_diagnostics(diag, out=None, show=True):
    """Draw all figures; return ``{name: figure}`` and optionally save them as PNG in ``out``."""
    figs = {"topic_usage": plot_topic_usage(diag), "collapse": plot_collapse(diag)}
    if diag.adata is not None:
        figs["umap_structure"] = plot_structure(diag)
        if len(diag.live_topics):
            figs["umap_live_topics"] = plot_live_topic_umaps(diag)
    if diag.group_topic_mean is not None and len(diag.live_topics):
        figs["group_topic_heatmap"] = plot_group_topic_heatmap(diag)
    if diag.gene_top is not None:
        figs["gene_programs"], figs["reference_effect"] = plot_gene_programs(diag)
        figs["tf_programs"] = plot_tf_programs(diag)
        figs["tf_gate"] = plot_tf_gate(diag)
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(out / f"{name}.png", dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close("all")
    return figs


def topic_status_table(diag):
    """Compact per-topic table for notebooks: status, usage, dominant groups and gate support."""
    cols = [
        c
        for c in (
            "status",
            "mean_theta",
            "max_theta",
            "dominant_cells",
            "dominant_purity",
            "enriched_groups",
            "mu_theta_bias",
            "gate_cells_mean_theta",
            "gate_nonzero_tfs",
        )
        if c in diag.topic_qc
    ]
    return pd.DataFrame(diag.topic_qc[cols])
