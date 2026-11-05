"""Compare TF-score views using one hierarchical order per network sign."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage


def plot_tf_score_views(scores, topics, tfs, views=None, kind="activator", eligible=None):
    """Cluster the mean scores once; show all requested views in that same order.

    Average linkage and Euclidean distance are applied to the displayed mean-score
    matrix, without further scaling. Activator/repressor inputs are passed separately.
    Return the figure and its labelled row/column order. Score tables are untouched.
    Optional boolean ``eligible`` masks excluded topic/TF entries in grey in all
    views; clustering still uses the original mean scores, without rescaling.
    """
    views = list(views or ["minmax_consensus", "within_group_minmax", "across_groups_minmax"])
    reference = scores["minmax_consensus"].loc[topics, tfs]
    mask = None
    if eligible is not None:
        selected = eligible.reindex(index=reference.index, columns=reference.columns)
        if selected.isna().any().any() or not all(dtype == bool for dtype in selected.dtypes):
            raise ValueError("Eligibility must provide a boolean for every displayed topic and TF")
        mask = ~selected
    if reference.empty or not np.isfinite(reference.to_numpy()).all():
        raise ValueError("Clustering needs a nonempty finite score matrix")
    if not reference.index.is_unique or not reference.columns.is_unique:
        raise ValueError("Topic and TF labels must be unique")
    row_tree = linkage(reference.to_numpy(), method="average", metric="euclidean") if len(reference) > 1 else None
    col_tree = linkage(reference.to_numpy().T, method="average", metric="euclidean") if reference.shape[1] > 1 else None
    rows = reference.index[leaves_list(row_tree)] if row_tree is not None else reference.index
    cols = reference.columns[leaves_list(col_tree)] if col_tree is not None else reference.columns
    titles = {"minmax_consensus": "Mean of both", "within_group_minmax": "Within topic", "across_groups_minmax": "Across topics"}
    with sns.plotting_context("notebook", font_scale=1):
        fig = plt.figure(figsize=(7 * len(views) + 1, max(6, .46 * len(rows) + 1)))
        grid = fig.add_gridspec(2, len(views) + 1, width_ratios=[.4] + [4] * len(views),
                               height_ratios=[.65, 5], wspace=.55, hspace=.05)
        row_ax = col_ax = first_ax = None
        if row_tree is not None:
            ax = fig.add_subplot(grid[1, 0])
            row_ax = ax
            dendrogram(row_tree, orientation="left", no_labels=True, color_threshold=0,
                       above_threshold_color="#666666", ax=ax)
            ax.invert_yaxis()
            ax.axis("off")
        if col_tree is not None:
            ax = fig.add_subplot(grid[0, 1])
            col_ax = ax
            dendrogram(col_tree, no_labels=True, color_threshold=0, above_threshold_color="#666666", ax=ax)
            ax.axis("off")
        for j, view in enumerate(views):
            values = scores[view].loc[rows, cols]
            ax = fig.add_subplot(grid[1, j + 1])
            if j == 0:
                first_ax = ax
            if view == "share_across_groups_zscore":
                limit = max(float(np.abs(values.to_numpy()).max()), 1e-8)
                colors = dict(cmap="RdBu_r", center=0, vmin=-limit, vmax=limit)
            else:
                colors = dict(cmap="viridis", vmin=0)
                if view != "raw":
                    colors["vmax"] = 1
            if mask is not None:
                ax.set_facecolor("#dddddd")
            sns.heatmap(values, ax=ax, xticklabels=True, yticklabels=True,
                        mask=None if mask is None else mask.loc[rows, cols], **colors)
            title_ax = col_ax if j == 0 and col_ax is not None else ax
            title_ax.set_title(f"{kind.capitalize()}: {titles.get(view, view)}", fontsize=12, pad=12)
            ax.set(xlabel="TF", ylabel="Topic")
            ax.set_xticklabels(cols, rotation=90, fontsize=8)
            ax.set_yticklabels([str(t).replace("_", " ") for t in rows], rotation=0, fontsize=10)
        fig.subplots_adjust(bottom=.22, top=.94)
        # Heatmap colour bars shrink their axes. Match the dendrograms to the
        # remaining heatmap area so each leaf aligns with its row or column.
        box = first_ax.get_position()
        if col_ax is not None:
            tree_box = col_ax.get_position()
            col_ax.set_position([box.x0, tree_box.y0, box.width, tree_box.height])
        if row_ax is not None:
            tree_box = row_ax.get_position()
            row_ax.set_position([tree_box.x0, box.y0, tree_box.width, box.height])
    return fig, {"topics": list(rows), "tfs": list(cols), "reference": "minmax_consensus",
                 "method": "average", "metric": "euclidean", "views": views}
