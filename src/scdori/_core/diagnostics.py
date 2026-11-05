"""Phase-1 diagnostics: topic usage, collapse indicators, structure and programs.

Run these after phase 1 and before phase 2, to decide whether to retrain phase 1
(for example with fewer topics or another seed) and which topics to carry into
phase 2. How many topics a dataset supports is data-dependent; unused topics
typically collapse early in phase 1 through the softmax/encoder dynamics.

All thresholds here are documented heuristics for review, not statistical tests.
Topic scores are phase-1 *expression/accessibility programs*, not GRN edges.
"""

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .evaluation import get_latent_topics
from .train_grn import scale_topic_tf_expression, topic_tf_detection_filter, topic_tf_expression_means

logger = logging.getLogger(__name__)

LIVE_TOPICS_FILE = "live_topics.json"
LIVE_TOPICS_FORMAT = 1


def preserve_random_state(function):
    """Keep optional inspection from changing subsequent optimization randomness."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        py_state, np_state = random.getstate(), np.random.get_state()
        devices = []
        modes = {}
        if args and hasattr(args[0], "parameters"):
            modes = {module: module.training for module in args[0].modules()}
            device = next(args[0].parameters()).device
            if device.type == "cuda":
                devices = [device.index]
        try:
            with torch.random.fork_rng(devices=devices):
                return function(*args, **kwargs)
        finally:
            for module, training in modes.items():
                module.training = training
            random.setstate(py_state)
            np.random.set_state(np_state)

    return wrapped


@dataclass
class Phase1Diagnostics:
    """Results of :func:`phase1_diagnostics`; tables keep every topic, dead ones included."""

    topic_ids: list
    theta: np.ndarray
    topic_qc: pd.DataFrame
    live_topics: np.ndarray
    summary: dict
    guidance: list
    group_topic_mean: pd.DataFrame | None = None
    structure: dict = field(default_factory=dict)
    gene_views: dict = field(default_factory=dict)
    gene_top: pd.DataFrame | None = None
    gene_view_metrics: pd.DataFrame | None = None
    tf_views: dict = field(default_factory=dict)
    tf_top: pd.DataFrame | None = None
    tf_gate: pd.DataFrame | None = None
    tf_gate_topics: pd.DataFrame | None = None
    adata: object = None
    settings: dict = field(default_factory=dict)
    cell_ids: np.ndarray | None = None
    cell_names: list | None = None

    def print_guidance(self):
        """Print the summary and heuristic guidance as plain text."""
        s = self.summary
        print(
            f"{s['n_topics']} topics: {s['n_dead']} dead (collapsed), {s['n_diffuse']} diffuse, {s['n_weak']} weak; "
            f"effective number of topics {s['effective_topics']:.1f}"
        )
        for level, message in self.guidance:
            print(f"[{level.upper()}] {message}")

    @preserve_random_state
    def plot_all(self, out=None, show=True):
        """Draw every diagnostic figure; optionally save PNGs to ``out``."""
        from scdori.pl.diagnostics import plot_phase1_diagnostics

        return plot_phase1_diagnostics(self, out=out, show=show)

    def live_topic_record(self, checkpoint=None):
        """JSON-serializable record of the live-topic selection (original topic IDs)."""
        record = {
            "format": LIVE_TOPICS_FORMAT,
            "num_topics": len(self.topic_ids),
            "live_topics": [int(t) for t in self.live_topics],
            "dead_topics": [int(t) for t in np.setdiff1d(np.arange(len(self.topic_ids)), self.live_topics)],
            "rule": "dead = max theta < dead_max_theta in all cells and mean theta < dead_max_mean_fraction / K",
            "n_cells": self.settings.get("n_cells"),
            "dead_max_theta": self.settings.get("dead_max_theta"),
            "dead_max_mean_fraction": self.settings.get("dead_max_mean_fraction"),
            "max_dead_theta_mass_per_cell": self.summary.get("max_dead_theta_mass_per_cell"),
        }
        if checkpoint is not None:
            record["phase1_checkpoint"] = str(checkpoint)
            record["phase1_checkpoint_sha256"] = _sha256(checkpoint)
        return record

    def save(self, out, checkpoint=None):
        """Write tables, ``diagnostics_summary.json`` and ``live_topics.json`` to ``out``."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        self.topic_qc.to_csv(out / "topic_qc.csv", index_label="topic")
        np.save(out / "cell_topics.npy", self.theta.astype(np.float32))
        pd.DataFrame({"cell_index": self.cell_ids, "cell_id": self.cell_names}).to_csv(
            out / "cell_order.csv", index=False
        )
        tables = {
            "group_topic_mean.csv": self.group_topic_mean,
            "top_genes.csv": self.gene_top,
            "gene_view_metrics.csv": self.gene_view_metrics,
            "top_tfs_expression_program.csv": self.tf_top,
            "tf_gate_preview.csv": self.tf_gate,
            "tf_gate_topics.csv": self.tf_gate_topics,
        }
        for name, table in tables.items():
            if table is not None:
                table.to_csv(out / name, index=name not in {"top_genes.csv", "top_tfs_expression_program.csv"})
        for prefix, views in (("gene", self.gene_views), ("tf", self.tf_views)):
            for name, table in views.items():
                table.to_csv(out / f"{prefix}_{name}.csv.gz")
        summary = {
            "summary": self.summary,
            "structure": self.structure,
            "guidance": self.guidance,
            "settings": self.settings,
        }
        (out / "diagnostics_summary.json").write_text(json.dumps(summary, indent=1, default=_json_default))
        if len(self.live_topics):
            write_live_topics(out / LIVE_TOPICS_FILE, self.live_topic_record(checkpoint))
        else:
            (out / LIVE_TOPICS_FILE).unlink(missing_ok=True)
            logger.warning("No live topics; live_topics.json not written")
        return out


@preserve_random_state
def phase1_diagnostics(
    model,
    device,
    rna_anndata,
    atac_anndata,
    num_cells,
    tf_indices,
    encoding_batch_onehot,
    config,
    *,
    cell_indices=None,
    gate_indices=None,
    groupby=None,
    batch_key=None,
    dead_max_theta=0.1,
    dead_max_mean_fraction=0.1,
    weak_min_dominant_fraction=0.1,
    support_quantile=0.9,
    top_n_genes=50,
    top_n_tfs=20,
    structure=True,
    leiden_resolution=1.0,
    max_detection_cells=5000,
    out=None,
    checkpoint=None,
    random_state=0,
):
    """Diagnose a trained phase-1 model before phase 2.

    Parameters
    ----------
    model, device, rna_anndata, atac_anndata, num_cells, tf_indices, encoding_batch_onehot
        As for :func:`scdori.get_latent_topics`.
    config
        Training config (uses ``batch_size_cell_prediction``, ``tf_expression_mode``,
        ``cells_per_topic``, ``tf_expression_clamp``, ``umap_*`` and ``batch_col``).
    cell_indices : array-like of int, optional
        Cells used for topic usage and structure; default all cells.
    gate_indices : array-like of int, optional
        Cells used to preview the phase-2 TF-expression gate. Pass the phase-2
        training cells to match phase 2 exactly; default ``cell_indices``.
    groupby : str, optional
        ``rna_anndata.obs`` column with a user label (for example cell type). Optional.
    batch_key : str, optional
        ``rna_anndata.obs`` column with the batch/sample; default ``config.batch_col`` if present.
    dead_max_theta, dead_max_mean_fraction : float
        A topic is *dead* (collapsed) if its theta never reaches ``dead_max_theta`` in
        any cell **and** its mean theta is below ``dead_max_mean_fraction / K``. Only dead
        topics are eligible to be skipped in phase 2. A topic that never reaches
        ``dead_max_theta`` but still carries mass (a background topic, common with
        metacells) is flagged *diffuse* and kept.
    weak_min_dominant_fraction : float
        A kept topic is flagged *weak* if it is the dominant topic in fewer than
        ``weak_min_dominant_fraction * n_cells / K`` cells. Weak and diffuse topics are
        kept; the flags only ask for review.
    support_quantile : float
        Raw-support filter for gene programs: a gene is supported in a topic if its
        raw topic-gene signal is at least this quantile of the topic's raw values.
    structure : bool
        Compute neighbours, UMAP, Leiden and kNN metrics on theta (needs scanpy and igraph).
    out : str or Path, optional
        If given, write all tables and ``live_topics.json`` there.
    checkpoint : str or Path, optional
        Phase-1 checkpoint path, recorded with its SHA256 in ``live_topics.json``.

    Returns
    -------
    Phase1Diagnostics
    """
    import torch.utils.data as tud

    n_obs = rna_anndata.n_obs
    cells = np.arange(n_obs) if cell_indices is None else np.asarray(cell_indices)
    gate_cells = cells if gate_indices is None else np.asarray(gate_indices)
    if not rna_anndata.obs_names.equals(atac_anndata.obs_names):
        raise ValueError("RNA and ATAC must have identical cell IDs in the same order")
    for name, indices in (("cell_indices", cells), ("gate_indices", gate_cells)):
        if (
            indices.ndim != 1
            or indices.dtype.kind not in "iu"
            or not len(indices)
            or len(np.unique(indices)) != len(indices)
            or (indices < 0).any()
            or (indices >= n_obs).any()
        ):
            raise ValueError(f"{name} must be nonempty, unique integer cell indices within range")
    if not 0 <= support_quantile <= 1 or min(top_n_genes, top_n_tfs, max_detection_cells) < 1:
        raise ValueError("Invalid support quantile or nonpositive diagnostic list/sample size")
    loader = tud.DataLoader(tud.TensorDataset(torch.from_numpy(cells)), batch_size=config.batch_size_cell_prediction)
    theta = get_latent_topics(
        model, device, loader, rna_anndata, atac_anndata, num_cells, tf_indices, encoding_batch_onehot
    )
    K = theta.shape[1]
    topic_ids = [f"Topic_{k}" for k in range(K)]
    obs = rna_anndata.obs.iloc[cells]
    if batch_key is None and getattr(config, "batch_col", None) in rna_anndata.obs:
        batch_key = config.batch_col
    labels = None if groupby is None else obs[groupby].astype(str).to_numpy()

    qc = topic_usage_qc(theta, labels, dead_max_theta, dead_max_mean_fraction, weak_min_dominant_fraction)
    qc.index = topic_ids
    qc = qc.join(collapse_indicators(model).set_index(pd.Index(topic_ids)))
    if batch_key is not None:
        batch_mean = pd.DataFrame(theta, columns=topic_ids).groupby(obs[batch_key].astype(str).to_numpy()).mean()
        qc["batch_mean_cv"] = (batch_mean.std(0, ddof=0) / (batch_mean.mean(0) + 1e-12)).to_numpy()
    from scipy.stats import rankdata

    # Compute library-size confounding in batches instead of densifying all RNA.
    library = np.concatenate(
        [
            np.asarray(rna_anndata.X[np.sort(chunk)].sum(axis=1)).ravel()[np.argsort(np.argsort(chunk))]
            for chunk in np.array_split(cells, max(1, int(np.ceil(len(cells) / 512))))
        ]
    )
    for name, values in (("rna_library", library), ("num_cells", np.asarray(num_cells)[cells].ravel())):
        ranked = rankdata(values)
        qc[f"spearman_{name}"] = [
            float(np.corrcoef(ranked, rankdata(theta[:, k]))[0, 1])
            if np.ptp(values) > 0 and np.ptp(theta[:, k]) > 0
            else np.nan
            for k in range(K)
        ]
    live = np.flatnonzero(~qc["dead"].to_numpy())
    group_mean = None
    if labels is not None:
        group_mean = pd.DataFrame(theta, columns=topic_ids).groupby(labels).mean()

    diag = Phase1Diagnostics(
        topic_ids=topic_ids,
        theta=theta,
        topic_qc=qc,
        live_topics=live,
        summary={},
        guidance=[],
        cell_ids=cells.copy(),
        cell_names=list(obs.index.astype(str)),
        group_topic_mean=group_mean,
        settings={
            "n_cells": len(cells),
            "n_gate_cells": len(gate_cells),
            "groupby": groupby,
            "batch_key": batch_key,
            "dead_max_theta": dead_max_theta,
            "dead_max_mean_fraction": dead_max_mean_fraction,
            "weak_min_dominant_fraction": weak_min_dominant_fraction,
            "support_quantile": support_quantile,
            "top_n_genes": top_n_genes,
            "top_n_tfs": top_n_tfs,
            "tf_expression_mode": str(config.tf_expression_mode),
            "cells_per_topic": int(config.cells_per_topic),
            "tf_expression_clamp": float(config.tf_expression_clamp),
            "tf_expression_min_detection": float(getattr(config, "tf_expression_min_detection", 0.0)),
            "tf_expression_scaling": getattr(config, "tf_expression_scaling", "within_topic"),
            "effective_topics_definition": "exp(entropy(mean topic proportions across inspected cells))",
        },
    )
    if len(live):
        genes = np.asarray(rna_anndata.var_names)
        _gene_programs(
            diag,
            model,
            device,
            rna_anndata,
            cells,
            genes,
            support_quantile,
            top_n_genes,
            max_detection_cells,
            random_state,
        )
        tf_names = genes[np.asarray(tf_indices)]
        _tf_programs(diag, model, tf_names, top_n_tfs)
        _tf_gate(
            diag,
            model,
            device,
            rna_anndata,
            atac_anndata,
            num_cells,
            tf_indices,
            encoding_batch_onehot,
            config,
            cells,
            gate_cells,
            tf_names,
        )
    if structure:
        _structure(diag, obs, groupby, batch_key, config, leiden_resolution, random_state)
    diag.summary = _summary(diag)
    diag.guidance = phase1_guidance(diag)
    if out is not None:
        diag.save(out, checkpoint)
    return diag


# ---------------------------------------------------------------------------- tables


def effective_number_of_topics(theta):
    """exp(entropy) of the mean topic usage; equals K when usage is uniform."""
    p = np.asarray(theta, dtype=float).mean(0)
    p = p / p.sum()
    nz = p[p > 0]
    return float(np.exp(-(nz * np.log(nz)).sum()))


def topic_usage_qc(theta, labels=None, dead_max_theta=0.1, dead_max_mean_fraction=0.1, weak_min_dominant_fraction=0.1):
    """Per-topic usage table with ``dead``, ``diffuse`` and ``weak`` flags and a ``status`` column.

    Status is one of ``dead`` (collapsed: max theta < ``dead_max_theta`` and mean theta <
    ``dead_max_mean_fraction / K``), ``diffuse`` (never reaches ``dead_max_theta`` but carries
    mass), ``weak`` (dominant in fewer than ``weak_min_dominant_fraction * n / K`` cells) or
    ``live``. With ``labels``, dominant-cell purity and the groups most enriched for the topic
    (group mean over overall mean) are added.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 2 or not all(theta.shape) or not np.isfinite(theta).all() or (theta < 0).any():
        raise ValueError("theta must be a nonempty finite nonnegative cell-by-topic matrix")
    n, K = theta.shape
    dominant = theta.argmax(1)
    table = pd.DataFrame(
        {
            "topic_id": np.arange(K),
            "mean_theta": theta.mean(0),
            "max_theta": theta.max(0),
            "std_theta": theta.std(0),
            "q99_theta": np.quantile(theta, 0.99, axis=0),
            "dominant_cells": np.bincount(dominant, minlength=K),
        }
    )
    never_high = table["max_theta"] < dead_max_theta
    table["dead"] = never_high & (table["mean_theta"] < dead_max_mean_fraction / K)
    table["diffuse"] = never_high & ~table["dead"]
    table["weak"] = ~never_high & (table["dominant_cells"] < weak_min_dominant_fraction * n / K)
    if labels is not None:
        labels = np.asarray(labels)
        purity, top, enriched = np.full(K, np.nan), [""] * K, [""] * K
        group_mean = pd.DataFrame(theta).groupby(labels).mean()
        for k in range(K):
            counts = pd.Series(labels[dominant == k]).value_counts(normalize=True)
            if len(counts):
                purity[k] = counts.iloc[0]
                top[k] = "; ".join(f"{name}:{share:.2f}" for name, share in counts.head(3).items())
            enr = (group_mean[k] / (table.loc[k, "mean_theta"] + 1e-12)).sort_values(ascending=False)
            enriched[k] = "; ".join(f"{name}:{value:.1f}x" for name, value in enr.head(3).items())
        table["dominant_purity"] = purity
        table["dominant_groups"] = top
        table["enriched_groups"] = enriched
    table["status"] = np.select([table["dead"], table["diffuse"], table["weak"]], ["dead", "diffuse", "weak"], "live")
    return table


def collapse_indicators(model):
    """Decoder row statistics and the ``mu_theta`` bias per topic (collapsed topics shrink to ~0)."""
    with torch.no_grad():
        peak = model.topic_peak_decoder.detach().float().cpu()
        tf = model.topic_tf_decoder.detach().float().cpu()
        bias = model.mu_theta.bias.detach().float().cpu()
    return pd.DataFrame(
        {
            "peak_decoder_row_std": peak.std(1).numpy(),
            "peak_decoder_row_l2": peak.norm(dim=1).numpy(),
            "tf_decoder_row_std": tf.std(1).numpy(),
            "tf_decoder_row_l2": tf.norm(dim=1).numpy(),
            "mu_theta_bias": bias.numpy(),
        }
    )


def zscore_across_topics(matrix, rows=None, ddof=0, eps=1e-5):
    """Z-score each column across ``rows`` (the reference topics); other rows are NaN.

    ``ddof=0, eps=1e-5`` matches the gene z-score of the downstream code (fresh
    BatchNorm); ``ddof=1`` with ``eps=1e-8`` added to the std matches the TF z-scores.
    """
    matrix = np.asarray(matrix, dtype=float)
    rows = np.arange(matrix.shape[0]) if rows is None else np.asarray(rows)
    ref = matrix[rows]
    if ddof == 0:
        z = (ref - ref.mean(0)) / np.sqrt(ref.var(0) + eps)
    elif len(ref) > ddof:
        z = (ref - ref.mean(0)) / (ref.std(0, ddof=ddof) + eps)
    else:
        z = np.zeros_like(ref)
    out = np.full_like(matrix, np.nan)
    out[rows] = z
    return out


def _top_long(views, primary, topics, names, top_n, value_label):
    rows = []
    for k in topics:
        for view in primary:
            order = np.argsort(-np.nan_to_num(views[view][k], nan=-np.inf), kind="stable")[:top_n]
            order = [j for j in order if np.isfinite(views[view][k, j])]
            for rank, j in enumerate(order, 1):
                row = {"topic": f"Topic_{k}", "view": view, "rank": rank, value_label: names[j]}
                row.update({name: float(values[k, j]) for name, values in views.items()})
                rows.append(row)
    return pd.DataFrame(rows)


def _jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else np.nan


def _gene_programs(diag, model, device, rna, cells, genes, support_quantile, top_n, max_cells, random_state):
    from .downstream import compute_topic_gene_matrix

    live = diag.live_topics
    with torch.no_grad():
        raw = model.topic_gene_matrix().detach().float().cpu().numpy()
    support = (raw > 0) & (raw >= np.quantile(raw, support_quantile, axis=1, keepdims=True))
    z_live = zscore_across_topics(raw, live)
    views = {
        "raw": raw,
        "zscore_all_topics": zscore_across_topics(raw),
        "zscore_live_topics": z_live,
        "zscore_live_supported": np.where(support, z_live, np.nan),
    }
    bn = model.rna_batch_norm
    with torch.no_grad():
        gain = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().cpu().numpy()
    views["effective_eval_logits"] = raw * gain[None, :]
    softmax_live = np.full_like(raw, np.nan)
    softmax_live[live] = compute_topic_gene_matrix(model, device, selected_topics=live)
    views["downstream_softmax_live_topics"] = softmax_live
    ranked = ["zscore_all_topics", "zscore_live_topics", "zscore_live_supported"]
    top = _top_long(views, ranked, live, genes, top_n, "gene")
    top.insert(
        4,
        "raw_supported",
        [
            bool(support[int(t[6:]), g])
            for t, g in zip(top["topic"], pd.Index(genes).get_indexer(top["gene"]), strict=True)
        ],
    )

    # Generic plausibility: are the top genes detected more often in the cells the topic dominates?
    counts = rna.X
    rng = np.random.default_rng(random_state)
    dominant = diag.theta.argmax(1)
    background_cells = cells if len(cells) <= max_cells else rng.choice(cells, max_cells, replace=False)
    background = _detection(counts, np.sort(background_cells))
    metrics = []
    for k in live:
        name = f"Topic_{k}"
        sets = {v: top.loc[(top.topic == name) & (top.view == v), "gene"].tolist() for v in ranked}
        dom = cells[dominant == k]
        if len(dom) > max_cells:
            dom = rng.choice(dom, max_cells, replace=False)
        detection = _detection(counts, np.sort(dom)) if len(dom) >= 10 else None
        row = {
            "topic": name,
            "jaccard_all_vs_live": _jaccard(sets["zscore_all_topics"], sets["zscore_live_topics"]),
            "jaccard_live_vs_supported": _jaccard(sets["zscore_live_topics"], sets["zscore_live_supported"]),
        }
        for v in ranked:
            idx = pd.Index(genes).get_indexer(sets[v])
            row[f"support_fraction_{v}"] = float(support[k, idx].mean()) if len(idx) else np.nan
            row[f"detection_ratio_{v}"] = (
                float(np.mean(detection[idx]) / (np.mean(background[idx]) + 1e-12))
                if detection is not None and len(idx)
                else np.nan
            )
        metrics.append(row)
    diag.gene_views = {
        name: pd.DataFrame(values, index=diag.topic_ids, columns=genes) for name, values in views.items()
    }
    diag.gene_top = top
    diag.gene_view_metrics = pd.DataFrame(metrics).set_index("topic")


def _detection(counts, rows):
    x = counts[rows]
    if sp.issparse(x):
        return np.asarray((x > 0).mean(axis=0)).ravel()
    return (np.asarray(x) > 0).mean(axis=0)


def _tf_programs(diag, model, tf_names, top_n):
    live = diag.live_topics
    with torch.no_grad():
        share = torch.softmax(model.topic_tf_decoder.detach().float().cpu(), dim=1).numpy()
    views = {
        "decoder_share": share,
        # ddof=1 and +1e-8 match the across-topic TF z-score of the downstream code.
        "share_zscore_all_topics": zscore_across_topics(share, ddof=1, eps=1e-8),
        "share_zscore_live_topics": zscore_across_topics(share, live, ddof=1, eps=1e-8),
    }
    ranked = ["share_zscore_all_topics", "share_zscore_live_topics"]
    diag.tf_top = _top_long(views, ranked, live, tf_names, top_n, "tf")
    diag.tf_views = {name: pd.DataFrame(v, index=diag.topic_ids, columns=tf_names) for name, v in views.items()}
    jac = []
    for k in live:
        sel = diag.tf_top[diag.tf_top.topic == f"Topic_{k}"]
        jac.append(_jaccard(sel.loc[sel.view == ranked[0], "tf"], sel.loc[sel.view == ranked[1], "tf"]))
    diag.topic_qc["tf_top_jaccard_all_vs_live"] = np.nan
    diag.topic_qc.loc[[f"Topic_{k}" for k in live], "tf_top_jaccard_all_vs_live"] = jac


def _tf_gate(diag, model, device, rna, atac, num_cells, tf_indices, onehot, config, cells, gate_cells, tf_names):
    """Preview the configured phase-2 TF-expression multiplier and explain its zeros.

    The preview uses the same selection, scaling and detection rule as ``get_tf_expression``; theta can differ
    from a phase-2 run in the last float bits because cells are encoded in other batches.
    """
    mode = str(config.tf_expression_mode)
    K = len(diag.topic_ids)
    rows = pd.DataFrame(index=diag.topic_ids)
    keep = True
    if mode in ("True", "actual"):
        position = np.full(rna.n_obs, -1)
        position[cells] = np.arange(len(cells))
        if (position[gate_cells] >= 0).all():  # reuse theta already computed for these cells
            ids, topics = gate_cells, diag.theta[position[gate_cells]]
        else:
            import torch.utils.data as tud

            loader = tud.DataLoader(
                tud.TensorDataset(torch.from_numpy(gate_cells)), batch_size=config.batch_size_cell_prediction
            )
            topics, ids = get_latent_topics(
                model, device, loader, rna, atac, num_cells, tf_indices, onehot, return_indices=True
            )
        means, selected = topic_tf_expression_means(topics, ids, rna, tf_indices, config.cells_per_topic)
        row_of = np.full(rna.n_obs, -1)
        row_of[ids] = np.arange(len(ids))
        sel_theta = np.stack([topics[row_of[selected[:, k]], k] for k in range(K)])
        rows["gate_cells_mean_theta"] = sel_theta.mean(1)
        rows["gate_cells_min_theta"] = sel_theta.min(1)
        _, keep = topic_tf_detection_filter(selected, rna, tf_indices, config)
    elif mode == "latent":
        with torch.no_grad():
            means = torch.softmax(model.topic_tf_decoder.detach().float().cpu(), dim=1).numpy()
    else:
        raise ValueError(f"Unknown tf_expression_mode: {mode!r}")
    policy = getattr(config, "tf_expression_scaling", "within_topic")
    unthresholded = scale_topic_tf_expression(torch.as_tensor(means, dtype=torch.float32), 0, policy).numpy()
    scored = scale_topic_tf_expression(torch.as_tensor(means, dtype=torch.float32), config.tf_expression_clamp, policy).numpy()
    gate = scored * keep
    no_rna = means <= 0
    row_min = unthresholded == 0
    rows["gate_nonzero_tfs"] = (gate > 0).sum(1)
    rows["zero_no_rna"] = no_rna.sum(1)
    rows["zero_minmax_minimum"] = (row_min & ~no_rna).sum(1)
    rows["zero_below_cutoff"] = ((scored == 0) & ~row_min & ~no_rna).sum(1)
    rows["zero_below_detection"] = ((scored > 0) & (gate == 0)).sum(1)
    rows["status"] = diag.topic_qc["status"]
    diag.tf_gate = pd.DataFrame(gate, index=diag.topic_ids, columns=tf_names)
    diag.tf_gate_topics = rows
    for col in rows.columns.drop("status"):
        diag.topic_qc[col] = rows[col]


def _structure(diag, obs, groupby, batch_key, config, resolution, random_state):
    import anndata as ad
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    from .downstream import compute_neighbors_umap

    adata = ad.AnnData(X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs.copy())
    adata.obsm["X_scdori"] = diag.theta
    for k, name in enumerate(diag.topic_ids):
        adata.obs[name] = diag.theta[:, k]
    adata.obs["dominant_topic"] = pd.Categorical([diag.topic_ids[k] for k in diag.theta.argmax(1)])
    if len(obs) < 4:
        diag.structure = {"skipped": "UMAP/Leiden needs at least four inspected cells"}
        return
    neighbors = min(config.umap_n_neighbors, len(obs) - 1)
    compute_neighbors_umap(adata, neighbors, config.umap_min_dist, config.umap_random_state)
    metrics = {"n_neighbors": int(neighbors), "leiden_resolution": resolution}
    try:
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added="leiden",
            random_state=random_state,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
        metrics["n_leiden_clusters"] = int(adata.obs["leiden"].nunique())
    except ImportError as err:  # igraph missing: keep the rest of the report
        logger.warning(f"Leiden clustering skipped: {err}")
    graph = adata.obsp["connectivities"].tocsr()
    if groupby is not None:
        labels = adata.obs[groupby].astype(str)
        if "leiden" in adata.obs:
            metrics["ARI_leiden_vs_label"] = float(adjusted_rand_score(labels, adata.obs["leiden"]))
            metrics["NMI_leiden_vs_label"] = float(normalized_mutual_info_score(labels, adata.obs["leiden"]))
        metrics["ARI_dominant_topic_vs_label"] = float(adjusted_rand_score(labels, adata.obs["dominant_topic"]))
        metrics["NMI_dominant_topic_vs_label"] = float(
            normalized_mutual_info_score(labels, adata.obs["dominant_topic"])
        )
        metrics["knn_label_agreement"] = knn_same_fraction(graph, labels)
    if batch_key is not None:
        batches = adata.obs[batch_key].astype(str)
        metrics["knn_same_batch_fraction"] = knn_same_fraction(graph, batches)
        metrics["expected_same_batch_fraction_if_mixed"] = float((batches.value_counts(normalize=True) ** 2).sum())
        if "leiden" in adata.obs:
            metrics["NMI_leiden_vs_batch"] = float(normalized_mutual_info_score(batches, adata.obs["leiden"]))
    diag.structure = metrics
    diag.adata = adata


def knn_same_fraction(graph, labels):
    """Mean over cells of the fraction of graph neighbours sharing the cell's label."""
    codes = pd.Categorical(np.asarray(labels)).codes
    graph = sp.csr_matrix(graph)
    graph.data = np.ones_like(graph.data)
    degree = np.asarray(graph.sum(1)).ravel()
    rows = np.repeat(np.arange(graph.shape[0]), np.diff(graph.indptr))
    same = np.bincount(rows, weights=(codes[graph.indices] == codes[rows]), minlength=graph.shape[0])
    ok = degree > 0
    return float(np.mean(same[ok] / degree[ok]))


def _summary(diag):
    qc = diag.topic_qc
    summary = {
        "n_topics": len(qc),
        "n_live": int((~qc.dead).sum()),
        "n_dead": int(qc.dead.sum()),
        "n_diffuse": int(qc.diffuse.sum()),
        "n_weak": int(qc.weak.sum()),
        "dead_fraction": float(qc.dead.mean()),
        "effective_topics": effective_number_of_topics(diag.theta),
        "max_theta_of_dead_topics": float(qc.loc[qc.dead, "max_theta"].max()) if qc.dead.any() else None,
        # Largest theta mass any cell puts on dead topics: what skipping them in phase 2 drops.
        "max_dead_theta_mass_per_cell": float(diag.theta[:, qc.dead.to_numpy()].sum(1).max()) if qc.dead.any() else 0.0,
        "live_topics": [int(t) for t in diag.live_topics],
        "weak_topics": [int(t) for t in qc.loc[qc.weak, "topic_id"]],
        "diffuse_topics": [int(t) for t in qc.loc[qc.diffuse, "topic_id"]],
    }
    if diag.gene_view_metrics is not None and len(diag.gene_view_metrics):
        m = diag.gene_view_metrics
        summary["median_gene_jaccard_all_vs_live"] = float(m["jaccard_all_vs_live"].median())
        summary["median_support_fraction_all_topics_view"] = float(m["support_fraction_zscore_all_topics"].median())
        summary["median_support_fraction_live_view"] = float(m["support_fraction_zscore_live_topics"].median())
    if "tf_top_jaccard_all_vs_live" in qc:
        summary["median_tf_jaccard_all_vs_live"] = float(qc["tf_top_jaccard_all_vs_live"].median())
    if diag.tf_gate_topics is not None and qc.dead.any():
        g = diag.tf_gate_topics
        summary["dead_topics_with_nonzero_gate"] = int(((g.status == "dead") & (g.gate_nonzero_tfs > 0)).sum())
    return summary


def phase1_guidance(
    diag,
    max_dead_fraction=0.25,
    min_effective_fraction=0.5,
    batch_ratio=3.0,
    min_gate_enrichment=2.0,
    min_reference_overlap=0.9,
):
    """Heuristic guidance as ``(level, message)`` tuples; levels are ok, info, warn and error."""
    s, qc, out = diag.summary, diag.topic_qc, []
    K = s["n_topics"]
    if s["n_live"] == 0:
        return [
            (
                "error",
                "Every topic is collapsed (no cell reaches theta "
                f"{diag.settings['dead_max_theta']}). Phase 1 did not learn usable topics; check inputs "
                "and training before phase 2.",
            )
        ]
    if s["dead_fraction"] > max_dead_fraction:
        out.append(
            (
                "warn",
                f"{s['n_dead']} of {K} topics are dead (collapsed). Options: (a) continue with "
                "grn_topic_selection='live', so phase 2 and the permutations skip them; (b) retrain "
                "phase 1 with fewer topics or another seed, using separate output directories. "
                "Compare validation reconstruction and stability; the live count alone does not identify the best K.",
            )
        )
    elif s["n_dead"]:
        out.append(
            (
                "info",
                f"{s['n_dead']} of {K} topics are dead (collapsed); grn_topic_selection='live' skips them in phase 2.",
            )
        )
    else:
        out.append(("ok", "No topic is collapsed."))
    if s["n_dead"]:
        out.append(
            (
                "info",
                f"Skipping the dead topics drops at most {s['max_dead_theta_mass_per_cell']:.2g} of "
                "any inspected cell's topic mass. This is not a bound on reconstruction or GRN changes. "
                "Topic skipping requires fully frozen phase 1.",
            )
        )
    if s["effective_topics"] < min_effective_fraction * K:
        out.append(
            (
                "warn",
                f"Effective number of topics is {s['effective_topics']:.1f} of {K}: usage is "
                "concentrated in few topics.",
            )
        )
    if s["n_diffuse"]:
        out.append(
            (
                "info",
                "Diffuse topics (never above theta "
                f"{diag.settings['dead_max_theta']} but carrying mass in many cells): "
                f"{', '.join(qc.index[qc.diffuse])}. They are kept in phase 2.",
            )
        )
    if s["n_weak"]:
        out.append(
            (
                "info",
                "Weak topics (dominant in few cells): "
                + ", ".join(qc.index[qc.weak])
                + ". They are kept; review them before interpreting their programs.",
            )
        )
    st = diag.structure
    if "knn_same_batch_fraction" in st and st["expected_same_batch_fraction_if_mixed"] > 0:
        ratio = st["knn_same_batch_fraction"] / st["expected_same_batch_fraction_if_mixed"]
        if ratio > batch_ratio:
            out.append(
                (
                    "info",
                    f"Neighbours share the batch {ratio:.1f}x more often than under full mixing. "
                    "Check whether batch is confounded with biology before judging this.",
                )
            )
    m = diag.gene_view_metrics
    if s["n_dead"] and m is not None and len(m):
        overlap = min(s["median_gene_jaccard_all_vs_live"], s.get("median_tf_jaccard_all_vs_live", 1.0))
        effect = "change" if overlap < min_reference_overlap else "barely change"
        out.append(
            (
                "info" if effect == "change" else "ok",
                f"Dead topics {effect} the across-topic scaling: median overlap (Jaccard) of top genes between "
                f"the all-topic and live-topic reference is {s['median_gene_jaccard_all_vs_live']:.2f}, of top "
                f"TFs {s.get('median_tf_jaccard_all_vs_live', np.nan):.2f}. Use the live topics as the scaling "
                "reference for gene and TF rankings downstream.",
            )
        )
    g = diag.tf_gate_topics
    if g is not None and "gate_cells_mean_theta" in g:
        if s.get("dead_topics_with_nonzero_gate"):
            worst = g.loc[g.status == "dead", "gate_cells_mean_theta"].max()
            out.append(
                (
                    "info",
                    f"{s['dead_topics_with_nonzero_gate']} dead topics would still get a non-zero "
                    "TF-expression gate in phase 2 (default), built from cells that barely use them "
                    f"(mean theta of their gate cells <= {worst:.2g}).",
                )
            )
        enrichment = g["gate_cells_mean_theta"] / (qc["mean_theta"] + 1e-12)
        low = g.index[(g.status != "dead") & (enrichment < min_gate_enrichment)]
        if len(low):
            out.append(
                (
                    "info",
                    f"Kept topics whose TF-gate cells use the topic less than {min_gate_enrichment:g}x "
                    "its average (the gate mostly reflects other programs): " + ", ".join(low),
                )
            )
    return out


# ---------------------------------------------------------------------------- live-topic file


def write_live_topics(path, record):
    """Write a live-topic record (see :meth:`Phase1Diagnostics.live_topic_record`)."""
    validate_live_topic_record(record)
    Path(path).write_text(json.dumps(record, indent=1))


def validate_live_topic_record(record, num_topics=None):
    """Check a live-topic record; returns the live topic IDs as a sorted int array."""
    if record.get("format") != LIVE_TOPICS_FORMAT:
        raise ValueError(f"Unsupported live-topic record format: {record.get('format')!r}")
    K = int(record["num_topics"])
    if num_topics is not None and K != num_topics:
        raise ValueError(f"Live-topic record is for {K} topics, but the model has {num_topics}")
    live = np.asarray(record["live_topics"], dtype=int)
    values = np.asarray(record["live_topics"])
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError("live_topics must contain integer topic IDs")
    if not len(live) or len(np.unique(live)) != len(live) or (live < 0).any() or (live >= K).any():
        raise ValueError("live_topics must be nonempty, unique and within range")
    return np.sort(live)


def read_live_topics(path, num_topics=None, checkpoint=None):
    """Read ``live_topics.json``; optionally check the topic count and the phase-1 checkpoint SHA256."""
    record = json.loads(Path(path).read_text())
    live = validate_live_topic_record(record, num_topics)
    if checkpoint is not None:
        if "phase1_checkpoint_sha256" not in record:
            raise ValueError("Live-topic record is missing the phase-1 checkpoint checksum")
        if _sha256(checkpoint) != record["phase1_checkpoint_sha256"]:
            raise ValueError(f"{path} was computed from a different phase-1 checkpoint than {checkpoint}")
    return live


def _sha256(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, np.integer | np.floating | np.bool_):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)}")
