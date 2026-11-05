"""Optional evidence checks for TF displays; never modify fitted networks."""

import numpy as np
import pandas as pd
from scipy import sparse


def topic_expression_support(rna, cell_topics, tf_names, cells_per_topic=100):
    """Measure TF RNA in each topic's highest-weight cells.

    ``rna.X`` must contain nonnegative integer counts. Means are counts per
    10,000 counts across ALL genes supplied in rna (often a selected gene panel),
    not across TFs only. These are depth-normalized measurements, not absolute
    molecules. Zero-depth cells contribute zero. Detection uses original counts.
    Only selected cell blocks are read, including for backed AnnData.
    """
    if not isinstance(cells_per_topic, (int, np.integer)) or isinstance(cells_per_topic, bool) or cells_per_topic < 1:
        raise ValueError("cells_per_topic must be a positive integer")
    if not cell_topics.index.equals(rna.obs_names) or not cell_topics.index.is_unique:
        raise ValueError("cell_topics rows must match unique RNA cell names in order")
    if not cell_topics.columns.is_unique or cell_topics.empty:
        raise ValueError("Supply nonempty, uniquely named topics")
    theta = cell_topics.to_numpy()
    if not np.isfinite(theta).all() or (theta < 0).any():
        raise ValueError("Topic weights must be finite and nonnegative")
    names = pd.Index(tf_names)
    if not names.is_unique or not len(names) or not rna.var_names.is_unique:
        raise ValueError("TF and RNA gene names must be unique and nonempty")
    indices = rna.var_names.get_indexer(names)
    if (indices < 0).any():
        raise ValueError("Every TF must be present in RNA")
    k = min(cells_per_topic, rna.n_obs)
    means, detected, selected = [], [], {}
    for j, topic in enumerate(cell_topics.columns):
        rows = np.sort(np.argsort(theta[:, j], kind="stable")[-k:])
        block = rna.X[rows, :]
        values = block.data if sparse.issparse(block) else np.asarray(block)
        if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values, np.rint(values), rtol=0, atol=1e-6):
            raise ValueError("RNA X must contain untransformed nonnegative integer counts")
        totals = np.asarray(block.sum(axis=1)).reshape(-1)
        tf = block[:, indices]
        tf = tf.toarray() if sparse.issparse(tf) else np.asarray(tf)
        normalized = np.divide(tf * 1e4, totals[:, None], out=np.zeros(tf.shape, dtype=float), where=totals[:, None] > 0)
        means.append(normalized.mean(axis=0))
        detected.append((tf > 0).mean(axis=0))
        selected[topic] = list(map(str, rna.obs_names[rows]))
    return {
        "mean_counts_per_10k": pd.DataFrame(means, index=cell_topics.columns, columns=names),
        "detection_fraction": pd.DataFrame(detected, index=cell_topics.columns, columns=names),
        "selected_cells": selected,
        "normalization_gene_count": int(rna.n_vars),
    }


def filter_tf_display(scores, *, expression=None, min_share=0.0, min_mean=0.0, min_detection=0.0):
    """Return eligibility and a full audit without rescaling scores.

    Call separately for activator and repressor magnitudes. ``min_share`` is a
    fraction of total contribution for that sign within the topic. Expression
    cutoffs use the same observed RNA for both signs. All cutoffs are inclusive;
    zero raw contribution is always excluded. Zeros disable optional filters.
    """
    for name, value, maximum in [("min_share", min_share, 1), ("min_mean", min_mean, np.inf), ("min_detection", min_detection, 1)]:
        if not np.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f"Invalid {name}")
    raw, ranking = scores["raw"], scores["minmax_consensus"]
    if raw.empty or not raw.index.is_unique or not raw.columns.is_unique:
        raise ValueError("Supply a nonempty, uniquely labelled contribution matrix")
    if not raw.index.equals(ranking.index) or not raw.columns.equals(ranking.columns):
        raise ValueError("Ranking labels must match raw contributions")
    if not np.isfinite(raw.to_numpy()).all() or (raw.to_numpy() < 0).any() or not np.isfinite(ranking.to_numpy()).all():
        raise ValueError("Scores must be finite; raw contributions must be nonnegative")
    if expression is None and (min_mean or min_detection):
        raise ValueError("Observed expression is required for expression filters")
    expression_frames = {}
    if expression is not None:
        for name in ["mean_counts_per_10k", "detection_fraction"]:
            frame = expression[name].reindex(index=raw.index, columns=raw.columns)
            if not np.isfinite(frame.to_numpy()).all() or (frame.to_numpy() < 0).any():
                raise ValueError("Expression evidence must cover every topic and TF with finite nonnegative values")
            if name == "detection_fraction" and (frame.to_numpy() > 1).any():
                raise ValueError("Detection fractions must be at most one")
            expression_frames[name] = frame
    share = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    eligible = raw > 0
    reasons = {"no retained contribution": raw <= 0, "small contribution": share < min_share}
    if min_mean:
        reasons["low mean expression"] = expression_frames["mean_counts_per_10k"] < min_mean
    if min_detection:
        reasons["rarely detected"] = expression_frames["detection_fraction"] < min_detection
    for failed in reasons.values():
        eligible &= ~failed
    before = ranking.where(raw > 0).rank(axis=1, ascending=False, method="first")
    after = ranking.where(eligible).rank(axis=1, ascending=False, method="first")
    records = []
    for topic in raw.index:
        for tf in raw.columns:
            records.append({"topic": topic, "tf": tf, "raw_contribution": raw.loc[topic, tf],
                            "contribution_share": share.loc[topic, tf], "mean_score": ranking.loc[topic, tf],
                            **{name: frame.loc[topic, tf] for name, frame in expression_frames.items()},
                            "included": bool(eligible.loc[topic, tf]),
                            "reason": "; ".join(name for name, failed in reasons.items() if failed.loc[topic, tf]),
                            "rank_before": before.loc[topic, tf], "rank_after": after.loc[topic, tf]})
    return eligible, pd.DataFrame(records)
