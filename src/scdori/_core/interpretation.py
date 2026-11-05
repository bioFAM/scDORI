"""Explicit normalization for interpretation, independent of model training."""

import numpy as np
import pandas as pd


def _matrix(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Expected a nonempty DataFrame with named rows and features")
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("Row and feature identifiers must be unique")
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Scores must be finite")
    return values


def _frame(values, original):
    return pd.DataFrame(values, index=original.index, columns=original.columns)


def gene_importance_scores(raw):
    """Reproduce downstream fresh BatchNorm + softmax, with explicit axes.

    Rows are topics; columns are genes. Population variance and epsilon 1e-5
    match a fresh default BatchNorm1d in training mode. All supplied topics
    participate equally, including unused topics, matching the original helper.
    These are relative importance scores, not decoder expression predictions.
    """
    values = _matrix(raw)
    z = (values - values.mean(axis=0)) / np.sqrt(values.var(axis=0) + 1e-5)
    exp = np.exp(z - z.max(axis=1, keepdims=True))
    return {
        "raw": raw.copy(),
        "across_topics_zscore": _frame(z, raw),
        "downstream_softmax": _frame(exp / exp.sum(axis=1, keepdims=True), raw),
    }


def tf_importance_scores(raw):
    """Return complementary views of nonnegative group-by-TF contributions.

    Each contribution is first divided by its group's total. Across-group sample
    z-scores of these shares reproduce the existing top-TF helper (all groups
    supplied). Min-max consensus is the mean of min-max scaling of the shares along
    the two axes, as in the paper. It is a heuristic in [0, 1], not a confidence or
    expression gate.
    """
    values = _matrix(raw)
    if (values < 0).any():
        raise ValueError("Use separate nonnegative activator/repressor magnitudes")
    share = values / (values.sum(axis=1, keepdims=True) + 1e-8)
    std = share.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(values.shape[1])
    z = (share - share.mean(axis=0)) / (std + 1e-8)
    row_range = np.ptp(share, axis=1, keepdims=True)
    col_range = np.ptp(share, axis=0, keepdims=True)
    row_mm = np.divide(share - share.min(axis=1, keepdims=True), row_range, out=np.zeros_like(share), where=row_range > 0)
    col_mm = np.divide(share - share.min(axis=0, keepdims=True), col_range, out=np.zeros_like(share), where=col_range > 0)
    return {
        "raw": raw.copy(),
        "within_group_share": _frame(share, raw),
        "share_across_groups_zscore": _frame(z, raw),
        "within_group_minmax": _frame(row_mm, raw),
        "across_groups_minmax": _frame(col_mm, raw),
        "minmax_consensus": _frame((row_mm + col_mm) / 2, raw),
    }


def _rank_by_score_then_raw(values, raw):
    """1-based ranks by descending score; equal scores go to the larger raw contribution, then input order."""
    order = np.lexsort((np.arange(len(values)), -raw.to_numpy(dtype=np.float64), -values.to_numpy(dtype=np.float64)))
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return pd.Series(ranks, index=values.index)


def top_features(scores, ranking, top_n=30):
    """Keep all score views beside each ranked feature.

    Equal scores are ordered by the larger raw contribution when ``scores`` has a ``"raw"`` view,
    then by input order, so a negligible feature never outranks a large one on a tie.
    """
    if top_n < 1:
        raise ValueError("top_n must be positive")
    frame = scores[ranking]
    raw = scores.get("raw", frame)
    rows = []
    for group in frame.index:
        ranks = _rank_by_score_then_raw(frame.loc[group], raw.loc[group])
        for rank, feature in enumerate(ranks.sort_values(kind="stable").head(top_n).index, 1):
            rows.append(
                {
                    "group": group,
                    "feature": feature,
                    "rank": rank,
                    **{name: float(value.loc[group, feature]) for name, value in scores.items()},
                }
            )
    return pd.DataFrame(rows)


def compare_tf_rankings(scores, top_n=10, topics=None):
    """Show membership and ranks in the two min-max lists and their arithmetic mean.

    Scores must already use the desired reference topics. ``topics`` only selects
    display rows; it never recomputes scaling. Rank TFs with positive raw support
    only. Equal scores are common in the across-topic view (every TF scores 1 in its
    own strongest topic); they are resolved by the larger raw contribution, then by
    TF-column order. Call separately for nonnegative activator and repressor magnitudes.
    """
    if not isinstance(top_n, (int, np.integer)) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    views = {"within": "within_group_minmax", "across": "across_groups_minmax", "mean": "minmax_consensus"}
    raw = scores["raw"]
    if (_matrix(raw) < 0).any():
        raise ValueError("Compare activator and repressor magnitudes separately; raw scores must be nonnegative")
    for name in views.values():
        frame = scores[name]
        _matrix(frame)
        if not frame.index.equals(raw.index) or not frame.columns.equals(raw.columns):
            raise ValueError("Ranking views must have the same topic and TF order")
    topics = list(raw.index if topics is None else topics)
    if len(set(topics)) != len(topics) or not set(topics).issubset(raw.index):
        raise ValueError("Display topics must be unique and belong to the score reference")
    columns = ["topic", "tf", "raw"] + [f"{axis}_{suffix}" for axis in views for suffix in ("score", "rank", "top")]
    records = []
    for topic in topics:
        supported = raw.loc[topic] > 0
        ranks = {
            axis: _rank_by_score_then_raw(scores[name].loc[topic, supported], raw.loc[topic, supported])
            for axis, name in views.items()
        }
        union = [tf for tf in ranks["mean"].index if any(rank[tf] <= top_n for rank in ranks.values())]
        for tf in sorted(union, key=lambda tf: ranks["mean"][tf]):
            row = {"topic": topic, "tf": tf, "raw": float(raw.loc[topic, tf])}
            for axis, name in views.items():
                row[f"{axis}_score"] = float(scores[name].loc[topic, tf])
                row[f"{axis}_rank"] = int(ranks[axis][tf])
                row[f"{axis}_top"] = bool(ranks[axis][tf] <= top_n)
            records.append(row)
    return pd.DataFrame(records, columns=columns)
