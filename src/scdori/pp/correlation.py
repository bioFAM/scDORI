import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

logger = logging.getLogger(__name__)

CORRELATION_NORMALIZATIONS = ("mean_log", "none", "log_normalized")


def _log_normalize_rows(matrix, target_sum):
    """log1p of each row scaled to ``target_sum``; empty rows stay zero."""
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    matrix = matrix.astype(np.float64, copy=False)
    totals = matrix.sum(axis=1, keepdims=True)
    scaled = np.divide(matrix * target_sum, totals, out=np.zeros_like(matrix), where=totals > 0)
    return np.log1p(scaled)


def metacell_correlation_matrices(rna_metacell, atac_metacell, peak_names, normalization="mean_log"):
    """TF expression and peak accessibility used for the in-silico ChIP-seq correlation.

    ``"mean_log"`` (default) uses the mean of each cell's log-normalized RNA, from the ``mean_log``
    layer, and mean ATAC fragments. ``"none"`` uses the mean counts of both. ``"log_normalized"``
    removes differences in sequencing depth between metacells first: RNA rows are scaled to 10,000
    counts over all genes in ``rna_metacell`` and ATAC rows to 1,000,000 counts over all peaks
    in ``atac_metacell``, before selecting TFs and ``peak_names``, then log1p-transformed. The
    result is the same from summed counts as from means, because each row is rescaled. Only
    the correlation input changes; selected features and saved training matrices do not.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Metacells × TFs and metacells × ``peak_names``.
    """
    if normalization not in CORRELATION_NORMALIZATIONS:
        raise ValueError(f"correlation_normalization must be one of {CORRELATION_NORMALIZATIONS}")
    columns = atac_metacell.var_names.get_indexer(list(peak_names))
    if (columns < 0).any():
        raise ValueError("Every selected peak must be present in the ATAC metacells")
    tf_mask = (rna_metacell.var["gene_type"] == "TF").to_numpy()
    rna, atac = rna_metacell.X, atac_metacell.X
    if normalization == "mean_log":
        if "mean_log" not in rna_metacell.layers:
            raise ValueError("These RNA metacells have no mean_log layer; rebuild them with create_metacells or aggregate_metacells")
        rna = rna_metacell.layers["mean_log"]
    if normalization == "log_normalized":
        rna, atac = _log_normalize_rows(rna, 1e4), _log_normalize_rows(atac, 1e6)
    return rna[:, tf_mask], atac[:, columns]


def compute_in_silico_chipseq(
    atac_matrix: np.ndarray,
    rna_matrix: np.ndarray,
    motif_scores: pd.DataFrame,
    percentile: float = 99.9,
    n_bg: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute correlation-based in-silico ChIP-seq embeddings, separating activator and repressor signals.

    Steps
    -----
    1. Compute the Pearson correlation for each (peak, TF) pair.
    2. Retain only the positive part as an "activator" correlation and the negative part as a "repressor" correlation.
    3. For each TF, estimate significance thresholds for correlations from background peaks (i.e., peaks with minimal motif scores for each TF).
    4. Multiply thresholded correlations by the motif scores to produce final activator and repressor embeddings.
    5. Scale activators to [0, 1] and repressor magnitudes to [-1, 0], preserving absent evidence.

    Parameters
    ----------
    atac_matrix : np.ndarray
        Shape (n_metacells, n_peaks). Matrix of peak accessibility (or insertion counts)
        where each row is a metacell and each column is a peak.
    rna_matrix : np.ndarray
        Shape (n_metacells, n_tfs). Matrix of TF expression where each row is a metacell
        and each column is a TF.
    motif_scores : pd.DataFrame
        Shape (n_peaks, n_tfs). DataFrame containing motif scores for each (peak, TF) pair.
    percentile : float, optional
        The percentile threshold used to determine correlation significance (for activators
        and repressors). Default is 99.9.
    n_bg : int, optional
        Number of peaks per TF used to form the "background" distribution (lowest motif
        scores). Default is 10000.

    Returns
    -------
    insilico_chipseq_act_sig : np.ndarray
        Shape (n_peaks, n_tfs). Final activator scores after thresholding correlations
        and combining with motif scores. Scaled to [0, 1].
    insilico_chipseq_rep_sig : np.ndarray
        Shape (n_peaks, n_tfs). Final repressor scores after thresholding correlations
        and combining with motif scores. Scaled to [-1, 0].

    Notes
    -----
    - Positive correlation is interpreted as activator-like; negative correlation is repressor-like.
    - A background distribution is formed from peaks with the smallest motif scores, then used to find correlation thresholds at the requested percentile.
    - The final arrays are scaled per TF:
      * The repressor array is divided by its maximum absolute value, preserving zero columns.
      * The activator array is scaled into the range [0, 1].
    """
    logger.info("Computing in-silico ChIP-seq correlation...")

    n_cells, n_peaks = atac_matrix.shape
    _, n_tfs = rna_matrix.shape
    if motif_scores.shape != (n_peaks, n_tfs) or rna_matrix.shape[0] != n_cells:
        raise ValueError("RNA, ATAC and motif dimensions must agree")
    if n_cells < 2 or n_bg < 1 or not 0 < percentile < 100:
        raise ValueError("Need at least two metacells, positive n_bg and percentile in (0, 100)")

    # Z-score peaks & TF expression
    X = (atac_matrix - atac_matrix.mean(axis=0)) / (atac_matrix.std(axis=0) + 1e-8)
    Y = (rna_matrix - rna_matrix.mean(axis=0)) / (rna_matrix.std(axis=0) + 1e-8)

    # Pearson correlation => (n_peaks x n_tfs)
    pearson_r = (X.T @ Y) / n_cells
    pearson_r = np.nan_to_num(pearson_r)

    pearson_r_act = np.clip(pearson_r, 0, None)  # only positive
    pearson_r_rep = np.clip(pearson_r, None, 0)  # only negative

    pearson_r_act_sig = np.zeros_like(pearson_r_act)
    pearson_r_rep_sig = np.zeros_like(pearson_r_rep)

    tf_list = motif_scores.columns

    # Thresholding
    for t in tqdm(range(n_tfs), desc="Thresholding correlation"):
        tf_name = tf_list[t]
        # Find background peaks with smallest motif score
        scores_t = motif_scores[tf_name].values
        order = np.argsort(scores_t)
        bg_idx = order[: min(n_bg, n_peaks)]  # top n_bg smallest motif peaks

        # Activator significance
        bg_vals_act = pearson_r_act[bg_idx, t]
        cutoff_act = np.percentile(bg_vals_act, percentile)

        # Repressor significance
        bg_vals_rep = pearson_r_rep[bg_idx, t]
        cutoff_rep = np.percentile(bg_vals_rep, 100 - percentile)

        act_vec = pearson_r_act[:, t]
        rep_vec = pearson_r_rep[:, t]

        pearson_r_act_sig[:, t] = np.where(act_vec > cutoff_act, act_vec, 0)
        pearson_r_rep_sig[:, t] = np.where(rep_vec < cutoff_rep, rep_vec, 0)

    # Combine with motif
    insilico_chipseq_act_sig = motif_scores.values * pearson_r_act_sig
    insilico_chipseq_rep_sig = motif_scores.values * pearson_r_rep_sig

    # Keep zero as no evidence. MinMaxScaler((-1, 0)) incorrectly maps an
    # all-zero column to -1, inventing repressor binding at every peak.
    scale = np.abs(insilico_chipseq_rep_sig).max(axis=0, keepdims=True)
    insilico_chipseq_rep_sig = np.divide(
        insilico_chipseq_rep_sig, scale, out=np.zeros_like(insilico_chipseq_rep_sig), where=scale > 0
    )

    # Scale activator to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    insilico_chipseq_act_sig = scaler.fit_transform(insilico_chipseq_act_sig)

    logger.info("Finished in-silico ChIP-seq computation.")
    return insilico_chipseq_act_sig, insilico_chipseq_rep_sig
