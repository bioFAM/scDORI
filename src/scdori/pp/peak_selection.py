import logging

import anndata as ad
import numpy as np

from scdori._counts import ATAC_COUNT_MODE_KEY, as_fragment_counts, resolve_atac_count_mode

logger = logging.getLogger(__name__)


def retain_requested_peaks(data_atac, windowed_peaks, user_peaks=None):
    """Keep gene-window peaks plus requested input peaks, preserving input order and counts.

    Requested peaks outside the windows remain available for ATAC modelling;
    this does not create peak–gene links beyond the configured windows.
    """
    requested = set(user_peaks or [])
    windowed_peaks = set(windowed_peaks)
    available = set(data_atac.var_names)
    missing = sorted(requested - available)
    outside = sorted((requested & available) - set(windowed_peaks))
    if missing:
        logger.warning("Requested peaks absent from ATAC input (not added): %s", missing)
    if outside:
        logger.warning("Keeping requested peaks outside selected gene windows; no gene links are added: %s", outside)
    keep = set(windowed_peaks) | (requested & available)
    result = data_atac[:, data_atac.var_names.isin(keep)].copy()
    result.uns["scdori_requested_peaks"] = {
        "requested": sorted(requested), "missing": missing, "outside_gene_windows": outside,
        "retained": sorted(requested & available),
    }
    return result


def select_highly_variable_peaks_by_std(
    data_atac: ad.AnnData, n_top_peaks: int, cluster_key: str = "leiden", atac_count_mode: str | None = None
) -> ad.AnnData:
    """
    Select highly variable peaks based on the standard deviation of peak accessibility across clusters.

    This function:
    1. Groups cells by `cluster_key` in `data_atac.obs`.
    2. Computes ATAC fragment counts and calculates mean accessibility
        per peak for each cluster.
    3. Computes the standard deviation of each peak's accessibility across clusters.
    4. Selects the top `n_top_peaks` peaks with the highest standard deviation.

    Parameters
    ----------
    data_atac : anndata.AnnData
        An AnnData containing ATAC data. Expects `.obs[cluster_key]` to exist.
    n_top_peaks : int
        Number of peaks to retain based on the highest standard deviation across clusters.
    cluster_key : str, optional
        The column in `data_atac.obs` specifying cluster labels. Defaults to "leiden".

    atac_count_mode : {"insertions", "fragments"}, optional
        Input convention. Uses saved metadata, otherwise legacy insertions.
        X is preserved; the convention is saved for downstream training.

    Returns
    -------
    anndata.AnnData
        A subset of `data_atac` containing only the selected peaks.
        If `n_top_peaks >= data_atac.shape[1]`, returns the original data.

    Notes
    -----
    - If `cluster_key` is missing, a warning is logged and the original AnnData is returned.
    - Only insertion inputs are transformed with `(X + 1) // 2` for scoring.
    """
    mode = resolve_atac_count_mode(data_atac, atac_count_mode)
    data_atac.uns[ATAC_COUNT_MODE_KEY] = mode
    if cluster_key not in data_atac.obs.columns:
        logger.warning(f"{cluster_key} not found in data_atac.obs; skipping peak selection.")
        return data_atac

    clusters = data_atac.obs[cluster_key].unique()
    cluster_groups = data_atac.obs.groupby(cluster_key, observed=True)
    mean_list = []

    for c_label in clusters:
        idx_cells = cluster_groups.get_group(c_label).index
        mat = as_fragment_counts(data_atac[idx_cells].X, mode)
        mean_vec = np.asarray(mat.mean(axis=0)).ravel()
        mean_list.append(mean_vec)

    # shape => (n_clusters, n_peaks)
    cluster_matrix = np.vstack(mean_list)
    stdev_peaks = cluster_matrix.std(axis=0)
    data_atac.var["std_cluster"] = stdev_peaks

    if n_top_peaks < data_atac.shape[1]:
        sorted_idx = np.argsort(stdev_peaks)[::-1]
        keep_idx = sorted_idx[:n_top_peaks]
        mask = np.zeros(data_atac.shape[1], dtype=bool)
        mask[keep_idx] = True
        data_atac_sub = data_atac[:, mask].copy()
        logger.info(f"Selected top {n_top_peaks} variable peaks (by std across {cluster_key}).")
        return data_atac_sub
    else:
        logger.info("n_top_peaks >= total peaks; no filtering applied.")
        return data_atac


def keep_promoters_and_select_hv_peaks(
    data_atac: ad.AnnData,
    total_n_peaks: int,
    cluster_key: str = "leiden",
    promoter_col: str = "is_promoter",
    atac_count_mode: str | None = None,
    user_peaks: list[str] | None = None,
    keep_promoters: bool = True,
) -> ad.AnnData:
    """
    Retain requested peaks and optional promoters, then fill with variable peaks.

    Steps
    -----
    1. Identify peaks marked as promoters where `var[promoter_col] == True`.
    2. Keep all promoter peaks unconditionally.
    3. Among non-promoter peaks, select the top (total_n_peaks - #promoters) peaks by standard deviation across clusters.
    4. If the number of promoter peaks alone is >= `total_n_peaks`, keep all promoters.
    5. Combine the sets of promoter and HV non-promoter peaks.

    Parameters
    ----------
    data_atac : anndata.AnnData
        An AnnData containing ATAC data, with a boolean promoter column in `.var`.
    total_n_peaks : int
        The target total number of peaks to keep. May be exceeded if promoter peaks alone
        surpass this number.
    cluster_key : str, optional
        Column in `data_atac.obs` defining cluster labels for HV peak selection. Default is "leiden".
    promoter_col : str, optional
        Column in `data_atac.var` indicating which peaks are promoters. Default is "is_promoter".

    atac_count_mode : {"insertions", "fragments"}, optional
        Input convention. Uses saved metadata, otherwise legacy insertions.
        X is preserved; the convention is saved for downstream training.

    user_peaks : list of str, optional
        Exact input peak names to keep regardless of variability. Missing names
        are reported. Requested peaks and promoters may exceed total_n_peaks.
    keep_promoters : bool
        Retain annotated promoters as well as requested peaks; default True.

    Returns
    -------
    anndata.AnnData
        A subset of `data_atac` containing all promoter peaks plus HV non-promoter peaks.

    Notes
    -----
    - If `promoter_col` is missing, falls back to standard HV peak selection (without promoter logic).
    - The standard deviation for HV peaks is computed by `select_highly_variable_peaks_by_std`.
    """
    mode = resolve_atac_count_mode(data_atac, atac_count_mode)
    data_atac.uns[ATAC_COUNT_MODE_KEY] = mode
    if total_n_peaks < 0:
        raise ValueError("total_n_peaks must be nonnegative")
    if user_peaks or not keep_promoters:
        requested = set(user_peaks or [])
        missing = sorted(requested - set(data_atac.var_names))
        if missing:
            logger.warning("Requested peaks unavailable at peak selection: %s", missing)
        forced = data_atac.var_names.isin(requested)
        if keep_promoters and promoter_col in data_atac.var:
            forced |= data_atac.var[promoter_col].eq(True).to_numpy()
        n_needed = max(0, total_n_peaks - int(forced.sum()))
        candidates = data_atac[:, ~forced].copy()
        selected = set()
        if n_needed and candidates.n_vars:
            selected = set(select_highly_variable_peaks_by_std(candidates, n_needed, cluster_key, mode).var_names)
        if forced.sum() > total_n_peaks:
            logger.warning("Requested/promoter peaks exceed num_peaks; retaining all %d required peaks", forced.sum())
        return data_atac[:, forced | data_atac.var_names.isin(selected)].copy()
    if promoter_col not in data_atac.var.columns:
        logger.warning(f"Column {promoter_col} not found in data_atac.var; no special promoter logic.")
        return select_highly_variable_peaks_by_std(data_atac, total_n_peaks, cluster_key, mode)

    # (A) Extract promoter vs non-promoter
    promoter_mask = data_atac.var[promoter_col].eq(True).to_numpy()
    promoter_peaks = data_atac.var_names[promoter_mask]
    n_promoters = len(promoter_peaks)

    logger.info(f"Found {n_promoters} promoter peaks. Target total is {total_n_peaks}.")

    if n_promoters >= total_n_peaks:
        logger.warning(
            f"Promoter peaks ({n_promoters}) exceed total_n_peaks={total_n_peaks}. "
            "Keeping all promoters, final set might exceed user target."
        )
        data_atac_sub = data_atac[:, promoter_peaks].copy()
        return data_atac_sub
    else:
        # (B) Keep all promoters, then select HV among non-promoters
        n_needed = total_n_peaks - n_promoters
        logger.info(f"Selecting HV among non-promoters => picking {n_needed} peaks.")

        # Subset to non-promoters
        non_promoter_mask = ~promoter_mask
        data_atac_nonprom = data_atac[:, non_promoter_mask].copy()

        # HV selection among non-promoters
        data_atac_nonprom_hv = select_highly_variable_peaks_by_std(data_atac_nonprom, n_needed, cluster_key, mode)

        # Final union => promoter + HV(non-promoters)
        final_promoter_set = set(promoter_peaks)
        final_nonprom_set = set(data_atac_nonprom_hv.var_names)
        final_set = data_atac.var_names[data_atac.var_names.isin(final_promoter_set | final_nonprom_set)]

        data_atac_sub = data_atac[:, final_set].copy()
        logger.info(
            f"Final set => {len(promoter_peaks)} promoter + "
            f"{data_atac_nonprom_hv.shape[1]} HV => total {data_atac_sub.shape[1]} peaks."
        )
        return data_atac_sub
