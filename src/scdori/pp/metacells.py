import logging

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from scdori._counts import ATAC_COUNT_MODE_KEY, as_fragment_counts, resolve_atac_count_mode

logger = logging.getLogger(__name__)


def create_metacells(
    data_rna: ad.AnnData,
    data_atac: ad.AnnData,
    grouping_key: str = "leiden",
    resolution: float = 5.0,
    batch_key: str = "sample",
    atac_count_mode: str | None = None,
) -> tuple[ad.AnnData, ad.AnnData]:
    """
    Create metacell-level RNA and ATAC AnnData objects by clustering cells and computing mean values per cluster.

    This function:
    1. Normalizes and logs the RNA data, then runs PCA.
    2. Uses Harmony integration for batch correction on the PCA embeddings.
    3. Clusters the RNA data with Leiden at the specified resolution, storing the
        cluster labels in ``data_rna.obs[grouping_key]``.
    4. Summarizes RNA expression and ATAC accessibility for each cluster by taking
        the mean of each feature across all cells in that cluster.

    Parameters
    ----------
    data_rna : anndata.AnnData
        Raw RNA counts. Clustering uses a normalized copy; X is never modified.
        The shape is (n_cells, n_genes).
    data_atac : anndata.AnnData
        ATAC single-cell data with the same set or superset of cell IDs in
        `data_rna.obs_names`. The shape is (n_cells, n_peaks).
    grouping_key : str, optional
        The key in `data_rna.obs` where the Leiden cluster labels will be stored.
        Default is "leiden".
    resolution : float, optional
        The resolution parameter for Leiden clustering. Higher values yield more clusters.
        Default is 5.0.
    batch_key : str, optional
        The column in `data_rna.obs` indicating batch information for Harmony integration.
        Default is "sample".

    atac_count_mode : {"insertions", "fragments"}, optional
        Input count convention. Uses saved metadata when available, otherwise
        defaults to insertions. Output ATAC means are tagged as fragments.

    Returns
    -------
    (rna_metacell, atac_metacell) : tuple of anndata.AnnData
        - rna_metacell : shape (#clusters, n_genes)
        - atac_metacell : shape (#clusters, n_peaks)
        The `.obs` index is set to the cluster labels, and the `.var` is inherited from the original `data_rna`/`data_atac`.

    Notes
    -----
    - The function uses `scanpy.external.pp.harmony_integrate` for batch integration on the PCA representation stored in "X_pca_harmony".
    - Only insertion inputs are converted using `(atac_vals + 1) // 2`.
    - Mean values are computed across cells in each cluster for both RNA and ATAC.
    - Input RNA counts remain unchanged, including if clustering fails.
    - Missing ATAC cells are rejected rather than treated as zero measurements.
    """
    import scanpy.external as sce

    mode = resolve_atac_count_mode(data_atac, atac_count_mode)
    if not data_rna.obs_names.is_unique or not data_atac.obs_names.is_unique:
        raise ValueError("RNA and ATAC cell IDs must be unique")
    if not data_rna.obs_names.isin(data_atac.obs_names).all():
        raise ValueError("ATAC data must include every RNA cell before metacell construction")
    logger.info(f"Creating metacells with resolution={resolution} (grouping key={grouping_key}).")
    working = data_rna.copy()
    sc.pp.normalize_total(working)
    sc.pp.log1p(working)
    sc.pp.pca(working)
    sce.pp.harmony_integrate(working, batch_key)
    sc.pp.neighbors(working, use_rep="X_pca_harmony")
    sc.tl.leiden(working, resolution=resolution, key_added=grouping_key)
    data_rna.obs[grouping_key] = working.obs[grouping_key].copy()
    del working
    return aggregate_metacells(data_rna, data_atac, grouping_key, mode)


def aggregate_metacells(
    data_rna: ad.AnnData, data_atac: ad.AnnData, grouping_key: str, atac_count_mode: str | None = None
) -> tuple[ad.AnnData, ad.AnnData]:
    """Average RNA and ATAC per existing cluster label, without clustering again.

    ``create_metacells`` clusters and then calls this; call it directly to rebuild metacells
    from the labels saved in ``rna_processed.h5ad``. ``X`` holds means of raw RNA counts and of
    ATAC fragment counts. The RNA layer ``mean_log`` holds the mean of each cell's log-normalized
    counts (scaled to the median library, then log1p), which TF binding correlates by default.
    """
    mode = resolve_atac_count_mode(data_atac, atac_count_mode)
    normalized = data_rna.copy()
    sc.pp.normalize_total(normalized)
    sc.pp.log1p(normalized)
    clusters = data_rna.obs[grouping_key].unique()
    cluster_groups = data_rna.obs.groupby(grouping_key, observed=True)

    mean_rna_list = []
    mean_log_list = []
    mean_atac_list = []
    cluster_names = []

    for cluster_name in clusters:
        cell_idx = cluster_groups.get_group(cluster_name).index

        # Sparse means avoid materializing every cell-by-feature cluster.
        rna_vals = data_rna[cell_idx].X
        mean_rna_list.append(np.asarray(rna_vals.mean(axis=0)).ravel())
        mean_log_list.append(np.asarray(normalized[cell_idx].X.mean(axis=0)).ravel())
        atac_vals = as_fragment_counts(data_atac[cell_idx].X, mode)
        mean_atac_list.append(np.asarray(atac_vals.mean(axis=0)).ravel())

        cluster_names.append(cluster_name)

    # Build new AnnData for metacells
    mean_rna_arr = np.vstack(mean_rna_list)
    mean_atac_arr = np.vstack(mean_atac_list)

    obs_df = pd.DataFrame({grouping_key: cluster_names}).set_index(grouping_key)
    obs_df["num_cells"] = [len(cluster_groups.get_group(name)) for name in cluster_names]

    rna_metacell = ad.AnnData(X=mean_rna_arr, obs=obs_df, var=data_rna.var, layers={"mean_log": np.vstack(mean_log_list)})
    atac_metacell = ad.AnnData(X=mean_atac_arr, obs=obs_df, var=data_atac.var)

    atac_metacell.uns[ATAC_COUNT_MODE_KEY] = "fragments"

    logger.info(f"Metacell shapes: RNA={rna_metacell.shape}, ATAC={atac_metacell.shape}")
    return rna_metacell, atac_metacell
