import logging

import anndata as ad

logger = logging.getLogger(__name__)


def intersect_cells(data_rna: ad.AnnData, data_atac: ad.AnnData) -> tuple[ad.AnnData, ad.AnnData]:
    """
    Keep only the cells that are present in both RNA and ATAC datasets.

    Parameters
    ----------
    data_rna : anndata.AnnData
        The RNA single-cell data.
    data_atac : anndata.AnnData
        The ATAC single-cell data.

    Returns
    -------
    (data_rna_sub, data_atac_sub) : tuple of anndata.AnnData
        Two AnnData objects that share the same set of cells, in the same order.

    Notes
    -----
    - The function finds the intersection of `obs_names` (cell barcodes) between
      the two AnnData objects.
    - It logs the new shapes of the intersected RNA and ATAC data.
    """
    common_idx = data_rna.obs_names.intersection(data_atac.obs_names)
    data_rna_sub = data_rna[common_idx].copy()
    data_atac_sub = data_atac[common_idx].copy()
    logger.info(f"Intersected cells: now RNA={data_rna_sub.shape}, ATAC={data_atac_sub.shape}")
    return data_rna_sub, data_atac_sub


def remove_mitochondrial_genes(data_rna: ad.AnnData, mito_prefix: str = "mt-") -> ad.AnnData:
    """
    Remove mitochondrial genes from the RNA data based on a gene name prefix.

    Parameters
    ----------
    data_rna : anndata.AnnData
        The RNA AnnData containing gene expression counts.
    mito_prefix : str, optional
        The prefix used to identify mitochondrial genes. Default is "mt-".

    Returns
    -------
    anndata.AnnData
        A new AnnData object without mitochondrial genes. The original data
        is not modified in-place.

    Notes
    -----
    - Mitochondrial genes are identified by checking if gene names start with
      `mito_prefix` (case-insensitive).
    - A boolean column "mt" is added to `data_rna.var` to indicate whether
      each gene was marked as mitochondrial. This subset is then removed
      from the data.
    - Logs the number of removed genes.
    """
    prefix = mito_prefix.lower()
    data_rna.var["mt"] = [gene.lower().startswith(prefix) for gene in data_rna.var_names]
    keep_mask = ~data_rna.var["mt"]
    data_rna_sub = data_rna[:, keep_mask].copy()
    dropped = data_rna.shape[1] - data_rna_sub.shape[1]
    logger.info(f"Removed {dropped} mitochondrial genes with prefix={mito_prefix}")
    return data_rna_sub
