import logging
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc
from gtfparse import read_gtf

logger = logging.getLogger(__name__)


def load_gtf(gtf_path: Path) -> pd.DataFrame:
    """
    Load gene coordinates from a GTF file into a pandas DataFrame using gtfparse.

    Parameters
    ----------
    gtf_path : pathlib.Path
        The path to the GTF file (optionally gzipped).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing parsed GTF records. The columns correspond to
        GTF fields such as "gene_name", "gene_type", "start", "end", etc.
    """
    logger.info(f"Loading GTF from {gtf_path}")
    df = read_gtf(gtf_path)
    gene_coordinates = df if isinstance(df, pd.DataFrame) else df.to_pandas()
    return gene_coordinates


def filter_protein_coding_genes(data_rna: ad.AnnData, gtf_df: pd.DataFrame) -> ad.AnnData:
    """
    Retain only protein-coding genes in the RNA AnnData object based on GTF annotations.

    Parameters
    ----------
    data_rna : anndata.AnnData
        RNA single-cell data.
    gtf_df : pd.DataFrame
        A GTF DataFrame (from `load_gtf`) containing columns like "gene_type", "gene_name".

    Returns
    -------
    anndata.AnnData
        The AnnData subset containing only protein-coding genes found in both
        the original data and the GTF's "gene_type == 'protein_coding'".
    """
    df_protein_coding = gtf_df[gtf_df["gene_type"] == "protein_coding"]
    pc_genes = set(df_protein_coding["gene_name"].unique())
    rna_genes = set(data_rna.var_names)
    keep_genes = sorted(pc_genes & rna_genes)
    data_rna_sub = data_rna[:, keep_genes].copy()
    logger.info(f"Filtered to protein-coding genes: {data_rna_sub.shape[1]} genes left.")
    return data_rna_sub


def compute_hvgs_and_tfs(
    data_rna: ad.AnnData,
    tf_names: list[str],
    user_genes: list[str] = None,
    user_tfs: list[str] = None,
    num_genes: int = 3000,
    num_tfs: int = 300,
    min_cells: int = 20,
) -> tuple[ad.AnnData, list[str], list[str]]:
    """
    Compute sets of Highly Variable Genes (HVGs) and TFs (transcription factors) for scDORI training.

    This function:
      1. Identifies user-specified genes and TFs present in `data_rna`.
      2. Selects additional TFs by computing HVGs among potential TFs up to `num_tfs`.
      3. Selects non-TF HVGs up to `num_genes` (minus any user-specified genes and TFs).
      4. Combines these sets into a final AnnData subset and returns them.

    Parameters
    ----------
    data_rna : anndata.AnnData
        The RNA single-cell data from which to select HVGs and TFs.
    tf_names : list of str
        A list of all TF names (from a motif database or known TF list).
    user_genes : list of str, optional
        A list of user-specified genes that must be included in the final set,
        default is None.
    user_tfs : list of str, optional
        A list of user-specified TFs that must be included in the final set,
        default is None.
    num_genes : int, optional
        Target total RNA features including TFs. Requested features take priority.
    num_tfs : int, optional
        Total number of TFs desired. Default is 300.
    min_cells : int, optional
        Minimum number of cells in which a gene must be detected (e.g., nonzero)
        to be considered for HVG selection. Default is 20 (not fully enforced in
        this code snippet, but typically used with standard HVG filtering).

    Returns
    -------
    data_rna_processed : anndata.AnnData
        A subset of the original data containing the selected HVGs and TFs.
    final_genes : list of str
        The final list of HVGs (non-TFs).
    final_tfs : list of str
        The final list of TFs.

    Notes
    -----
    - HVG selection is done by `scanpy.pp.highly_variable_genes`, using normalized/log1p data.
    - User-provided genes and TFs are included by default, removing them from the
      HVG candidate pool if they were already selected.
    - TFs are not re-labeled or otherwise changed beyond this classification.
    - The column `data_rna_processed.var["gene_type"]` is set to "HVG" or "TF" for each gene.
    """
    if user_genes is None:
        user_genes = []
    if user_tfs is None:
        user_tfs = []

    logger.info("Selecting HVGs and TFs...")

    # 1) Validate user-specified lists
    valid_genes_user = list(set(data_rna.var_names).intersection(user_genes))
    valid_tfs_user = list(set(data_rna.var_names).intersection(user_tfs).intersection(tf_names))
    missing_genes = sorted(set(user_genes) - set(data_rna.var_names))
    missing_tfs = sorted(set(user_tfs) - set(data_rna.var_names))
    missing_motifs = sorted((set(user_tfs) & set(data_rna.var_names)) - set(tf_names))
    if missing_genes:
        logger.warning("Requested genes absent after RNA filtering (not added): %s", missing_genes)
    if missing_tfs:
        logger.warning("Requested TFs absent after RNA filtering (not added): %s", missing_tfs)
    if missing_motifs:
        logger.warning("Requested TFs without matching motif names (not added as TFs): %s", missing_motifs)

    num_tfs_hvg = max(0, num_tfs - len(valid_tfs_user))
    num_genes_hvg = max(0, num_genes - len(valid_genes_user) - num_tfs)

    # 2) HVGs among TFs
    tf_candidates = sorted((set(tf_names) - set(valid_tfs_user)) & set(data_rna.var_names))
    data_rna_tf = data_rna[:, tf_candidates].copy()
    if num_tfs_hvg and len(tf_candidates):
        sc.pp.normalize_total(data_rna_tf)
        sc.pp.log1p(data_rna_tf)
        sc.pp.highly_variable_genes(data_rna_tf, n_top_genes=min(num_tfs_hvg, len(tf_candidates)), subset=True)
    else:
        data_rna_tf = data_rna_tf[:, :0].copy()

    selected_tfs = sorted(list(data_rna_tf.var_names) + valid_tfs_user)

    # 3) HVGs among non-TFs
    non_tf_candidates = set(data_rna.var_names) - set(selected_tfs) - set(valid_genes_user)
    data_rna_non_tf = data_rna[:, sorted(non_tf_candidates)].copy()
    if num_genes_hvg and len(non_tf_candidates):
        sc.pp.normalize_total(data_rna_non_tf)
        sc.pp.log1p(data_rna_non_tf)
        sc.pp.highly_variable_genes(
            data_rna_non_tf, n_top_genes=min(num_genes_hvg, len(non_tf_candidates)), subset=True
        )
    else:
        data_rna_non_tf = data_rna_non_tf[:, :0].copy()

    selected_non_tfs = sorted(set(data_rna_non_tf.var_names).union(valid_genes_user))
    selected_non_tfs = [g for g in selected_non_tfs if g not in selected_tfs]

    final_genes = selected_non_tfs
    final_tfs = selected_tfs

    combined = final_genes + final_tfs
    data_rna_processed = data_rna[:, combined].copy()

    # Mark gene_type in .var
    gene_types = ["HVG"] * len(final_genes) + ["TF"] * len(final_tfs)
    data_rna_processed.var["gene_type"] = gene_types
    data_rna_processed.uns["scdori_requested_features"] = {
        "genes_retained": sorted(valid_genes_user), "tfs_retained": sorted(valid_tfs_user),
        "genes_missing_after_filtering": missing_genes, "tfs_missing_after_filtering": missing_tfs,
        "tfs_without_motifs": missing_motifs,
    }

    logger.info(f"Selected {len(final_genes)} HVGs + {len(final_tfs)} TFs.")
    return data_rna_processed, final_genes, final_tfs
