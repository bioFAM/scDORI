import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def create_extended_gene_bed(
    gtf_df: pd.DataFrame, final_genes: list[str], window_size: int, chrom_sizes_path: Path = None
) -> pd.DataFrame:
    """
    Create an extended gene BED-like DataFrame by adding a window around each gene.

    For each gene in `final_genes`, take the gene feature from `gtf_df` and extend
    the start and end coordinates by `window_size`. If a chromosome sizes file is
    provided, clamp the extended coordinates within valid chromosome boundaries.

    Parameters
    ----------
    gtf_df : pd.DataFrame
        A DataFrame containing GTF entries (e.g., loaded by gtfparse). Expected columns:
        ["feature", "gene_name", "seqname", "start", "end", "strand", ...].
    final_genes : list of str
        The list of gene names to be extended.
    window_size : int
        The number of base pairs to extend on each side of the gene.
    chrom_sizes_path : pathlib.Path, optional
        A path to a tab-delimited file of chromosome sizes (e.g., from UCSC),
        with columns [chrom, size]. If provided, coordinates are clamped to [0, chrom_size].

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ["chr", "start_new", "end_new", "gene_name", "score", "strand"]
        representing the extended regions for each gene.

    Notes
    -----
    - Only rows with `feature == "gene"` are considered in the GTF.
    - If a gene is not found in `gtf_df`, it is omitted.
    - The `chr` column is set from the GTF's "seqname" field.
    """
    # Filter for final genes & gene features
    gtf_gene = gtf_df[gtf_df["feature"] == "gene"].drop_duplicates("gene_name")
    gtf_gene = gtf_gene.set_index("gene_name")
    gtf_gene = gtf_gene.loc[sorted(set(final_genes) & set(gtf_gene.index))]

    # rename for clarity
    gtf_gene["chr"] = gtf_gene["seqname"]

    # If we want clamping, load chrom.sizes
    chrom_dict = {}
    if chrom_sizes_path is not None and chrom_sizes_path.exists():
        chrom_sizes = pd.read_csv(chrom_sizes_path, sep="\t", header=None, names=["chrom", "size"])
        chrom_dict = dict(zip(chrom_sizes["chrom"], chrom_sizes["size"], strict=False))

    extended_rows = []
    for gene in gtf_gene.index:
        row = gtf_gene.loc[gene]
        chr_ = row["chr"]
        start_ = row["start"]
        end_ = row["end"]
        strand_ = row["strand"] if "strand" in row else "+"

        # GTF starts are 1-based inclusive; BED starts are 0-based.
        start_extended = max(0, start_ - 1 - window_size)
        end_extended = end_ + window_size

        if chr_ in chrom_dict:
            chr_len = chrom_dict[chr_]
            if end_extended > chr_len:
                end_extended = chr_len
            # start_extended already clamped to 0 above

        extended_rows.append([chr_, start_extended, end_extended, gene, 0, strand_])

    df_extended = pd.DataFrame(extended_rows, columns=["chr", "start_new", "end_new", "gene_name", "score", "strand"])
    return df_extended


def compute_gene_peak_distance_matrix(
    data_atac: pd.DataFrame, data_rna: pd.DataFrame, gene_coordinates_intersect: pd.DataFrame
) -> np.ndarray:
    """
    Compute a distance matrix between peaks (ATAC) and genes (RNA), accounting for strand.

    This function:
    -----
    1. Expects the ATAC data (`data_atac.var`) to have columns "chr", "start", "end", and "peak_name".
    2. Expects the gene coordinates DataFrame (`gene_coordinates_intersect`) to have columns
        ["chr_gene", "start", "end", "strand", "gene"].
    3. For each gene, we compute a distance to each peak by:
        - 0 if the peak midpoint is within gene-body or promoter (5kb upstream of TSS by default),
        e.g., if on the "+" strand and midpoint is between (start - 5000) and end => distance=0.
        If on the "-" strand and midpoint is between start and (end + 5000) => distance=0.
        - Otherwise, distance = min(abs(mid - start), abs(mid - end)).
        - If the chromosome of the gene != chromosome of the peak => distance = -1.

    Parameters
    ----------
    data_atac : pd.DataFrame
        The ATAC dataset. Its `.var` should contain "chr", "start", "end", "peak_name".
    data_rna : pd.DataFrame
        The RNA dataset. Its `.var` is expected to have gene names as indices or relevant columns.
    gene_coordinates_intersect : pd.DataFrame
        A DataFrame with gene info, columns at least:
        ["chr_gene", "start", "end", "strand", "gene"].

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (n_genes, n_peaks) representing gene-peak distances.
        Each row corresponds to a gene (matching `data_rna.var.index`),
        and each column corresponds to a peak (matching `data_atac.var.index` order).

    """
    logger.info("Starting computation of gene-peak distances...")
    logger.info(f"Number of genes: {len(data_rna.var.index)}, Number of peaks: {len(data_atac.var.index)}")
    logger.debug(f"ATAC var columns: {data_atac.var.columns}")
    assert "chr" in data_atac.var.columns, "'chr' column is missing in data_atac.var!"
    assert "start" in data_atac.var.columns, "'start' column is missing in data_atac.var!"
    assert "end" in data_atac.var.columns, "'end' column is missing in data_atac.var!"

    logger.debug(f"Gene coordinates columns: {gene_coordinates_intersect.columns}")
    assert "chr_gene" in gene_coordinates_intersect.columns, "'chr_gene' column is missing!"
    assert "start" in gene_coordinates_intersect.columns, "'start' column is missing!"
    assert "end" in gene_coordinates_intersect.columns, "'end' column is missing!"
    assert "strand" in gene_coordinates_intersect.columns, "'strand' column is missing!"

    peak_gene_distance = []

    # Add midpoints to peaks in the ATAC data
    data_atac.var["mid"] = (data_atac.var["start"] + data_atac.var["end"]) // 2
    logger.debug("Added midpoint column to ATAC peaks.")

    # Iterate over all genes
    for gene in tqdm(data_rna.var.index, total=len(data_rna.var.index)):
        strand_gene = gene_coordinates_intersect.loc[gene, "strand"]

        # Build pairs of all peaks with this gene
        atac_var_all = pd.DataFrame(
            itertools.product(data_atac.var["peak_name"].values, [gene]), columns=["peak_name", "gene"]
        )

        # Merge to get peak coordinates
        atac_var_all = atac_var_all.merge(data_atac.var[["peak_name", "chr", "mid"]], on="peak_name", how="inner")
        atac_var_all = atac_var_all.merge(
            gene_coordinates_intersect[["gene", "chr_gene", "start", "end"]], on="gene", how="inner"
        )

        mid = atac_var_all["mid"].values
        start_arr = atac_var_all["start"].values
        end_arr = atac_var_all["end"].values

        if strand_gene == "+":
            # Zero distance if peak midpoint is between [start_gene - 5000, end_gene]
            within_mask = (mid >= (start_arr - 5000)) & (mid <= end_arr)
        else:
            # Zero distance if peak midpoint is between [start_gene, end_gene + 5000]
            within_mask = (mid >= start_arr) & (mid <= (end_arr + 5000))

        dist_to_start = np.abs(mid - start_arr)
        dist_to_end = np.abs(mid - end_arr)
        min_dist = np.minimum(dist_to_start, dist_to_end)
        distance = np.where(within_mask, 0, min_dist)

        # If chromosome doesn't match
        distance = np.where(atac_var_all["chr"].values == atac_var_all["chr_gene"].values, distance, -1)

        peak_gene_distance.append(distance)

        # Optional debug checkpoint
        if len(peak_gene_distance) % 100 == 0:
            logger.debug(f"Processed {len(peak_gene_distance)} genes so far.")

    # Convert to NumPy array => shape (n_genes, n_peaks)
    peak_gene_distance_matrix = np.array(peak_gene_distance)
    logger.info(f"Gene-peak distance matrix computed with shape: {peak_gene_distance_matrix.shape}")
    return peak_gene_distance_matrix
