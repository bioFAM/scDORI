import logging
from dataclasses import dataclass, field

import numpy as np


# Peak-gene window around each gene, in bp, when a config does not set window_size.
DEFAULT_WINDOW_SIZE = {"human": 150_000, "mouse": 80_000}
FALLBACK_WINDOW_SIZE = 80_000


@dataclass
class PreprocessingConfig:
    """
    Global configuration module for the single-cell multiome data pre-processing used in scDORI.

    This module holds constants and parameters used across the pipeline, including:
    1. Logging settings
    2. Paths for data, genome references, and motif databases
    3. Species and genome assembly details
    4. HVG/peak selection parameters
    5. Promoter logic
    6. In-silico ChIP-seq correlation settings
    7. Other hyperparameters or default values

    Variables
    ---------
    random_seed : int
        Random seed for reproducibility.
    data_dir : str
        Directory where RNA and ATAC AnnData files are stored.
    genome_dir : str
        Directory containing genome FASTA and chromosome size files.
    motif_directory : str
        Directory containing .meme motif files for TF databases.
    output_subdir_name : str
        Name of the subdirectory within data_dir to store intermediate outputs.
    rna_adata_file_name : str
        Filename for the RNA AnnData file (H5AD).
    atac_adata_file_name : str
        Filename for the ATAC AnnData file (H5AD).
    species : str
        Species name, e.g., "mouse" or "human".
    genome_assembly : str
        Genome assembly version, e.g., "mm10" or "hg38".
    gtf_url : str or None
        URL to the GTF file override; if None, defaults are used based on species and assembly.
    chrom_sizes_url : str or None
        URL to the chromosome sizes file override; if None, defaults are used if known.
    fasta_url : str or None
        URL to the FASTA genome file override; if None, defaults are used if known.
    chrom_sizes_file : str
        Local filename for the chromosome sizes, e.g., "mm10.chrom.sizes" or "hg38.chrom.sizes".
    mitochondrial_prefix : str
        Prefix used to identify mitochondrial genes in the RNA AnnData.
    genes_user : list of str
        Genes retained during variable-feature selection if present after upstream RNA filters.
    tfs_user : list of str
        TFs retained if present after RNA filters and represented in the motif database.
    peaks_user : list of str
        Exact input ATAC peak names to retain, even outside gene windows or below
        the variability cutoff. No counts or gene links are invented.
    motif_database : str
        Name of the motif database, e.g., "cisbp".
    num_genes : int
        Number of genes to select for scDORI training (via HVG + user overrides).
    num_tfs : int
        Number of TFs to select for scDORI training (via HV among potential TFs + user overrides).
    min_cells_per_gene : int
        Minimum cell count threshold for gene detection (not enforced in current code).
    window_size : int or None
        Genomic window (bp) around each gene for selecting peaks. Example: 80,000 => ±80 kb.
        None uses the default for ``species``: 150 kb for human, 80 kb otherwise.
    num_peaks : int
        Target number of peaks for training scDORI; some are forced (promoters), the rest are HV.
    peak_std_batch_key : str
        Key in `.obs` used to group cells and measure peak standard deviation for HV selection.
    batch_key : str
        Key in `.obs` denoting experimental batch/covariate for integration or Harmony correction.
    leiden_resolution : float
        Resolution parameter for the Leiden clustering used to create metacells.
    keep_promoter_peaks : bool
        Whether to keep promoter peaks unconditionally in the final set of peaks.
    promoter_col : str
        Column in ATAC `.var` indicating if a peak is a promoter peak (True/False).
    motif_match_pvalue_threshold : float
        P-value threshold for motif hits in FIMO for in-silico ChIP-seq.
    correlation_percentile : float
        Percentile cutoff for correlation significance with TF expression (e.g., 99 => p≈0.01).
    n_bg_peaks_for_corr : int
        Number of peaks (lowest motif scores) used per TF as background in correlation tests.
    correlation_normalization : str
        Metacell values correlated for in-silico ChIP-seq: ``"mean_log"`` (default) uses the mean of
        each cell's log-normalized RNA and mean ATAC fragments, ``"none"`` uses mean counts of both,
        ``"log_normalized"`` removes depth differences between metacells first, for both.
    peak_distance_scaling_factor : float
        Decay factor for exponential distance weighting in initial gene-peak links.
    peak_distance_min_cutoff : float
        Minimum allowed scaled distance (exponential) threshold within the user-defined window.
    """

    # Logging
    logging_level: int = logging.INFO

    # Seed
    random_seed: int = 3141

    # Directory structure
    data_dir: str = "data"
    genome_dir: str = "references/genome"
    motif_directory: str = "assets/motif_database"
    output_subdir_name: str = "generated"
    # Save CSC count matrices as CSR for repeated cell-row access during training.
    count_matrix_csr: bool = True

    # Input Filenames
    rna_adata_file_name: str = "anndata.h5ad"
    atac_adata_file_name: str = "PeakMatrix_anndata.h5ad"

    # Species & references
    species: str = "mouse"
    genome_assembly: str = "mm10"

    # Optional user-provided URLs for genome files
    gtf_url: str | None = None
    chrom_sizes_url: str | None = None
    fasta_url: str | None = None

    chrom_sizes_file: str = "mm10.chrom.sizes"

    # Genes & TF selection
    mitochondrial_prefix: str = "mt-"

    genes_user = ["Myt1l"]
    tfs_user = ["Tbx5", "Myt1l", "Cdx2", "Sox2"]
    peaks_user: list[str] = field(default_factory=list)
    motif_database = "cisbp"

    num_genes: int = 4000
    num_tfs: int = 300
    min_cells_per_gene: int = 4

    # Genomic window; None uses the default for this species
    window_size: int | None = None

    num_peaks: int = 90000
    peak_std_batch_key: str = "leiden"

    # Batch key & metacell parameters
    batch_key: str = "sample"
    leiden_resolution: float = 10

    # Promoter logic
    keep_promoter_peaks: bool = True
    promoter_col: str = "promoter_col"

    # Correlation & in-silico ChIP-seq
    motif_match_pvalue_threshold: float = 1e-3
    correlation_percentile: float = 99
    n_bg_peaks_for_corr: int = 5000
    correlation_normalization: str = "mean_log"

    # Distance matrix parameters
    peak_distance_scaling_factor: float = 20000

    def __post_init__(self):  # noqa: D105
        self._window_size_from_species = self.window_size is None
        self.resolve_window_size()

    def resolve_window_size(self, explicit: int | None = None) -> None:
        """Use ``explicit`` if given; otherwise follow ``species`` unless a window was set in this config's defaults."""
        if explicit is not None:
            self._window_size_from_species = False
            self.window_size = explicit
        elif self._window_size_from_species or self.window_size is None:
            self.window_size = DEFAULT_WINDOW_SIZE.get(str(self.species).lower(), FALLBACK_WINDOW_SIZE)

    # Calculate the min cutoff based on window_size and scaling factor
    @property
    def peak_distance_min_cutoff(self) -> float:  # noqa: D102
        return np.e ** (-1 * (self.window_size / self.peak_distance_scaling_factor))


ppConfig = PreprocessingConfig()
