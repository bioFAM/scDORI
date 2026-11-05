import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def download_file(url: str, out_path: Path) -> None:
    """
    Download a file from a given URL and save it locally using the `wget` command.

    Parameters
    ----------
    url : str
        The URL to download from.
    out_path : pathlib.Path
        The local file path where the downloaded file will be saved.

    Returns
    -------
    None
        If the file already exists, no download is performed. Otherwise, runs
        `wget` to fetch the file, and leaves no file behind if that fails.
    """
    if out_path.exists():
        logger.info(f"File already exists: {out_path}. Skipping download.")
        return
    # wget creates its output file even when it fails. Download beside the target and move
    # it only on success, so a failed or interrupted download cannot look like a finished one.
    partial = out_path.with_name(out_path.name + ".part")
    logger.info(f"Downloading {url} -> {out_path}")
    try:
        subprocess.run(["wget", "-O", str(partial), url], check=True)
    except subprocess.CalledProcessError as error:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {url}. Check your connection and run this cell again. If the "
            f"address no longer works, set the matching URL in the preprocessing settings, or "
            f"place the file at {out_path} yourself."
        ) from error
    partial.replace(out_path)


def unzip_gz(file_path: Path, remove_input: bool = False) -> None:
    """
    Decompress a GZIP file (i.e., `.gz`) in-place using `gzip -d`.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the `.gz` file to be decompressed.
    remove_input : bool, optional
        If True, delete the original `.gz` file after successful decompression.
        Default is False.

    Returns
    -------
    None
        Decompresses the file in-place, leaving a new file without the `.gz` extension.
    """
    logger.info(f"Decompressing {file_path}")
    subprocess.run(["gzip", "-d", str(file_path)], check=True)
    if remove_input:
        gz_file = file_path
        if gz_file.exists():
            gz_file.unlink()


def resolve_genome_urls(
    species: str, assembly: str, gtf_url: str, chrom_sizes_url: str, fasta_url: str
) -> tuple[str, str, str]:
    """
    Resolve final URLs for GTF, chromosome sizes, and FASTA files based on species and assembly.

    1. If user provides URLs (gtf_url, chrom_sizes_url, fasta_url), use them directly.
    2. If not, attempt to use known defaults for recognized combos (e.g. mouse/mm10, human/hg38).
    3. If the combo is unrecognized and the user hasn't provided all URLs, raise an error.

    Parameters
    ----------
    species : str
        Species name (e.g. "mouse", "human").
    assembly : str
        Genome assembly (e.g. "mm10", "hg38").
    gtf_url : str or None
        A user-provided URL for the GTF file, or None to try defaults.
    chrom_sizes_url : str or None
        A user-provided URL for the chromosome sizes file, or None to try defaults.
    fasta_url : str or None
        A user-provided URL for the FASTA file, or None to try defaults.

    Returns
    -------
    (final_gtf_url, final_chrom_sizes_url, final_fasta_url) : tuple of str
        The resolved URLs for GTF, chromosome sizes, and FASTA. These may come from
        user-provided values or known defaults if recognized. If no defaults
        exist and user inputs are missing, raises ValueError.
    """
    final_gtf_url = gtf_url
    final_chrom_sizes_url = chrom_sizes_url
    final_fasta_url = fasta_url

    # Known defaults for mouse mm10
    if species.lower() == "mouse" and assembly.lower() == "mm10":
        if final_gtf_url is None:
            final_gtf_url = (
                "https://ftp.ebi.ac.uk/pub/databases/gencode/"
                "Gencode_mouse/release_M18/gencode.vM18.basic.annotation.gtf.gz"
            )
        if final_chrom_sizes_url is None:
            final_chrom_sizes_url = "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.chrom.sizes"
        if final_fasta_url is None:
            final_fasta_url = "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz"

    # Known defaults for human hg38
    elif species.lower() == "human" and assembly.lower() == "hg38":
        if final_gtf_url is None:
            final_gtf_url = (
                "https://ftp.ebi.ac.uk/pub/databases/gencode/"
                "Gencode_human/release_47/gencode.v47.primary_assembly.basic.annotation.gtf.gz"
            )
        if final_chrom_sizes_url is None:
            final_chrom_sizes_url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes"
        if final_fasta_url is None:
            final_fasta_url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"

    else:
        # Unknown assembly => user must provide or raise an error if any is None
        if final_gtf_url is None or final_chrom_sizes_url is None or final_fasta_url is None:
            raise ValueError(
                f"Unknown assembly '{assembly}' for species='{species}'. "
                "Please provide gtf_url, chrom_sizes_url, and fasta_url in config."
            )

    return (final_gtf_url, final_chrom_sizes_url, final_fasta_url)


def download_genome_references(
    genome_dir: Path,
    species: str,
    assembly: str,
    gtf_url: str = None,
    chrom_sizes_url: str = None,
    fasta_url: str = None,
) -> None:
    """
    Download reference genome files (GTF, chrom.sizes, FASTA) for a given species and assembly.

    1. Resolve the URLs from user input or known defaults for recognized combos (mouse/mm10, human/hg38).
    2. Download each file if not already present.
    3. Decompress .gz files for GTF and FASTA.

    Parameters
    ----------
    genome_dir : pathlib.Path
        Directory where the reference files will be downloaded.
    species : str
        Species name (e.g. "mouse", "human").
    assembly : str
        Genome assembly version (e.g. "mm10", "hg38").
    gtf_url : str, optional
        GTF file URL to override defaults. If None, use known default (if available).
    chrom_sizes_url : str, optional
        Chromosome sizes file URL to override defaults. If None, use known default (if available).
    fasta_url : str, optional
        FASTA file URL to override defaults. If None, use known default (if available).

    Returns
    -------
    None
        Files are downloaded into `genome_dir`. If a file already exists, no new download occurs.
    """
    genome_dir.mkdir(parents=True, exist_ok=True)

    # 1) Resolve the final URLs based on species/assembly + user overrides
    final_gtf_url, final_chrom_sizes_url, final_fasta_url = resolve_genome_urls(
        species, assembly, gtf_url, chrom_sizes_url, fasta_url
    )
    logger.info(
        f"Using genome references for species='{species}', assembly='{assembly}'.\n"
        f"GTF: {final_gtf_url}\n"
        f"Chrom.sizes: {final_chrom_sizes_url}\n"
        f"FASTA: {final_fasta_url}"
    )

    # Decide on local filenames
    gtf_gz = genome_dir / "annotation.gtf.gz"
    gtf_final = genome_dir / "annotation.gtf"
    chrom_sizes_path = genome_dir / f"{assembly}.chrom.sizes"
    fasta_gz = genome_dir / f"{assembly}.fa.gz"
    fasta_final = genome_dir / f"{assembly}.fa"

    # 2) GTF
    if not gtf_final.exists():
        download_file(final_gtf_url, gtf_gz)
        unzip_gz(gtf_gz, remove_input=True)

    # 3) chrom sizes
    if not chrom_sizes_path.exists():
        download_file(final_chrom_sizes_url, chrom_sizes_path)

    # 4) FASTA
    if not fasta_final.exists():
        download_file(final_fasta_url, fasta_gz)
        unzip_gz(fasta_gz, remove_input=True)

    logger.info(f"Reference files are ready in {genome_dir}")
