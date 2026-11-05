import logging
from pathlib import Path

import anndata as ad
import scanpy as sc

from scdori._sparse import prepare_count_matrix

logger = logging.getLogger(__name__)


def create_dir_if_not_exists(directory: Path) -> None:
    """
    Create a directory (and parent directories) if it does not already exist.

    Parameters
    ----------
    directory : pathlib.Path
        The directory path to create.

    Returns
    -------
    None
        If the directory already exists, does nothing. Otherwise, it is created
        along with any necessary parent folders.
    """
    if not directory.exists():
        logger.info(f"Creating directory: {directory}")
        directory.mkdir(parents=True, exist_ok=True)


def load_anndata(data_dir: Path, rna_file: str, atac_file: str) -> tuple[ad.AnnData, ad.AnnData]:
    """
    Load RNA and ATAC data from disk into AnnData objects.

    Parameters
    ----------
    data_dir : pathlib.Path
        Base directory containing the `.h5ad` files.
    rna_file : str
        The filename for the RNA data (H5AD).
    atac_file : str
        The filename for the ATAC data (H5AD).

    Returns
    -------
    (AnnData, AnnData)
        A tuple containing:
        - data_rna: AnnData
            The RNA AnnData object.
        - data_atac: AnnData
            The ATAC AnnData object.

    Notes
    -----
    Both `.h5ad` files are expected to be located in `data_dir`.
    This function logs the path from which each file is loaded.
    """
    rna_path = data_dir / rna_file
    atac_path = data_dir / atac_file
    logger.info(f"Loading RNA from {rna_path}, ATAC from {atac_path}")
    data_rna = sc.read_h5ad(rna_path)
    data_atac = sc.read_h5ad(atac_path)
    return data_rna, data_atac


def save_processed_datasets(
    data_rna: ad.AnnData, data_atac: ad.AnnData, out_dir: Path, *, count_matrix_csr: bool = True
) -> None:
    """
    Save processed RNA and ATAC AnnData objects with matching cell order.

    Parameters
    ----------
    data_rna : anndata.AnnData
        The RNA AnnData object, potentially subset or processed.
    data_atac : anndata.AnnData
        The ATAC AnnData object, potentially subset or processed.
    out_dir : pathlib.Path
        The directory where the processed `.h5ad` files will be saved.
    count_matrix_csr : bool, optional
        Save CSC counts as CSR for faster cell-row reads (default True).
        False preserves layout. Only the aligned copies are converted; the
        caller's objects are unchanged. Conversion needs temporary host RAM.

    Returns
    -------
    None
        Writes two files, `rna_processed.h5ad` and `atac_processed.h5ad`,
        ensuring both have the same set and order of cells.

    Notes
    -----
    - This function intersects the cell indices (obs_names) of `data_rna` and `data_atac` to keep only the common cells.
    - The final shapes of the saved AnnData objects are logged.
    """
    # Ensure same cell order
    common_cells = data_rna.obs_names.intersection(data_atac.obs_names)
    data_rna = data_rna[common_cells].copy()
    data_atac = data_atac[common_cells].copy()

    data_rna = prepare_count_matrix(data_rna, use_csr=count_matrix_csr)
    data_atac = prepare_count_matrix(data_atac, use_csr=count_matrix_csr)

    # Save
    rna_path = out_dir / "rna_processed.h5ad"
    atac_path = out_dir / "atac_processed.h5ad"
    data_rna.write_h5ad(rna_path)
    data_atac.write_h5ad(atac_path)
    logger.info(f"Saved processed RNA to {rna_path} with shape={data_rna.shape}")
    logger.info(f"Saved processed ATAC to {atac_path} with shape={data_atac.shape}")
