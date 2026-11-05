import json
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from scdori._sparse import prepare_count_matrix

logger = logging.getLogger(__name__)


def load_scdori_inputs(config_file):
    """
    Load RNA & ATAC data (.h5ad files), plus gene-peak distance and in silico chip-seq matrix.

    Parameters
    ----------
    config_file : object
        A configuration file containing the attributes:

        - ``data_dir`` (pathlib.Path): the base directory for input data.
        - ``output_subdir`` (str): the subdirectory where output files are located.
        - ``rna_metacell_file`` (str): the filename for the RNA data (single cell or metacell) (H5AD).
        - ``atac_metacell_file`` (str): the filename for the ATAC data (single cell or metacell) (H5AD).
        - ``gene_peak_distance_file`` (str): the filename for the NumPy array with gene-peak distance matrix.
        - ``insilico_chipseq_act_file`` (str): the filename for the in silico ChIP-seq activator matrix.
        - ``insilico_chipseq_rep_file`` (str): the filename for the in silico ChIP-seq repressor matrix.
        - ``count_matrix_csr`` (bool, optional): convert in-memory CSC counts to CSR for faster cell
          minibatches (default True). False preserves the on-disk sparse layout. Conversion temporarily
          needs an additional sparse representation in host RAM.

    Returns
    -------
    tuple
        A tuple containing:
        rna_metacell : anndata.AnnData
            RNA data loaded from H5AD.
        atac_metacell : anndata.AnnData
            ATAC data loaded from H5AD.
        gene_peak_dist : torch.Tensor
            A tensor of shape (num_genes, num_peaks) representing gene-peak distances.
        insilico_act : torch.Tensor
            A tensor of shape (num_peaks, num_motifs) for in silico ChIP-seq (activator) embeddings.
        insilico_rep : torch.Tensor
            A tensor of shape (num_peaks, num_motifs) for in silico ChIP-seq (repressor) embeddings.
    """
    out_dir = Path(config_file.data_dir) / config_file.output_subdir

    rna_path = out_dir / config_file.rna_metacell_file
    atac_path = out_dir / config_file.atac_metacell_file
    dist_path = out_dir / config_file.gene_peak_distance_file
    act_path = out_dir / config_file.insilico_chipseq_act_file
    rep_path = out_dir / config_file.insilico_chipseq_rep_file

    logger.info(f"Loading RNA from {rna_path}")
    use_csr = getattr(config_file, "count_matrix_csr", True)
    rna_metacell = prepare_count_matrix(ad.read_h5ad(rna_path), use_csr=use_csr)

    logger.info(f"Loading ATAC from {atac_path}")
    atac_metacell = prepare_count_matrix(ad.read_h5ad(atac_path), use_csr=use_csr)

    logger.info(f"Loading gene-peak dist from {dist_path}")
    gene_peak_dist = torch.from_numpy(np.load(dist_path))

    logger.info(f"Loading insilico embeddings from {act_path} & {rep_path}")
    insilico_act = torch.from_numpy(np.load(act_path))
    insilico_rep = torch.from_numpy(np.load(rep_path))

    if not rna_metacell.obs_names.equals(atac_metacell.obs_names) or not rna_metacell.obs_names.is_unique:
        raise ValueError("RNA and ATAC must have identical, unique cell IDs in the same order")
    if gene_peak_dist.shape != (rna_metacell.n_vars, atac_metacell.n_vars):
        raise ValueError("Gene-peak dimensions disagree with RNA/ATAC features")
    if insilico_act.shape != insilico_rep.shape or insilico_act.shape[0] != atac_metacell.n_vars:
        raise ValueError("TF-binding dimensions disagree with ATAC features")
    # Custom callers may supply TF indices themselves. Validate notebook metadata
    # when present without making it mandatory for the general input loader.
    if "gene_type" in rna_metacell.var:
        tf_names = rna_metacell.var_names[rna_metacell.var["gene_type"] == "TF"]
        if len(tf_names) != insilico_act.shape[1]:
            raise ValueError("RNA TF count disagrees with binding columns")
        tf_order_path = out_dir / "tf_order.csv"
        if tf_order_path.exists():
            import pandas as pd

            saved_tfs = pd.read_csv(tf_order_path)["tf"].astype(str).tolist()
            if saved_tfs != list(tf_names):
                raise ValueError("RNA TF order disagrees with saved binding-column order")

    return rna_metacell, atac_metacell, gene_peak_dist, insilico_act, insilico_rep


def save_model_weights(model, path: Path, tag: str):
    """
    Save model weights to a specified path with a given tag.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model whose state_dict is to be saved.
    path : pathlib.Path
        The directory path where the weights file will be saved.
    tag : str
        An identifier to include in the saved filename (e.g., "best_eval").

    Returns
    -------
    None
    """
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"best_{tag}.pth"
    torch.save(model.state_dict(), file_path)
    # Keep selection with each checkpoint, not just the mutable run-directory record.
    record = getattr(model, "_grn_selection_record", None)
    if record is not None:
        file_path.with_suffix(".topics.json").write_text(json.dumps(record, indent=2) + "\n")
    else:
        file_path.with_suffix(".topics.json").unlink(missing_ok=True)
    logger.info(f"Saved model weights => {file_path}")
