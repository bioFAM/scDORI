import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .data_io import load_scdori_inputs, save_model_weights
from .models import initialize_scdori_parameters, scDoRI
from .train_grn import train_model_grn
from .train_scdori import train_scdori_phases
from .utils import set_seed

logger = logging.getLogger(__name__)


def run_scdori_pipeline():
    """
    Run the scDORI pipeline in three main phases.

    The main phases are:
    1) ATAC+TF warmup (phase 1 warmup),
    2) Add RNA (phase 1 full),
    3) GRN training (phase 2).

    Steps
    -----
    1. Configure logging, set random seed, determine computing device.
    2. Load data: RNA/ATAC AnnData, gene-peak distances, in silico ChIP-seq embeddings.
    3. Split cells into train and eval sets, create DataLoaders.
    4. Build and initialize the scDORI model:
       - The model is configured with the number of genes, peaks, TFs, and topics.
       - Initialize parameters (gene-peak, in silico matrices, etc.).
    5. Train phases 1 & 2 (integrated ATAC + TF, then add RNA).
    6. Save model weights.
    7. Re-initialize GRN-related parameters and run phase 3 (GRN training).
    8. Save final model weights for the GRN phase.

    Returns
    -------
    None
        The pipeline executes end-to-end training of the scDORI model,
        saving intermediate and final weights to disk as specified in `config`.

    Notes
    -----
    - This function relies on configuration settings in `config.py`.
    - The pipeline uses `train_scdori_phases` for phases 1 & 2,
      and `train_model_grn` for the GRN phase.
    - Outputs (model weights) are saved to the paths specified by `config.weights_folder_scdori`
      and `config.weights_folder_grn`.
    """
    logging.basicConfig(level=config.logging_level)
    logger.info("Starting scDORI pipeline with integrated GRN.")
    set_seed(config.random_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1) Load data
    rna_metacell, atac_metacell, gene_peak_dist, insilico_act, insilico_rep = load_scdori_inputs()
    gene_peak_fixed = gene_peak_dist.copy()
    gene_peak_fixed[gene_peak_fixed > 0] = 1

    # 2) Make small train/test sets
    n_cells = rna_metacell.n_obs
    indices = np.arange(n_cells)
    train_idx, eval_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.from_numpy(train_idx))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size_cell, shuffle=True)

    eval_dataset = TensorDataset(torch.from_numpy(eval_idx))
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size_cell, shuffle=False)

    # 3) Build integrated model
    num_genes = rna_metacell.n_vars
    num_peaks = atac_metacell.n_vars
    # Suppose we deduce num_tfs from somewhere
    num_tfs = 100  # example
    model = scDoRI(
        device=device,
        num_genes=num_genes,
        num_peaks=num_peaks,
        num_tfs=num_tfs,
        num_topics=config.num_topics,
        num_batches=config.num_batches,
        dim_encoder1=config.dim_encoder1,
        dim_encoder2=config.dim_encoder2,
    ).to(device)

    initialize_scdori_parameters(
        model, gene_peak_dist, gene_peak_fixed, insilico_act=insilico_act, insilico_rep=insilico_rep, phase="warmup"
    )

    # 4) Train Phase 1 + 2
    model = train_scdori_phases(model, device, train_loader, eval_loader)
    save_model_weights(model, Path(config.weights_folder_scdori), "scdori_final")

    # 5) Phase 3 => GRN
    initialize_scdori_parameters(
        model, gene_peak_dist, gene_peak_fixed, insilico_act=insilico_act, insilico_rep=insilico_rep, phase="grn"
    )
    model = train_model_grn(model, device, train_loader, eval_loader)
    save_model_weights(model, Path(config.weights_folder_grn), "grn_final")

    logger.info("All phases complete. scDORI pipeline done.")
