import logging
from pathlib import Path

import torch
from tqdm import tqdm

from .data_io import save_model_weights
from .dataloader import create_minibatch
from .topic_penalties import topic_usage_penalties, validate_topic_penalties
from .utils import log_nb_positive

logger = logging.getLogger(__name__)


def get_phase_scdori(epoch, config_file):
    """
    Determine which training phase to use at a given epoch. In warmup_1, only module 1 and 3 (ATAC and TF reconstruction are trained), after which RNA construction from ATAC is added in warmup_2

    Parameters
    ----------
    epoch : int
        The current training epoch.
    config_file : object
        Configuration object that includes `epoch_warmup_1` to define the cutoff
        for switching from phase "warmup_1" to "warmup_2".

    Returns
    -------
    str
        The phase: "warmup_1" if `epoch < config_file.epoch_warmup_1`, else "warmup_2".
    """
    if epoch < config_file.epoch_warmup_1:
        return "warmup_1"
    else:
        return "warmup_2"


def get_loss_weights_scdori(phase, config_file):
    """
    Get the loss weight dictionary for the specified phase.

    Parameters
    ----------
    phase : str
        The phase of training, one of {"warmup_1", "warmup_2"}.
    config_file : object
        Configuration object containing attributes like `weight_atac_phase1`,
        `weight_tf_phase1`, `weight_rna_phase1`, etc.

    Returns
    -------
    dict
        A dictionary with keys {"atac", "tf", "rna"} indicating the respective loss weights.
    """
    if phase == "warmup_1":
        return {
            "atac": config_file.weight_atac_phase1,
            "tf": config_file.weight_tf_phase1,
            "rna": config_file.weight_rna_phase1,
        }
    else:
        # warmup_2
        return {
            "atac": config_file.weight_atac_phase2,
            "tf": config_file.weight_tf_phase2,
            "rna": config_file.weight_rna_phase2,
        }


def compute_eval_loss_scdori(
    model, device, eval_loader, rna_anndata, atac_anndata, num_cells, tf_indices, encoding_batch_onehot, config_file
):
    """
    Compute the validation loss for scDORI.

    Parameters
    ----------
    model : torch.nn.Module
        The scDORI model to evaluate.
    device : torch.device
        The device (CPU or CUDA) used for PyTorch operations.
    eval_loader : torch.utils.data.DataLoader
        A DataLoader providing validation cell indices.
    rna_anndata : anndata.AnnData
        RNA single-cell data in AnnData format.
    atac_anndata : anndata.AnnData
        ATAC single-cell data in AnnData format.
    num_cells : np.ndarray
        Number of cells per row (if metacells) or ones for single-cell data.
    tf_indices : list or np.ndarray
        Indices of transcription factor genes in the RNA data.
    encoding_batch_onehot : np.ndarray
        One-hot encoding for batch information (cell x num_batches).
    config_file : object
        Configuration object with hyperparameters (loss weights, penalties, etc.).

    Returns
    -------
    tuple
        (eval_loss, eval_loss_atac, eval_loss_tf, eval_loss_rna), each a float.
    """
    model.eval()
    running_loss = 0.0
    running_loss_atac = 0.0
    running_loss_tf = 0.0
    running_loss_rna = 0.0
    nbatch = 0

    with torch.no_grad():
        for batch_data in eval_loader:
            cell_indices = batch_data[0].to(device)
            B = cell_indices.shape[0]

            (input_matrix, tf_exp, library_size_value, num_cells_value, input_batch) = create_minibatch(
                device, cell_indices, rna_anndata, atac_anndata, num_cells, tf_indices, encoding_batch_onehot
            )

            rna_input = input_matrix[:, : model.num_genes]
            atac_input = input_matrix[:, model.num_genes :]
            tf_input = tf_exp

            log_lib_rna = library_size_value[:, 0].reshape(-1, 1)
            log_lib_atac = library_size_value[:, 1].reshape(-1, 1)
            batch_onehot = input_batch

            # Evaluate in "warmup_2" style
            out = model(
                rna_input,
                atac_input,
                tf_input,
                tf_input,
                log_lib_rna,
                log_lib_atac,
                num_cells_value,
                batch_onehot,
                phase="warmup_2",
            )
            preds_atac = out["preds_atac"]
            mu_nb_tf = out["mu_nb_tf"]
            mu_nb_rna = out["mu_nb_rna"]

            # ATAC => Poisson
            library_factor_peak = torch.exp(log_lib_atac.view(B, 1))
            preds_poisson = preds_atac * library_factor_peak
            criterion_poisson = torch.nn.PoissonNLLLoss(log_input=False, reduction="sum")
            loss_atac = criterion_poisson(preds_poisson, atac_input)

            # TF => NB
            alpha_tf = torch.nn.functional.softplus(model.tf_alpha_nb)
            nb_tf_ll = log_nb_positive(tf_input, mu_nb_tf, alpha_tf).sum(dim=1).mean()
            loss_tf = -nb_tf_ll

            # RNA => NB
            alpha_rna = torch.nn.functional.softplus(model.rna_alpha_nb)
            nb_rna_ll = log_nb_positive(rna_input, mu_nb_rna, alpha_rna).sum(dim=1).mean()
            loss_rna = -nb_rna_ll

            # Regularization
            l1_norm_tf = torch.norm(model.topic_tf_decoder, p=1)
            l2_norm_tf = torch.norm(model.topic_tf_decoder, p=2)
            l1_norm_peak = torch.norm(model.topic_peak_decoder, p=1)
            l2_norm_peak = torch.norm(model.topic_peak_decoder, p=2)
            l1_norm_gene_peak = torch.norm(model.gene_peak_factor_learnt, p=1)
            l2_norm_gene_peak = torch.norm(model.gene_peak_factor_learnt, p=2)

            loss_norm = (
                config_file.l1_penalty_topic_tf * l1_norm_tf
                + config_file.l2_penalty_topic_tf * l2_norm_tf
                + config_file.l1_penalty_topic_peak * l1_norm_peak
                + config_file.l2_penalty_topic_peak * l2_norm_peak
                + config_file.l1_penalty_gene_peak * l1_norm_gene_peak
                + config_file.l2_penalty_gene_peak * l2_norm_gene_peak
            )

            total_loss = (
                config_file.weight_atac_phase2 * loss_atac
                + config_file.weight_tf_phase2 * loss_tf
                + config_file.weight_rna_phase2 * loss_rna
                + loss_norm
            )

            running_loss += total_loss.item()
            running_loss_atac += loss_atac.item()
            running_loss_tf += loss_tf.item()
            running_loss_rna += loss_rna.item()
            nbatch += 1

    eval_loss = running_loss / max(1, nbatch)
    eval_loss_atac = running_loss_atac / max(1, nbatch)
    eval_loss_tf = running_loss_tf / max(1, nbatch)
    eval_loss_rna = running_loss_rna / max(1, nbatch)

    return eval_loss, eval_loss_atac, eval_loss_tf, eval_loss_rna


def train_scdori_phases(
    model,
    device,
    train_loader,
    eval_loader,
    rna_anndata,
    atac_anndata,
    num_cells,
    tf_indices,
    encoding_batch_onehot,
    config_file,
):
    """
    Train the scDORI model in two warmup phases:

    The two phases are:
    1) Warmup Phase 1 (ATAC + TF focus).
    2) Warmup Phase 2 (adding RNA).
    Includes early stopping based on validation performance.

    Parameters
    ----------
    model : torch.nn.Module
        The scDORI model to be trained.
    device : torch.device
        The device (CPU or CUDA) for running PyTorch operations.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training set, providing cell indices.
    eval_loader : torch.utils.data.DataLoader
        DataLoader for the validation set, providing cell indices.
    rna_anndata : anndata.AnnData
        RNA single-cell data in AnnData format.
    atac_anndata : anndata.AnnData
        ATAC single-cell data in AnnData format.
    num_cells : np.ndarray
        Number of cells per row (metacells) or ones for single-cell data.
    tf_indices : list or np.ndarray
        Indices of transcription factor genes in the RNA data.
    encoding_batch_onehot : np.ndarray
        One-hot encoding matrix for batch information (cells x num_batches).
    config_file : object
        Configuration with hyperparameters including:
        - learning_rate_scdori
        - max_scdori_epochs
        - epoch_warmup_1
        - weight_atac_phase1, weight_tf_phase1, weight_rna_phase1
        - weight_atac_phase2, weight_tf_phase2, weight_rna_phase2
        - l1_penalty_topic_tf, etc.
        - eval_frequency
        - phase1_patience (early stopping patience for validation loss)

    Returns
    -------
    torch.nn.Module
        The trained scDORI model after both warmup phases.
    """
    validate_topic_penalties(config_file)
    optimizer = torch.optim.Adam(model.parameters(), lr=config_file.learning_rate_scdori)

    best_eval_loss = float("inf")
    val_patience = 0
    max_val_patience = config_file.phase1_patience
    previous_phase = None
    # Notebook inspection only; not part of model state_dict or optimization.
    model.phase1_history = []

    logger.info("Starting scDORI phase 1 training (module 1,2,3) with validation + early stopping.")
    for epoch in range(config_file.max_scdori_epochs):
        phase = get_phase_scdori(epoch, config_file)
        if phase != previous_phase:
            # Adding RNA changes the objective; losses across phases are not comparable.
            best_eval_loss, val_patience = float("inf"), 0
            previous_phase = phase
        weights = get_loss_weights_scdori(phase, config_file)

        model.train()
        running_loss = 0.0
        running_loss_atac = 0.0
        running_loss_tf = 0.0
        running_loss_rna = 0.0
        running_topic_terms = {}
        nbatch = 0

        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch} [{phase}]"):
            cell_indices = batch_data[0].to(device)
            B = cell_indices.shape[0]

            (input_matrix, tf_exp, library_size_value, num_cells_value, input_batch) = create_minibatch(
                device, cell_indices, rna_anndata, atac_anndata, num_cells, tf_indices, encoding_batch_onehot
            )

            rna_input = input_matrix[:, : model.num_genes]
            atac_input = input_matrix[:, model.num_genes :]
            tf_input = tf_exp

            log_lib_rna = library_size_value[:, 0].reshape(-1, 1)
            log_lib_atac = library_size_value[:, 1].reshape(-1, 1)
            batch_onehot = input_batch

            # This phase does not use a separate topic_tf_input
            topic_tf_input = tf_input

            out = model(
                rna_input,
                atac_input,
                tf_input,
                topic_tf_input,
                log_lib_rna,
                log_lib_atac,
                num_cells_value,
                batch_onehot,
                phase=phase,
            )

            preds_atac = out["preds_atac"]
            mu_nb_tf = out["mu_nb_tf"]
            mu_nb_rna = out["mu_nb_rna"]

            # 1) ATAC => Poisson
            library_factor_peak = torch.exp(log_lib_atac.view(B, 1))
            preds_poisson = preds_atac * library_factor_peak
            criterion_poisson = torch.nn.PoissonNLLLoss(log_input=False, reduction="sum")
            loss_atac = criterion_poisson(preds_poisson, atac_input)

            # 2) TF => NB
            alpha_tf = torch.nn.functional.softplus(model.tf_alpha_nb)
            nb_tf_ll = log_nb_positive(tf_input, mu_nb_tf, alpha_tf).sum(dim=1).mean()
            loss_tf = -nb_tf_ll

            # 3) RNA => NB
            alpha_rna = torch.nn.functional.softplus(model.rna_alpha_nb)
            nb_rna_ll = log_nb_positive(rna_input, mu_nb_rna, alpha_rna).sum(dim=1).mean()
            loss_rna = -nb_rna_ll

            # Regularization
            l1_norm_tf = torch.norm(model.topic_tf_decoder, p=1)
            l2_norm_tf = torch.norm(model.topic_tf_decoder, p=2)

            l1_norm_peak = torch.norm(model.topic_peak_decoder, p=1)
            l2_norm_peak = torch.norm(model.topic_peak_decoder, p=2)

            l1_norm_gene_peak = torch.norm(model.gene_peak_factor_learnt, p=1)
            l2_norm_gene_peak = torch.norm(model.gene_peak_factor_learnt, p=2)

            loss_norm = (
                config_file.l1_penalty_topic_tf * l1_norm_tf
                + config_file.l2_penalty_topic_tf * l2_norm_tf
                + config_file.l1_penalty_topic_peak * l1_norm_peak
                + config_file.l2_penalty_topic_peak * l2_norm_peak
                + config_file.l1_penalty_gene_peak * l1_norm_gene_peak
                + config_file.l2_penalty_gene_peak * l2_norm_gene_peak
            )

            total_loss = weights["atac"] * loss_atac + weights["tf"] * loss_tf + weights["rna"] * loss_rna + loss_norm

            for name, term in topic_usage_penalties(out["theta"], config_file).items():
                total_loss = total_loss + term
                running_topic_terms[name] = running_topic_terms.get(name, 0.0) + term.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_loss_atac += loss_atac.item()
            running_loss_tf += loss_tf.item()
            running_loss_rna += loss_rna.item()
            nbatch += 1

            # Clamp gene-peak factors to [0, 1]
            model.gene_peak_factor_learnt.data.clamp_(min=0)
            model.gene_peak_factor_learnt.data.clamp_(max=1)

        epoch_loss = running_loss / max(1, nbatch)
        epoch_loss_atac = running_loss_atac / max(1, nbatch)
        epoch_loss_tf = running_loss_tf / max(1, nbatch)
        epoch_loss_rna = running_loss_rna / max(1, nbatch)
        history = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": epoch_loss,
            "train_atac": epoch_loss_atac,
            "train_tf": epoch_loss_tf,
            "train_rna": epoch_loss_rna,
        }
        model.phase1_history.append(history)
        history.update({f"train_{name}": value / max(1, nbatch) for name, value in running_topic_terms.items()})

        logger.info(
            f"[Train] Epoch={epoch}, Phase={phase}, Loss={epoch_loss:.4f}, "
            f"Atac={epoch_loss_atac:.4f}, TF={epoch_loss_tf:.4f}, RNA={epoch_loss_rna:.4f}"
        )

        # Evaluate periodically
        if (epoch + 1) % config_file.eval_frequency == 0:
            eval_loss, eval_loss_atac, eval_loss_tf, eval_loss_rna = compute_eval_loss_scdori(
                model,
                device,
                eval_loader,
                rna_anndata,
                atac_anndata,
                num_cells,
                tf_indices,
                encoding_batch_onehot,
                config_file,
            )

            logger.info(
                f"[Eval ] Epoch={epoch}, Phase={phase}, EvalLoss={eval_loss:.4f}, "
                f"EvalAtac={eval_loss_atac:.4f}, EvalTF={eval_loss_tf:.4f}, EvalRNA={eval_loss_rna:.4f}"
            )
            history.update(
                validation_loss=eval_loss,
                validation_atac=eval_loss_atac,
                validation_tf=eval_loss_tf,
                validation_rna=eval_loss_rna,
            )

            # Early stopping based on eval_loss
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                val_patience = 0
                save_model_weights(model, Path(config_file.weights_folder_scdori), "scdori_best_eval")
            else:
                val_patience += 1
                if val_patience > max_val_patience:
                    logger.info(f"Validation loss not improving => early stop at epoch={epoch}")
                    break

    logger.info("Finished scDORI phase 1 training (module 1,2,3) with validation checks.")
    return model
