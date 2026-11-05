import logging
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

from .data_io import save_model_weights
from .dataloader import create_minibatch
from .evaluation import get_latent_topics
from .utils import log_nb_positive

logger = logging.getLogger(__name__)


def set_encoder_frozen(model, freeze=True):
    """
    Freeze or unfreeze the encoder parameters.

    Parameters
    ----------
    model : torch.nn.Module
        scDORI model containing the encoder modules.
    freeze : bool, optional
        If True, freeze the encoder parameters; if False, unfreeze them. Default is True.
    """
    for param in model.encoder_rna.parameters():
        param.requires_grad = not freeze
    for param in model.encoder_atac.parameters():
        param.requires_grad = not freeze

    for param in model.mu_theta.parameters():
        param.requires_grad = not freeze

    logger.info(f"Encoder is now {'frozen' if freeze else 'unfrozen'} in GRN phase.")


def set_peak_gene_frozen(model, freeze=True):
    """
    Freeze or unfreeze the peak-gene link parameters.

    Parameters
    ----------
    model : torch.nn.Module
        scDORI model containing the peak-gene factor.
    freeze : bool, optional
        If True, freeze the peak-gene parameters; if False, unfreeze them. Default is True.
    """
    model.gene_peak_factor_learnt.requires_grad = not freeze
    logger.info(f"Peak-gene links are now {'frozen' if freeze else 'unfrozen'} in GRN phase.")


def set_topic_peak_frozen(model, freeze=True):
    """
    Freeze or unfreeze the topic-peak decoder parameters.

    Parameters
    ----------
    model : torch.nn.Module
        scDORI model containing the topic-peak decoder.
    freeze : bool, optional
        If True, freeze the topic-peak decoder; if False, unfreeze it. Default is True.
    """
    model.topic_peak_decoder.requires_grad = not freeze
    logger.info(f"Topic-peak decoder is now {'frozen' if freeze else 'unfrozen'} in GRN phase.")


def set_topic_tf_frozen(model, freeze=True):
    """
    Freeze or unfreeze the topic-TF decoder parameters.

    Parameters
    ----------
    model : torch.nn.Module
        scDORI model containing the topic-TF decoder.
    freeze : bool, optional
        If True, freeze the topic-TF decoder; if False, unfreeze it. Default is True.
    """
    model.topic_tf_decoder.requires_grad = not freeze
    logger.info(f"Topic-tf decoder is now {'frozen' if freeze else 'unfrozen'} in GRN phase.")


def get_tf_expression(
    tf_expression_mode,
    model,
    device,
    train_loader,
    rna_anndata,
    atac_anndata,
    num_cells,
    tf_indices,
    encoding_batch_onehot,
    config_file,
):
    """
    Compute TF expression per topic.

    If `tf_expression_mode` is "True", this function computes the mean TF expression
    for the top-k cells in each topic and zeroes TFs detected in fewer than
    ``tf_expression_min_detection`` of those cells. Otherwise, it uses a normalized
    topic-TF decoder matrix from the model, and there are no cells to apply a floor to.

    Parameters
    ----------
    tf_expression_mode : str
        Mode for TF expression. "True" calculates per-topic TF expression from top-k cells,
        "latent" uses the topic-TF decoder matrix.
    model : torch.nn.Module
        The scDORI model containing encoder and decoder modules.
    device : torch.device
        The device (CPU or CUDA) used for PyTorch tensors.
    train_loader : DataLoader
        DataLoader for training data.
    rna_anndata : anndata.AnnData
        RNA single-cell data in AnnData format.
    atac_anndata : anndata.AnnData
        ATAC single-cell data in AnnData format.
    num_cells : np.ndarray
        number of cells constituting each input metacell, set to 1 for single cell data.
    tf_indices : list of int
        Indices of TF features in the RNA data.
    encoding_batch_onehot : np.ndarray
        One-hot encoding for batch information.
    config_file : python file
        Configuration object with model training.

    Returns
    -------
    torch.Tensor
        A (num_topics x num_tfs) tensor of TF expression values for each topic.
    """
    if tf_expression_mode in ("True", "actual"):
        topics, cell_ids = get_latent_topics(
            model,
            device,
            train_loader,
            rna_anndata,
            atac_anndata,
            num_cells,
            tf_indices,
            encoding_batch_onehot,
            return_indices=True,
        )
        topic_tf, selected = topic_tf_expression_means(
            topics, cell_ids, rna_anndata, tf_indices, config_file.cells_per_topic
        )
        topic_tf = torch.as_tensor(topic_tf, device=device, dtype=model.topic_tf_decoder.dtype)
    elif tf_expression_mode == "latent":
        topic_tf = torch.softmax(model.topic_tf_decoder.detach(), dim=1).to(device)
        selected = None
    else:
        raise ValueError(f"Unknown tf_expression_mode: {tf_expression_mode!r}")
    topic_tf = scale_topic_tf_expression(
        topic_tf, config_file.tf_expression_clamp, getattr(config_file, "tf_expression_scaling", "within_topic")
    )
    if selected is not None:
        _, keep = topic_tf_detection_filter(selected, rna_anndata, tf_indices, config_file)
        topic_tf = topic_tf * torch.as_tensor(keep, device=device, dtype=topic_tf.dtype)
    from .topic_selection import excluded_grn_topics

    topic_tf[excluded_grn_topics(model)] = 0
    return topic_tf


def topic_tf_expression_means(topics, cell_ids, rna_anndata, tf_indices, cells_per_topic):
    """Unscaled per-topic TF expression used by the "True" TF-expression gate.

    For every topic, take the ``cells_per_topic`` cells with the highest theta,
    normalise each cell's TF counts by its TF-panel total, average them and
    multiply by the median TF total.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        ``(num_topics, num_tfs)`` float32 means and the selected absolute cell IDs,
        shape ``(cells_per_topic, num_topics)``.
    """
    if cells_per_topic <= 0:
        raise ValueError("cells_per_topic must be positive")
    # Canonical order makes selection independent of loader shuffling, including ties.
    cell_ids, positions = np.unique(cell_ids, return_index=True)
    topics = topics[positions]
    top_k = np.argsort(topics, axis=0, kind="stable")[-cells_per_topic:]
    rna_tf_vals = rna_anndata.X[cell_ids][:, tf_indices]
    if sp.issparse(rna_tf_vals):
        rna_tf_vals = rna_tf_vals.toarray(order="C")
    rna_tf_vals = np.ascontiguousarray(rna_tf_vals, dtype=np.float32)
    totals = rna_tf_vals.sum(axis=1, keepdims=True)
    median_cell = np.median(totals)
    normalized = np.divide(rna_tf_vals, totals, out=np.zeros_like(rna_tf_vals), where=totals > 0)
    topic_tf = np.stack([normalized[top_k[:, t]].mean(axis=0) for t in range(topics.shape[1])])
    return topic_tf * median_cell, cell_ids[top_k]


def topic_tf_detection(selected_cells, rna_anndata, tf_indices):
    """Fraction of each topic's selected cells with at least one RNA count, per TF.

    ``selected_cells`` is the second result of ``topic_tf_expression_means``, shape
    ``(cells_per_topic, num_topics)``. Returns a ``(num_topics, num_tfs)`` float64 array.
    """
    detection = []
    for cells in np.asarray(selected_cells).T:
        counts = rna_anndata.X[cells][:, tf_indices]
        detected = (counts > 0).sum(axis=0)
        detection.append(np.asarray(detected, dtype=np.float64).ravel() / len(cells))
    return np.stack(detection)


def tfs_exempt_from_detection_floor(rna_anndata, tf_indices, config_file):
    """Boolean mask of TFs the detection floor never removes.

    These are the TFs requested with ``tfs_user`` in preprocessing and those listed in
    ``tf_expression_keep_tfs``. A listed name outside the fitted TF panel is an error, since
    keeping it would otherwise silently do nothing.
    """
    names = np.asarray(rna_anndata.var_names)[np.asarray(tf_indices)]
    keep = set(getattr(config_file, "tf_expression_keep_tfs", None) or [])
    missing = sorted(keep - set(names))
    if missing:
        raise ValueError(
            f"tf_expression_keep_tfs lists TFs that are not in the fitted TF panel: {missing}. "
            "A TF must be selected in preprocessing (tfs_user) before it can be kept here."
        )
    requested = rna_anndata.uns.get("scdori_requested_features", {}).get("tfs_retained", [])
    return np.isin(names, list(keep) + [str(tf) for tf in np.atleast_1d(requested)])


def topic_tf_detection_filter(selected_cells, rna_anndata, tf_indices, config_file):
    """Apply ``tf_expression_min_detection``: which TFs have enough RNA evidence in each topic.

    Returns ``(detection, keep)``, both ``(num_topics, num_tfs)``. ``keep`` is True where the TF was
    seen in at least the required fraction of the topic's selected cells, or is exempt. A config
    from before this setting existed keeps every TF.
    """
    floor = getattr(config_file, "tf_expression_min_detection", 0.0)
    if not 0 <= floor <= 1:
        raise ValueError("tf_expression_min_detection must be between 0 and 1")
    detection = topic_tf_detection(selected_cells, rna_anndata, tf_indices)
    return detection, (detection >= floor) | tfs_exempt_from_detection_floor(rna_anndata, tf_indices, config_file)


def scale_topic_tf_expression(topic_tf, clamp, policy="mean_within_across"):
    """Scale TF expression and zero values strictly below the cutoff.

    ``within_topic`` retains the historical row-wise operation exactly.
    ``mean_within_across`` averages that score with each TF's min-max score
    across all fitted topics, then applies the cutoff to the average. This
    is the default policy. All input topics participate
    before any phase-2 topic mask; constant axes contribute zero. This is a
    training multiplier, not a downstream ranking or an expression probability.
    """
    if policy not in {"within_topic", "mean_within_across"}:
        raise ValueError(f"Unknown tf_expression_scaling: {policy!r}")
    if not 0 <= clamp <= 1:
        raise ValueError("tf_expression_clamp must be between 0 and 1")
    if topic_tf.ndim != 2 or not all(topic_tf.shape):
        raise ValueError("TF expression must be a nonempty topic-by-TF matrix")
    if not torch.isfinite(topic_tf).all() or (topic_tf < 0).any():
        raise ValueError("TF expression must be finite and nonnegative")
    low = topic_tf.amin(dim=1, keepdim=True)
    high = topic_tf.amax(dim=1, keepdim=True)
    scaled = (topic_tf - low) / (high - low + 1e-9)
    if policy == "mean_within_across":
        low = topic_tf.amin(dim=0, keepdim=True)
        high = topic_tf.amax(dim=0, keepdim=True)
        across = (topic_tf - low) / (high - low + 1e-9)
        scaled = (scaled + across) / 2
    return scaled.masked_fill(scaled < clamp, 0)


def compute_eval_loss_grn(
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
    Compute the validation (evaluation) loss for the GRN phase.

    This function evaluates loss components for ATAC, TF, RNA, and RNA-from-GRN
    on a validation dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The scDORI model.
    device : torch.device
        The device (CPU or CUDA) used for PyTorch tensors.
    train_loader : DataLoader
        DataLoader for the training set (used to compute TF expression).
    eval_loader : DataLoader
        DataLoader for the validation set.
    rna_anndata : anndata.AnnData
        RNA single-cell data in AnnData format.
    atac_anndata : anndata.AnnData
        ATAC single-cell data in AnnData format.
    num_cells : np.ndarray
        number of cells constituting each input metacell, set to 1 for single cell data
    tf_indices : list of int
        Indices of TF features in the RNA data.
    encoding_batch_onehot : np.ndarray
        One-hot encoding for batch information.
    config_file : python file
        Configuration file for model training.

    Returns
    -------
    tuple of float
        A tuple containing:
        (eval_loss, eval_loss_atac, eval_loss_tf, eval_loss_rna, eval_loss_rna_grn).
    """
    model.eval()
    running_loss = 0.0
    running_loss_atac = 0.0
    running_loss_tf = 0.0
    running_loss_rna = 0.0
    running_loss_rna_grn = 0.0
    nbatch = 0

    topic_tf_input = get_tf_expression(
        config_file.tf_expression_mode,
        model,
        device,
        train_loader,
        rna_anndata,
        atac_anndata,
        num_cells,
        tf_indices,
        encoding_batch_onehot,
        config_file,
    )

    with torch.no_grad():
        for batch_data in eval_loader:
            cell_indices = batch_data[0].to(device)
            B = cell_indices.shape[0]

            input_matrix, tf_exp, library_size_value, num_cells_value, input_batch = create_minibatch(
                device, cell_indices, rna_anndata, atac_anndata, num_cells, tf_indices, encoding_batch_onehot
            )
            rna_input = input_matrix[:, : model.num_genes]
            atac_input = input_matrix[:, model.num_genes :]
            log_lib_rna = library_size_value[:, 0].reshape(-1, 1)
            log_lib_atac = library_size_value[:, 1].reshape(-1, 1)

            out = model(
                rna_input,
                atac_input,
                tf_exp,
                topic_tf_input,
                log_lib_rna,
                log_lib_atac,
                num_cells_value,
                input_batch,
                phase="grn",
            )
            preds_atac = out["preds_atac"]
            mu_nb_tf = out["mu_nb_tf"]
            mu_nb_rna = out["mu_nb_rna"]
            mu_nb_rna_grn = out["mu_nb_rna_grn"]

            criterion_poisson = torch.nn.PoissonNLLLoss(log_input=False, reduction="sum")
            library_factor_peak = torch.exp(log_lib_atac.view(B, 1))
            preds_poisson = preds_atac * library_factor_peak
            loss_atac = criterion_poisson(preds_poisson, atac_input)

            alpha_tf = torch.nn.functional.softplus(model.tf_alpha_nb)
            nb_tf_ll = log_nb_positive(tf_exp, mu_nb_tf, alpha_tf).sum(dim=1).mean()
            loss_tf = -nb_tf_ll

            alpha_rna = torch.nn.functional.softplus(model.rna_alpha_nb)
            nb_rna_ll = log_nb_positive(rna_input, mu_nb_rna, alpha_rna).sum(dim=1).mean()
            loss_rna = -nb_rna_ll

            nb_rna_grn_ll = log_nb_positive(rna_input, mu_nb_rna_grn, alpha_rna).sum(dim=1).mean()
            loss_rna_grn = -nb_rna_grn_ll

            l1_norm_tf = torch.norm(model.topic_tf_decoder, p=1)
            l2_norm_tf = torch.norm(model.topic_tf_decoder, p=2)
            l1_norm_peak = torch.norm(model.topic_peak_decoder, p=1)
            l2_norm_peak = torch.norm(model.topic_peak_decoder, p=2)
            l1_norm_gene_peak = torch.norm(model.gene_peak_factor_learnt, p=1)
            l2_norm_gene_peak = torch.norm(model.gene_peak_factor_learnt, p=2)
            l1_norm_grn_activator = torch.norm(model.tf_gene_topic_activator_grn, p=1)
            l1_norm_grn_repressor = torch.norm(model.tf_gene_topic_repressor_grn, p=1)

            loss_norm = (
                config_file.l1_penalty_topic_tf * l1_norm_tf
                + config_file.l2_penalty_topic_tf * l2_norm_tf
                + config_file.l1_penalty_topic_peak * l1_norm_peak
                + config_file.l2_penalty_topic_peak * l2_norm_peak
                + config_file.l1_penalty_gene_peak * l1_norm_gene_peak
                + config_file.l2_penalty_gene_peak * l2_norm_gene_peak
                + config_file.l1_penalty_grn_activator * l1_norm_grn_activator
                + config_file.l1_penalty_grn_repressor * l1_norm_grn_repressor
            )

            total_loss = (
                config_file.weight_atac_grn * loss_atac
                + config_file.weight_tf_grn * loss_tf
                + config_file.weight_rna_grn * loss_rna
                + config_file.weight_rna_from_grn * loss_rna_grn
                + loss_norm
            )

            running_loss += total_loss.item()
            running_loss_atac += loss_atac.item()
            running_loss_tf += loss_tf.item()
            running_loss_rna += loss_rna.item()
            running_loss_rna_grn += loss_rna_grn.item()
            nbatch += 1

    eval_loss = running_loss / max(1, nbatch)
    eval_loss_atac = running_loss_atac / max(1, nbatch)
    eval_loss_tf = running_loss_tf / max(1, nbatch)
    eval_loss_rna = running_loss_rna / max(1, nbatch)
    eval_loss_rna_grn = running_loss_rna_grn / max(1, nbatch)

    return eval_loss, eval_loss_atac, eval_loss_tf, eval_loss_rna, eval_loss_rna_grn


def train_model_grn(
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
    Train the model in Phase 2 (GRN phase).

    In this phase, the model focuses on learning activator and repressor TF-gene links per topic (module 4 of scDORI). Other modules of the model can be optionally frozen
    or unfrozen based on the configuration.

    Parameters
    ----------
    model : torch.nn.Module
        The scDORI model to train.
    device : torch.device
        The device (CPU or CUDA) used for PyTorch tensors.
    train_loader : DataLoader
        DataLoader for the training set.
    eval_loader : DataLoader
        DataLoader for the validation set, used to check early stopping criteria.
    rna_anndata : anndata.AnnData
        RNA single-cell data in AnnData format.
    atac_anndata : anndata.AnnData
        ATAC single-cell data in AnnData format.
    num_cells : np.ndarray
        number of cells constituting each input metacell, set to 1 for single cell data
    tf_indices : list of int
        Indices of TF features in the RNA data.
    encoding_batch_onehot : np.ndarray
        One-hot encoding for batch information.
    config_file : python file
        Configuration file for model training.

    Returns
    -------
    torch.nn.Module
        The trained model after the GRN phase completes or early stopping occurs.
    """
    if getattr(config_file, "freeze_phase1_in_grn", False):
        from .frozen_grn import train_frozen_grn

        return train_frozen_grn(
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
        )
    from .topic_selection import apply_grn_topic_selection, write_grn_topics

    # Non-frozen phase 2 must retain every topic: encoder/BN state can change.
    selection = apply_grn_topic_selection(model, config_file)
    write_grn_topics(config_file.weights_folder_grn, selection)
    # A state_dict does not restore requires_grad flags when resuming phase 2.
    model.gene_peak_factor_fixed.requires_grad_(False)
    model.tf_binding_matrix_activator.requires_grad_(False)
    model.tf_binding_matrix_repressor.requires_grad_(False)
    model.tf_gene_topic_activator_grn.requires_grad_(True)
    model.tf_gene_topic_repressor_grn.requires_grad_(True)

    if not config_file.update_encoder_in_grn:
        set_encoder_frozen(model, freeze=True)
    else:
        set_encoder_frozen(model, freeze=False)

    if not config_file.update_peak_gene_in_grn:
        set_peak_gene_frozen(model, freeze=True)
    else:
        set_peak_gene_frozen(model, freeze=False)

    if not config_file.update_topic_peak_in_grn:
        set_topic_peak_frozen(model, freeze=True)
    else:
        set_topic_peak_frozen(model, freeze=False)

    if not config_file.update_topic_tf_in_grn:
        set_topic_tf_frozen(model, freeze=True)
    else:
        set_topic_tf_frozen(model, freeze=False)

    optimizer_grn = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config_file.learning_rate_grn
    )

    best_eval_loss = float("inf")
    val_patience = 0
    max_val_patience = config_file.grn_val_patience
    topic_tf_input = None

    if config_file.tf_expression_mode in ("True", "actual"):
        topic_tf_input = get_tf_expression(
            config_file.tf_expression_mode,
            model,
            device,
            train_loader,
            rna_anndata,
            atac_anndata,
            num_cells,
            tf_indices,
            encoding_batch_onehot,
            config_file,
        )

    logger.info("Starting GRN training")
    for epoch in range(config_file.max_grn_epochs):
        model.train()
        running_loss = 0.0
        running_loss_atac = 0.0
        running_loss_tf = 0.0
        running_loss_rna = 0.0
        running_loss_rna_grn = 0.0
        nbatch = 0

        # If the encoder is being updated, recalc topic_tf_input each epoch:
        if config_file.update_encoder_in_grn:
            topic_tf_input = get_tf_expression(
                config_file.tf_expression_mode,
                model,
                device,
                train_loader,
                rna_anndata,
                atac_anndata,
                num_cells,
                tf_indices,
                encoding_batch_onehot,
                config_file,
            )

        for batch_data in tqdm(train_loader, desc=f"GRN Epoch {epoch}"):
            cell_indices = batch_data[0].to(device)
            B = cell_indices.shape[0]

            input_matrix, tf_exp, library_size_value, num_cells_value, input_batch = create_minibatch(
                device, cell_indices, rna_anndata, atac_anndata, num_cells, tf_indices, encoding_batch_onehot
            )
            rna_input = input_matrix[:, : model.num_genes]
            atac_input = input_matrix[:, model.num_genes :]
            tf_input = tf_exp
            log_lib_rna = library_size_value[:, 0].reshape(-1, 1)
            log_lib_atac = library_size_value[:, 1].reshape(-1, 1)
            batch_onehot = input_batch

            if config_file.tf_expression_mode == "latent":
                topic_tf_input = get_tf_expression(
                    config_file.tf_expression_mode,
                    model,
                    device,
                    train_loader,
                    rna_anndata,
                    atac_anndata,
                    num_cells,
                    tf_indices,
                    encoding_batch_onehot,
                    config_file,
                )

            out = model(
                rna_input,
                atac_input,
                tf_input,
                topic_tf_input,
                log_lib_rna,
                log_lib_atac,
                num_cells_value,
                batch_onehot,
                phase="grn",
            )
            preds_atac = out["preds_atac"]
            mu_nb_tf = out["mu_nb_tf"]
            mu_nb_rna = out["mu_nb_rna"]
            mu_nb_rna_grn = out["mu_nb_rna_grn"]

            criterion_poisson = torch.nn.PoissonNLLLoss(log_input=False, reduction="sum")
            library_factor_peak = torch.exp(log_lib_atac.view(B, 1))
            preds_poisson = preds_atac * library_factor_peak
            loss_atac = criterion_poisson(preds_poisson, atac_input)

            alpha_tf = torch.nn.functional.softplus(model.tf_alpha_nb)
            nb_tf_ll = log_nb_positive(tf_input, mu_nb_tf, alpha_tf).sum(dim=1).mean()
            loss_tf = -nb_tf_ll

            alpha_rna = torch.nn.functional.softplus(model.rna_alpha_nb)
            nb_rna_ll = log_nb_positive(rna_input, mu_nb_rna, alpha_rna).sum(dim=1).mean()
            loss_rna = -nb_rna_ll

            nb_rna_grn_ll = log_nb_positive(rna_input, mu_nb_rna_grn, alpha_rna).sum(dim=1).mean()
            loss_rna_grn = -nb_rna_grn_ll

            l1_norm_tf = torch.norm(model.topic_tf_decoder, p=1)
            l2_norm_tf = torch.norm(model.topic_tf_decoder, p=2)
            l1_norm_peak = torch.norm(model.topic_peak_decoder, p=1)
            l2_norm_peak = torch.norm(model.topic_peak_decoder, p=2)
            l1_norm_gene_peak = torch.norm(model.gene_peak_factor_learnt, p=1)
            l2_norm_gene_peak = torch.norm(model.gene_peak_factor_learnt, p=2)
            l1_norm_grn_activator = torch.norm(model.tf_gene_topic_activator_grn, p=1)
            l1_norm_grn_repressor = torch.norm(model.tf_gene_topic_repressor_grn, p=1)

            loss_norm = (
                config_file.l1_penalty_topic_tf * l1_norm_tf
                + config_file.l2_penalty_topic_tf * l2_norm_tf
                + config_file.l1_penalty_topic_peak * l1_norm_peak
                + config_file.l2_penalty_topic_peak * l2_norm_peak
                + config_file.l1_penalty_gene_peak * l1_norm_gene_peak
                + config_file.l2_penalty_gene_peak * l2_norm_gene_peak
                + config_file.l1_penalty_grn_activator * l1_norm_grn_activator
                + config_file.l1_penalty_grn_repressor * l1_norm_grn_repressor
            )

            total_loss = (
                config_file.weight_atac_grn * loss_atac
                + config_file.weight_tf_grn * loss_tf
                + config_file.weight_rna_grn * loss_rna
                + config_file.weight_rna_from_grn * loss_rna_grn
                + loss_norm
            )

            optimizer_grn.zero_grad()
            total_loss.backward()
            optimizer_grn.step()

            running_loss += total_loss.item()
            running_loss_atac += loss_atac.item()
            running_loss_tf += loss_tf.item()
            running_loss_rna += loss_rna.item()
            running_loss_rna_grn += loss_rna_grn.item()
            nbatch += 1

            model.gene_peak_factor_learnt.data.clamp_(min=0)
            model.gene_peak_factor_learnt.data.clamp_(max=1)

        epoch_loss = running_loss / max(1, nbatch)
        epoch_loss_atac = running_loss_atac / max(1, nbatch)
        epoch_loss_tf = running_loss_tf / max(1, nbatch)
        epoch_loss_rna = running_loss_rna / max(1, nbatch)
        epoch_loss_rna_grn = running_loss_rna_grn / max(1, nbatch)

        logger.info(
            f"[GRN-Train] Epoch={epoch}, Loss={epoch_loss:.4f},"
            f"Atac={epoch_loss_atac:.4f}, TF={epoch_loss_tf:.4f}, "
            f"RNA={epoch_loss_rna:.4f}, RNA-GRN={epoch_loss_rna_grn:.4f}"
        )

        # Evaluate every config.eval_frequency epochs
        if (epoch + 1) % config_file.eval_frequency == 0:
            eval_loss, eval_loss_atac, eval_loss_tf, eval_loss_rna, eval_loss_rna_grn = compute_eval_loss_grn(
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
            )

            logger.info(
                f"[GRN-Eval] Epoch={epoch}, EvalLoss={eval_loss:.4f},"
                f"EvalAtac={eval_loss_atac:.4f}, EvalTF={eval_loss_tf:.4f}, "
                f"EvalRNA={eval_loss_rna:.4f}, EvalRNA-GRN={eval_loss_rna_grn:.4f}"
            )

            # Early stopping on eval_loss_rna_grn
            if eval_loss_rna_grn < best_eval_loss:
                best_eval_loss = eval_loss_rna_grn
                val_patience = 0
                save_model_weights(model, Path(config_file.weights_folder_grn), "scdori_best_eval")
            else:
                val_patience += 1
                if val_patience > max_val_patience:
                    logger.info(f"[GRN] Validation not improving => early stop at epoch={epoch}.")
                    break

    logger.info("Finished Phase 3 (GRN) with validation checks.")
    return model
