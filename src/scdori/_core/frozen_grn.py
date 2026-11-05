"""Training from fixed phase 1 features; all caches live for one training call."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from .data_io import save_model_weights
from .dataloader import create_minibatch, read_count_rows
from .topic_selection import apply_grn_topic_selection, grn_topic_ids, sequential_loader, write_grn_topics
from .utils import log_nb_positive

logger = logging.getLogger(__name__)


def estimate_frozen_cache_bytes(model, num_cells):
    """Estimate retained cache storage; excludes model, optimizer and temporaries."""
    size = model.topic_peak_decoder.element_size()
    grn_topics = len(grn_topic_ids(model))
    return {
        "cpu_cell_cache": num_cells * (model.num_topics + 4) * size,
        "device_grn_factors": 2 * grn_topics * model.num_tfs * model.num_genes * size,
        "device_topic_tf": model.num_topics * model.num_tfs * size,
    }


def freeze_phase1(model):
    """Freeze phase 1 parameters and state, enabling only the GRN branch."""
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith(("tf_gene_topic_", "rna_grn_batch_")))
        param.grad = None
    model.eval()


@torch.no_grad()
def build_grn_factors(model):
    """Compute normalized ATAC priors once, preserving reference reduction order.

    Returns (activator, repressor), each (selected topics, TFs, genes), in the order of
    ``grn_topic_ids(model)``.
    """
    names = (
        "topic_peak_decoder",
        "gene_peak_factor_learnt",
        "gene_peak_factor_fixed",
        "tf_binding_matrix_activator",
        "tf_binding_matrix_repressor",
    )
    if any(getattr(model, name).requires_grad for name in names):
        raise ValueError("GRN factors require frozen topic, peak-gene, and TF-binding parameters")
    peaks = F.softmax(model.topic_peak_decoder, dim=1)
    gene_peak = (model.gene_peak_factor_learnt * model.gene_peak_factor_fixed).T
    denominator = gene_peak.sum(dim=0, keepdim=True) + 1e-7
    topics = grn_topic_ids(model)
    shape = (len(topics), model.num_tfs, model.num_genes)
    activator = gene_peak.new_empty(shape)
    repressor = gene_peak.new_empty(shape)
    for i, topic in enumerate(topics):
        act = model.tf_binding_matrix_activator.T @ (peaks[topic][:, None] * gene_peak)
        rep = model.tf_binding_matrix_repressor.T @ ((1 / (peaks[topic] + 1e-20))[:, None] * gene_peak)
        act = act / denominator
        rep = rep / denominator
        activator[i] = act / (act.max() + 1e-15)
        repressor[i] = rep / (rep.min() + 1e-15)
    if not torch.isfinite(activator).all() or not torch.isfinite(repressor).all():
        raise ValueError("Non-finite ATAC GRN factors; check input priors and phase 1 weights")
    return activator, repressor


def forward_from_topics(model, theta, library, batch_onehot, topic_tf, factors):
    """Reconstruct RNA using fixed topics, TF expression, and ATAC priors."""
    act, rep = factors
    # Preserve the reference reduction order: tiny float32 differences before
    # BatchNorm can otherwise be amplified into different Adam updates.
    topics = grn_topic_ids(model)
    contributions = torch.stack(
        [
            topic_tf[t] @ (act[i] * F.relu(model.tf_gene_topic_activator_grn[t]))
            + topic_tf[t] @ (rep[i] * -1 * F.relu(model.tf_gene_topic_repressor_grn[t]))
            for i, t in enumerate(topics)
        ]
    )
    if model.grn_topics is not None:  # excluded topics drop theta[:, excluded] @ C[excluded]
        theta = theta[:, topics]
    logits = theta @ contributions + batch_onehot @ model.rna_grn_batch_factor
    return F.softmax(model.rna_grn_batch_norm(logits), dim=1) * library


def _auxiliary_losses(model, out, rna, atac, tf, libraries):
    atac_loss = F.poisson_nll_loss(
        out["preds_atac"] * libraries[:, 1:].exp(), atac, log_input=False, reduction="none"
    ).sum(dim=1)
    tf_loss = -log_nb_positive(tf, out["mu_nb_tf"], F.softplus(model.tf_alpha_nb)).sum(dim=1)
    rna_loss = -log_nb_positive(rna, out["mu_nb_rna"], F.softplus(model.rna_alpha_nb)).sum(dim=1)
    return torch.stack((atac_loss, tf_loss, rna_loss), dim=1)


@dataclass
class FrozenCellCache:
    """CPU tensors indexed by absolute cell ID; no observed ATAC is retained."""

    theta: torch.Tensor
    library: torch.Tensor
    auxiliary: torch.Tensor


@torch.no_grad()
def build_cell_cache(model, device, rna, atac, num_cells, tf_indices, batches, batch_size):
    """Encode cells once and retain compact features and constant loss terms."""
    if any(
        p.requires_grad
        for name, p in model.named_parameters()
        if not name.startswith(("tf_gene_topic_", "rna_grn_batch_"))
    ):
        raise ValueError("Cell caching requires all phase 1 parameters to be frozen")
    model.eval()
    topic_gene = model.topic_gene_matrix()
    cache = FrozenCellCache(
        torch.empty((rna.n_obs, model.num_topics), dtype=model.topic_peak_decoder.dtype),
        torch.empty((rna.n_obs, 1), dtype=model.topic_peak_decoder.dtype),
        torch.empty((rna.n_obs, 3), dtype=model.topic_peak_decoder.dtype),
    )
    for start in range(0, rna.n_obs, batch_size):
        ids = torch.arange(start, min(start + batch_size, rna.n_obs))
        inputs, tf, libraries, cells, batch = create_minibatch(device, ids, rna, atac, num_cells, tf_indices, batches)
        x, a = inputs[:, : model.num_genes], inputs[:, model.num_genes :]
        out = model(x, a, tf, None, libraries[:, :1], libraries[:, 1:], cells, batch, fixed_topic_gene=topic_gene)
        cache.theta[ids] = out["theta"].cpu()
        cache.library[ids] = out["library_factor_rna"].cpu()
        cache.auxiliary[ids] = _auxiliary_losses(model, out, x, a, tf, libraries).cpu()
    return cache


def _regularization(model, config, grn):
    if grn:
        return (
            config.l1_penalty_grn_activator * model.tf_gene_topic_activator_grn.abs().sum()
            + config.l1_penalty_grn_repressor * model.tf_gene_topic_repressor_grn.abs().sum()
        )
    result = model.topic_peak_decoder.new_zeros(())
    for suffix, name in (
        ("topic_tf", "topic_tf_decoder"),
        ("topic_peak", "topic_peak_decoder"),
        ("gene_peak", "gene_peak_factor_learnt"),
    ):
        param = getattr(model, name)
        result = result + getattr(config, "l1_penalty_" + suffix) * param.norm(p=1)
        result = result + getattr(config, "l2_penalty_" + suffix) * param.norm(p=2)
    return result


def _train_theta(model, device, loader, rna, atac, num_cells, tf_indices, batches):
    from .evaluation import get_latent_topics

    return get_latent_topics(model, device, sequential_loader(loader), rna, atac, num_cells, tf_indices, batches)


def train_frozen_grn(model, device, train_loader, eval_loader, rna, atac, num_cells, tf_indices, batches, config):
    """Train GRN parameters with fixed phase 1 state, with or without caching."""
    from .train_grn import get_tf_expression

    flags = ("update_encoder_in_grn", "update_peak_gene_in_grn", "update_topic_peak_in_grn", "update_topic_tf_in_grn")
    if any(getattr(config, flag) for flag in flags):
        raise ValueError("freeze_phase1_in_grn cannot be combined with update_*_in_grn flags")
    if not rna.obs_names.equals(atac.obs_names):
        raise ValueError("RNA and ATAC cell IDs must have identical order")
    freeze_phase1(model)
    selection = apply_grn_topic_selection(
        model, config, lambda: _train_theta(model, device, train_loader, rna, atac, num_cells, tf_indices, batches)
    )
    write_grn_topics(config.weights_folder_grn, selection)
    logger.info("Frozen cache storage estimate (bytes): %s", estimate_frozen_cache_bytes(model, rna.n_obs))
    topic_tf = get_tf_expression(
        config.tf_expression_mode, model, device, train_loader, rna, atac, num_cells, tf_indices, batches, config
    )
    use_cache = getattr(config, "cache_frozen_grn", True)
    factors = build_grn_factors(model) if use_cache else None
    cache = (
        build_cell_cache(model, device, rna, atac, num_cells, tf_indices, batches, config.batch_size_cell_prediction)
        if use_cache
        else None
    )
    with torch.no_grad():
        fixed_penalty = _regularization(model, config, grn=False)
        alpha = F.softplus(model.rna_alpha_nb)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate_grn)

    def run_epoch(loader, training):
        # All phase 1 BN statistics and dropout remain fixed even during GRN training.
        model.eval()
        model.rna_grn_batch_norm.train(training)
        totals = torch.zeros(5, device=device)
        count = 0
        with torch.set_grad_enabled(training):
            for (cell_ids,) in loader:
                ids = cell_ids.cpu().long()
                if cache is not None:
                    target = read_count_rows(rna, ids.numpy())
                    if sp.issparse(target):
                        target = target.toarray(order="C")
                    x = torch.as_tensor(np.ascontiguousarray(target), device=device, dtype=alpha.dtype)
                    batch = torch.as_tensor(batches[ids.numpy()], device=device, dtype=alpha.dtype)
                    mu = forward_from_topics(
                        model, cache.theta[ids].to(device), cache.library[ids].to(device), batch, topic_tf, factors
                    )
                    aux = cache.auxiliary[ids].to(device)
                else:
                    inputs, tf, libraries, cells, batch = create_minibatch(
                        device, ids, rna, atac, num_cells, tf_indices, batches
                    )
                    x, a = inputs[:, : model.num_genes], inputs[:, model.num_genes :]
                    out = model(x, a, tf, topic_tf, libraries[:, :1], libraries[:, 1:], cells, batch, phase="grn")
                    mu = out["mu_nb_rna_grn"]
                    aux = _auxiliary_losses(model, out, x, a, tf, libraries)
                loss_atac, loss_tf, loss_rna = aux[:, 0].sum(), aux[:, 1].mean(), aux[:, 2].mean()
                loss_grn = -log_nb_positive(x, mu, alpha).sum(dim=1).mean()
                loss = (
                    config.weight_atac_grn * loss_atac
                    + config.weight_tf_grn * loss_tf
                    + config.weight_rna_grn * loss_rna
                    + config.weight_rna_from_grn * loss_grn
                    + fixed_penalty
                    + _regularization(model, config, grn=True)
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite frozen GRN loss")
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                totals += torch.stack((loss, loss_atac, loss_tf, loss_rna, loss_grn)).detach()
                count += 1
        if count == 0:
            raise ValueError("GRN training and evaluation loaders must not be empty")
        return (totals / count).cpu().tolist()

    best, patience = float("inf"), 0
    for epoch in range(config.max_grn_epochs):
        train_losses = run_epoch(train_loader, True)
        logger.info("[Frozen-GRN] epoch=%d train=%s cached=%s", epoch, train_losses, use_cache)
        if (epoch + 1) % config.eval_frequency == 0:
            losses = run_epoch(eval_loader, False)
            logger.info("[Frozen-GRN] epoch=%d eval=%s", epoch, losses)
            if losses[-1] < best:
                best, patience = losses[-1], 0
                save_model_weights(model, Path(config.weights_folder_grn), "scdori_best_eval")
            else:
                patience += 1
                if patience > config.grn_val_patience:
                    break
    model.eval()
    return model
