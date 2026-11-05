import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import torch.nn as nn
import umap
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_best_model(model, best_model_path, device):
    """
    Load the best model weights from disk into the given model.

    Parameters
    ----------
    model : torch.nn.Module
        The model instance to which the weights will be loaded.
    best_model_path : str or Path
        Path to the file containing the best model weights.
    device : torch.device
        The device (CPU or CUDA) where the model will be moved.

    Returns
    -------
    torch.nn.Module
        The same model, now loaded with weights and set to eval mode.

    Raises
    ------
    FileNotFoundError
        If the specified `best_model_path` does not exist.
    """
    if not os.path.isfile(best_model_path):
        raise FileNotFoundError(f"Best model file {best_model_path} not found.")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    from .topic_selection import load_grn_topics

    sidecar = Path(best_model_path).with_suffix(".topics.json")
    load_grn_topics(model, sidecar if sidecar.is_file() else Path(best_model_path).parent)
    model.to(device)
    model.eval()
    logger.info(f"Loaded best model weights from {best_model_path}")
    return model


def compute_neighbors_umap(rna_anndata, umap_n_neighbors, umap_min_dist, umap_random_state, rep_key="X_scdori"):
    """
    Compute neighbors and UMAP on the specified representation in an AnnData object.

    Parameters
    ----------
    rna_anndata : anndata.AnnData
        An AnnData object containing single-cell RNA data.
    umap_n_neighbors : int
        The number of neighbors for the UMAP computation.
    umap_min_dist : float
        The minimum distance between points in the UMAP embedding.
    umap_random_state : int
        The random seed for the UMAP computation.
    rep_key : str, optional
        The key in `rna_anndata.obsm` that holds the latent representation used for computing UMAP.
        Default is "X_scdori".

    Returns
    -------
    None
        Updates `rna_anndata` in place with neighbor graph and UMAP coordinates.
    """
    logger.info("=== Computing neighbors + UMAP on scDORI latent ===")
    sc.pp.neighbors(rna_anndata, use_rep=rep_key, n_neighbors=umap_n_neighbors)
    sc.tl.umap(rna_anndata, min_dist=umap_min_dist, spread=1.0, random_state=umap_random_state)
    logger.info("Done. UMAP stored in rna_anndata.obsm['X_umap'].")


def compute_topic_peak_umap(model, device, umap_n_neighbors, umap_min_dist, umap_random_state):
    """
    Compute a UMAP embedding of the topic-peak decoder matrix. Each point on this embedding is a peak.

    Steps
    -----
    1. Apply softmax to `model.topic_peak_decoder` => (num_topics, num_peaks).
    2. Min-max normalize across topics.
    3. Transpose to get (num_peaks, num_topics).
    4. Run UMAP on the resulting matrix to get a (num_peaks, 2) embedding.

    Parameters
    ----------
    model : torch.nn.Module
        The scDORI model containing the topic_peak_decoder.
    device : torch.device
        The device (CPU or CUDA) used for PyTorch operations.
    umap_n_neighbors : int
        The number of neighbors for the UMAP computation.
    umap_min_dist : float
        The minimum distance between points in the UMAP embedding.
    umap_random_state : int
        The random seed for the UMAP computation.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        embedding_peaks : shape (num_peaks, 2)
            The UMAP embedding of the peaks.
        peak_mat : shape (num_peaks, num_topics)
            The min-max normalized topic-peak matrix.
    """
    model.eval()
    with torch.no_grad():
        topic_peaks = model.topic_peak_decoder.detach().to(device)
        topic_peaks_smx = torch.nn.functional.softmax(topic_peaks, dim=1)
        # shape => (num_topics, num_peaks)

    # min-max across topics
    tmin, _ = torch.min(topic_peaks_smx, dim=0, keepdim=True)
    tmax, _ = torch.max(topic_peaks_smx, dim=0, keepdim=True)
    topic_peaks_norm = (topic_peaks_smx - tmin) / (tmax - tmin + 1e-8)

    peak_mat = topic_peaks_norm.T.cpu().numpy()  # shape => (num_peaks, num_topics)

    reducer = umap.UMAP(n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, random_state=umap_random_state)
    embedding_peaks = reducer.fit_transform(peak_mat)
    logger.info(
        f"Done. umap_embedding_peaks shape => {embedding_peaks.shape} topic_embedding_peaks shape => {peak_mat.shape}"
    )
    return embedding_peaks, peak_mat


def compute_topic_gene_matrix(model, device, selected_topics=None):
    """
    Compute a topic-gene matrix for downstream analysis (e.g., GSEA).

    Steps
    -----
    1. Apply softmax to `model.topic_peak_decoder` => (num_topics, num_peaks).
    2. Min-max normalize each peak across topics.
    3. Multiply by (gene_peak_factor_fixed * gene_peak_factor_learnt).
    4. Standardize each gene across the selected reference topics (population
       variance, epsilon 1e-5), then apply softmax across genes within each topic.
       This reproduces fresh, non-affine BatchNorm; it does not update the
       trained model's BatchNorm parameters or running statistics.
    5. Return relative topic-gene scores, not measured expression or link probabilities.

    Parameters
    ----------
    model : torch.nn.Module
        The scDORI model containing topic_peak_decoder and gene_peak_factor.
    device : torch.device
        The device (CPU or CUDA) used for PyTorch operations.
    selected_topics : sequence of int, optional
        Topics defining the downstream standardization reference and output
        rows, in this order. Peak normalization still uses the full learned
        decoder. None preserves the historical all-topic calculation.

    Returns
    -------
    np.ndarray
        A matrix of shape (num_topics, num_genes) representing topic-gene scores.
    """
    model.eval()
    with torch.no_grad():
        topic_peaks = model.topic_peak_decoder.detach()
        topic_peaks_smx = torch.nn.functional.softmax(topic_peaks, dim=1)
        # shape => (num_topics, num_peaks)

        gene_peak_factor1 = model.gene_peak_factor_fixed.detach()
        gene_peak_factor2 = model.gene_peak_factor_learnt.detach()
        gene_peak_factor = gene_peak_factor1 * gene_peak_factor2

        tmin, _ = torch.min(topic_peaks_smx, dim=0, keepdim=True)
        tmax, _ = torch.max(topic_peaks_smx, dim=0, keepdim=True)
        topic_peaks_norm = (topic_peaks_smx - tmin) / (tmax - tmin + 1e-8)

        preds_gene = torch.mm(topic_peaks_norm, gene_peak_factor.T)
        ids = _topic_indices(selected_topics, model.num_topics)
        preds_gene = preds_gene[ids].detach().cpu()
        # Explicit population standardization matches fresh BatchNorm, while
        # supporting a singleton selection (zero centered importance).
        preds_gene = (preds_gene - preds_gene.mean(dim=0)) / torch.sqrt(preds_gene.var(dim=0, unbiased=False) + 1e-5)
        preds_gene = nn.Softmax(dim=1)(preds_gene.detach().cpu())

        logger.info(f"Done. computing topic gene matrix shape => {preds_gene.numpy().shape}")

    return preds_gene.detach().cpu().numpy()


def _topic_indices(selected_topics, num_topics):
    ids = np.arange(num_topics) if selected_topics is None else np.asarray(list(selected_topics))
    if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu":
        raise ValueError("selected_topics must be a nonempty sequence of integer topic IDs")
    if len(np.unique(ids)) != len(ids) or (ids < 0).any() or (ids >= num_topics).any():
        raise ValueError("selected_topics contains duplicates or out-of-range topic IDs")
    return ids.tolist()


def _validate_permutation_settings(cutoff, permutations):
    if not 0 < cutoff < 1 or not isinstance(permutations, (int, np.integer)) or permutations < 1:
        raise ValueError("Use a cutoff in (0, 1) and a positive integer number of permutations")


def _tested_topics(model, topics):
    from .topic_selection import grn_topic_ids

    return set(grn_topic_ids(model) if topics is None else _topic_indices(topics, model.num_topics))


def _permutation_generator(seed, topic, device):
    """``randperm`` keyword arguments: none for the global stream, else a per-topic generator."""
    if seed is None:
        return {}
    return {"generator": torch.Generator(device=device).manual_seed(int(seed) + int(topic))}


def _stack_with_excluded(slices):
    """Stack per-topic results; untested topics (None) become all-zero slices."""
    template = next(s for s in slices if s is not None)
    return np.stack([np.zeros_like(template) if s is None else s for s in slices])


def compute_atac_grn_activator_with_significance(
    model, device, cutoff_val, outdir, num_permutations, topics=None, seed=None
):
    """
    Compute ATAC-derived TF–gene links for activators, keeping those above a permutation background.

    Uses only the learned peak-gene links and in silico ChIP-seq activator matrices.
    The background comes from permuting TF-binding profiles on peaks.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model containing peak and TF decoders.
    device : torch.device
        The device (CPU or CUDA) for PyTorch operations.
    cutoff_val : float
        Upper-tail fraction (e.g., 0.05 uses the 95th background percentile).
        This is not an FDR-adjusted significance level.
    outdir : str
        Directory to save the computed GRN results.
    num_permutations : int
        The number of permutations for the background.
    topics : sequence of int, optional
        Topics to test. Default: the model's GRN topics (all topics unless a GRN topic
        selection is active). Other topics are returned as all-zero slices.
    seed : int, optional
        If given, topic ``k`` draws its permutations from its own generator seeded with
        ``seed + k``, so results for a topic do not depend on which other topics are tested.
        None keeps the global random stream.

    Returns
    -------
    np.ndarray
        A (num_topics, num_tfs, num_genes) array of background-filtered ATAC-derived activator GRNs.
    """
    _validate_permutation_settings(cutoff_val, num_permutations)
    os.makedirs(outdir, exist_ok=True)
    num_topics = model.num_topics
    tested = _tested_topics(model, topics)
    logger.info(f"Computing background-filtered ATAC-derived TF–gene links for activators. Output => {outdir}")

    with torch.no_grad():
        if model.device != device:
            model = model.to(device)

        topic_peaks = torch.nn.Softmax(dim=1)(model.topic_peak_decoder)
        effective_gene_peak_factor1 = model.gene_peak_factor_fixed
        effective_gene_peak_factor2 = model.gene_peak_factor_learnt
        effective_gene_peak_factor = effective_gene_peak_factor1 * effective_gene_peak_factor2
        insilico_chipseq_embeddings = model.tf_binding_matrix_activator

        grn_atac_significant1 = []
        cutoff_val1 = cutoff_val

        for i in range(num_topics):
            if i not in tested:
                grn_atac_significant1.append(None)
                continue
            logger.info(f"Processing Topic {i + 1}/{num_topics}")
            print(f"Processing Topic {i + 1}/{num_topics}")
            generator = _permutation_generator(seed, i, insilico_chipseq_embeddings.device)

            topic_gene_peak = (topic_peaks[i][:, None].T.clone()) * effective_gene_peak_factor
            topic_gene_tf = torch.matmul(topic_gene_peak, insilico_chipseq_embeddings)
            grn_fg = topic_gene_tf / (effective_gene_peak_factor.sum(dim=1, keepdim=True) + 1e-8)
            grn_fg = grn_fg.T

            # Compute background distribution by shuffling
            grn_bg_topic = []
            for _ in tqdm(range(num_permutations), desc=f"Permutations for Topic {i + 1}"):
                insilico_chipseq_random = insilico_chipseq_embeddings[
                    torch.randperm(
                        insilico_chipseq_embeddings.size(0), device=insilico_chipseq_embeddings.device, **generator
                    )
                ]
                topic_gene_tf_bg = torch.matmul(topic_gene_peak, insilico_chipseq_random)
                grn_bg = topic_gene_tf_bg / (effective_gene_peak_factor.sum(dim=1, keepdim=True) + 1e-8)
                grn_bg_topic.append(grn_bg.T.detach().cpu())

            grn_bg_topic = torch.stack(grn_bg_topic).cpu().numpy()
            cutoff1 = np.percentile(grn_bg_topic, 100 * (1 - cutoff_val1), axis=0)

            grn_fg1 = grn_fg.cpu().numpy()
            significant_grn1 = np.where(grn_fg1 > cutoff1, grn_fg1, 0)
            significant_grn1 = significant_grn1 / (significant_grn1.max() + 1e-15)
            grn_atac_significant1.append(significant_grn1)

        grn_atac_significant1 = _stack_with_excluded(grn_atac_significant1)
        np.save(os.path.join(outdir, f"grn_atac_activator_{cutoff_val}.npy"), grn_atac_significant1)

        logger.info("Completed computing activator ATAC GRNs.")
        return grn_atac_significant1


def compute_atac_grn_repressor_with_significance(
    model, device, cutoff_val, outdir, num_permutations, topics=None, seed=None
):
    """
    Compute ATAC-derived TF–gene links for repressors, keeping those beyond a permutation background.

    Uses the learned peak-gene links and in silico ChIP-seq repressor matrices.
    The background comes from permuting TF-binding profiles on peaks.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model containing peak and TF decoders.
    device : torch.device
        The device (CPU or CUDA) for PyTorch operations.
    cutoff_val : float
        Background percentile cutoff (e.g., 0.05).
    outdir : str
        Directory to save the computed GRN results.

    topics, seed
        As for :func:`compute_atac_grn_activator_with_significance`.

    Returns
    -------
    np.ndarray
        A (num_topics, num_tfs, num_genes) array of background-filtered ATAC-derived repressor GRNs.
    """
    _validate_permutation_settings(cutoff_val, num_permutations)
    os.makedirs(outdir, exist_ok=True)
    num_topics = model.num_topics
    tested = _tested_topics(model, topics)
    logger.info(f"Computing background-filtered ATAC-derived TF–gene links for repressors. Output => {outdir}")
    with torch.no_grad():
        if model.device != device:
            model = model.to(device)

        topic_peaks = torch.nn.Softmax(dim=1)(model.topic_peak_decoder)
        effective_gene_peak_factor1 = model.gene_peak_factor_fixed
        effective_gene_peak_factor2 = model.gene_peak_factor_learnt
        effective_gene_peak_factor = effective_gene_peak_factor1 * effective_gene_peak_factor2
        insilico_chipseq_embeddings_repressor = model.tf_binding_matrix_repressor

        grn_atac_significant1 = []
        cutoff_val1 = cutoff_val

        for i in range(num_topics):
            if i not in tested:
                grn_atac_significant1.append(None)
                continue
            logger.info(f"Processing Topic {i + 1}/{num_topics}")
            print(f"Processing Topic {i + 1}/{num_topics}")
            generator = _permutation_generator(seed, i, insilico_chipseq_embeddings_repressor.device)
            topic_gene_peak_rep = (1 / (topic_peaks[i].clone() + 1e-20))[:, None].T * effective_gene_peak_factor
            topic_gene_tf = torch.matmul(topic_gene_peak_rep, insilico_chipseq_embeddings_repressor)
            grn_fg = topic_gene_tf / (effective_gene_peak_factor.sum(dim=1, keepdim=True) + 1e-8)
            grn_fg = grn_fg.T

            grn_bg_topic = []
            for _ in tqdm(range(num_permutations), desc=f"Permutations for Topic {i + 1}"):
                insilico_chipseq_random = insilico_chipseq_embeddings_repressor[
                    torch.randperm(
                        insilico_chipseq_embeddings_repressor.size(0),
                        device=insilico_chipseq_embeddings_repressor.device,
                        **generator,
                    )
                ]
                topic_gene_tf_bg = torch.matmul(topic_gene_peak_rep, insilico_chipseq_random)
                grn_bg = topic_gene_tf_bg / (effective_gene_peak_factor.sum(dim=1, keepdim=True) + 1e-8)
                grn_bg_topic.append(grn_bg.T.detach().cpu())

            grn_bg_topic = torch.stack(grn_bg_topic).cpu().numpy()
            cutoff1 = np.percentile(grn_bg_topic, 100 * (cutoff_val1), axis=0)
            grn_fg1 = grn_fg.cpu().numpy()
            significant_grn1 = np.where(grn_fg1 < cutoff1, grn_fg1, 0)

            significant_grn1 = significant_grn1 / (significant_grn1.min() + 1e-15)
            grn_atac_significant1.append(significant_grn1)

        grn_atac_significant1 = _stack_with_excluded(grn_atac_significant1)
        np.save(os.path.join(outdir, f"grn_atac_repressor_{cutoff_val}.npy"), grn_atac_significant1)
        logger.info("Completed computing repressor ATAC GRNs.")
        return grn_atac_significant1


def compute_significant_grn(model, device, cutoff_val_activator, cutoff_val_repressor, tf_normalised, outdir):
    """
    Combine background-filtered ATAC-derived and scDORI-learned GRN links into final activator and repressor GRNs.

    Parameters
    ----------
    model : torch.nn.Module
        The scDORI model containing learned TF-gene topic parameters.
    device : torch.device
        CPU or CUDA device for PyTorch operations.
    cutoff_val_activator : float
        Background percentile cutoff used for the activator GRN file.
    cutoff_val_repressor : float
        Background percentile cutoff used for the repressor GRN file.
    tf_normalised : np.ndarray or torch.Tensor
        A (num_topics x num_tfs, 1) or (num_topics x num_tfs) matrix of normalized TF usage.
    outdir : str
        Directory containing the ATAC-based GRN files and to save computed results.

    Returns
    -------
    tuple of np.ndarray
        grn_act : shape (num_topics, num_tfs, num_genes)
            Computed activator GRN array.
        grn_rep : shape (num_topics, num_tfs, num_genes)
            Computed repressor GRN array.

    Raises
    ------
    FileNotFoundError
        If the required ATAC-derived GRN files are missing.
    """
    os.makedirs(outdir, exist_ok=True)
    activator_path = os.path.join(outdir, f"grn_atac_activator_{cutoff_val_activator}.npy")
    repressor_path = os.path.join(outdir, f"grn_atac_repressor_{cutoff_val_repressor}.npy")
    num_topics = model.num_topics
    num_tfs = model.num_tfs
    tf_normalised = torch.as_tensor(tf_normalised).detach().cpu().numpy()
    if tf_normalised.shape not in ((num_topics, num_tfs), (num_topics, num_tfs, 1)):
        raise ValueError("TF gate must have shape (topics, TFs) or (topics, TFs, 1)")
    if not np.isfinite(tf_normalised).all() or (tf_normalised < 0).any():
        raise ValueError("TF gate must be finite and nonnegative")

    # Check if ATAC-derived GRN files exist
    if not os.path.exists(activator_path):
        raise FileNotFoundError(
            f"Activator GRN file not found: {activator_path}\n"
            "Please compute ATAC-based GRNs using the `compute_atac_grn_activator_with_significance` function first."
        )
    if not os.path.exists(repressor_path):
        raise FileNotFoundError(
            f"Repressor GRN file not found: {repressor_path}\n"
            "Please compute ATAC-based GRNs using the `compute_atac_grn_repressor_with_significance` function first."
        )

    logger.info("Loading ATAC-derived GRNs...")
    grn_atac_activator = np.load(activator_path)
    grn_atac_repressor = np.load(repressor_path)

    expected_shape = (num_topics, num_tfs, model.num_genes)
    if grn_atac_activator.shape != expected_shape or grn_atac_repressor.shape != expected_shape:
        raise ValueError("ATAC GRNs do not match this model's topic/TF/gene dimensions")
    if not np.isfinite(grn_atac_activator).all() or not np.isfinite(grn_atac_repressor).all():
        raise ValueError("ATAC GRNs must be finite")
    logger.info("Computing combined GRNs...")

    grn_rep = torch.tensor(grn_atac_repressor) * (
        -1 * torch.nn.functional.relu(model.tf_gene_topic_repressor_grn).detach().cpu()
    )
    grn_act = torch.tensor(grn_atac_activator) * (
        torch.nn.functional.relu(model.tf_gene_topic_activator_grn).detach().cpu()
    )

    grn_tot = grn_rep.clone() + grn_act.clone()
    grn_tot = grn_tot.numpy()
    grn_rep = grn_tot.copy()
    grn_rep[grn_rep > 0] = 0
    grn_rep = torch.from_numpy(tf_normalised).reshape((num_topics, num_tfs, 1)) * grn_rep

    grn_act = grn_tot.copy()
    grn_act[grn_act < 0] = 0
    grn_act = torch.from_numpy(tf_normalised).reshape((num_topics, num_tfs, 1)) * grn_act

    logger.info("Saving computed GRNs...")
    np.save(os.path.join(outdir, f"grn_activator__{cutoff_val_activator}.npy"), grn_act.numpy())
    np.save(os.path.join(outdir, f"grn_repressor__{cutoff_val_repressor}.npy"), grn_rep.numpy())

    logger.info("GRN computation completed successfully.")
    return grn_act.numpy(), grn_rep.numpy()


def save_regulons(grn_matrix, tf_names, gene_names, num_topics, output_dir, mode="activator", verbose=True):
    """
    Save regulons (TF-gene links across topics) for each TF based on a given GRN matrix.

    Parameters
    ----------
    grn_matrix : np.ndarray
        A GRN matrix of shape (num_topics, num_tfs, num_genes).
    tf_names : list of str
        List of transcription factor names, length = num_tfs.
    gene_names : list of str
        List of gene names, length = num_genes.
    num_topics : int
        Number of topics in the GRN matrix.
    output_dir : str
        Directory where the regulon files will be saved.
    mode : str, optional
        "activator" or "repressor", used to name the output subdirectory/files.
    verbose : bool, optional
        Print each saved path. False prints a single completion summary instead.

    Returns
    -------
    None
        Saves individual TSV files for each TF in `output_dir` of shape (num_topics, num_genes), where non-zero values represent a link.
    """
    output_path = os.path.join(output_dir, f"regulons_tf/{mode}")
    os.makedirs(output_path, exist_ok=True)

    for i, tf_name in enumerate(tf_names):
        regulon = pd.DataFrame(grn_matrix[:, i, :], index=[f"Topic_{k}" for k in range(num_topics)], columns=gene_names)
        regulon.to_csv(os.path.join(output_path, f"{tf_name}_{mode}.tsv"), sep="\t")
        if verbose:
            print(f"Saved {mode} regulon for TF: {tf_name} in {output_path}")
    if not verbose:
        print(f"Saved {len(tf_names)} {mode} regulon files in {output_path}")


def get_top_activators_per_topic(
    grn_final,  # => shape (num_topics, num_tfs, num_genes)
    tf_names,  # list of TF names => length = num_tfs
    latent_all_torch,  # shape (num_cells, num_topics)
    selected_topics=None,
    top_k=10,
    clamp_value=1e-8,
    zscore=True,
    figsize=(25, 10),
    out_fig=None,
    display_topics=None,
    plot=True,
):
    """
    Identify and plot top activator transcription factors per topic (Topic regulators, TRs).

    Parameters
    ----------
    grn_final : np.ndarray or torch.Tensor
        An array of shape (num_topics, num_tfs, num_genes), representing an activator GRN.
    tf_names : list of str
        List of TF names, length = num_tfs.
    latent_all_torch : np.ndarray or torch.Tensor
        scDORI latent topic activity of shape (num_cells, num_topics). Not always used, but can be referenced.
    selected_topics : list of int, optional
        Which topics to analyze. If None, all topics are used.
    top_k : int, optional
        Number of top TFs to select per topic. Default is 10.
    clamp_value : float, optional
        Small cutoff to avoid division by zero. Default is 1e-8.
    zscore : bool, optional
        If True, apply z-score normalization across topics in the final matrix. Default is True.
    figsize : tuple, optional
        Size for the Seaborn clustermap. Default is (25, 10).
    out_fig : str or Path, optional
        If provided, the figure is saved to this path; otherwise it is shown.

    display_topics : list of int, optional
        Subset to draw after scaling over selected_topics. Returned scores and
        selected TFs keep their original reference. None displays all rows.
    plot : bool, optional
        Draw the regulator heatmap. False returns the same scores without a figure.

    Returns
    -------
    tuple
        df_topic_grn : pd.DataFrame
            The final DataFrame of shape (#topics, #TF).
        selected_tf : list of str
            A sorted list of TFs used in the final clustermap.
    """
    logger.info("=== Plotting top activator regulators per topic ===")

    num_topics = grn_final.shape[0]
    selected_topics = _topic_indices(selected_topics, num_topics)

    topic_tf_grn = []
    topic_tf_grn_norm = []

    grn_final_np = grn_final
    if isinstance(grn_final, torch.Tensor):
        grn_final_np = grn_final.detach().cpu().numpy()

    # sum across genes => shape => (num_tfs,)
    for i in range(num_topics):
        grn_topic = grn_final_np[i].sum(axis=1)
        topic_tf_grn.append(grn_topic)
        total_activity = grn_topic.sum() + clamp_value
        topic_tf_grn_norm.append(grn_topic / total_activity)

    topic_tf_grn_norm_act = np.array(topic_tf_grn_norm)

    df_topic_grn = pd.DataFrame(
        topic_tf_grn_norm_act[selected_topics], columns=tf_names, index=[f"Topic_{k}" for k in selected_topics]
    )

    if zscore:
        df_topic_grn = df_topic_grn.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8), axis=0).fillna(0)

    selected_tf = set()
    for _i, row_name in enumerate(df_topic_grn.index):
        row = df_topic_grn.loc[row_name].sort_values(ascending=False)
        top_tfs = row.head(top_k).index.values
        selected_tf.update(top_tfs)
    selected_tf = list(selected_tf)
    selected_tf = sorted(selected_tf)

    df_plot = df_topic_grn[selected_tf]

    if not plot:
        return df_topic_grn, selected_tf

    sns.set_style("darkgrid")
    plt.rcParams["figure.facecolor"] = "white"
    sns.set(font_scale=1.2)

    display_names = df_plot.index if display_topics is None else [
        f"Topic_{topic}" for topic in _topic_indices(display_topics, num_topics)
    ]
    if any(topic not in df_plot.index for topic in display_names):
        raise ValueError("display_topics must be included in selected_topics")
    plot_tfs = selected_tf if display_topics is None else sorted({
        tf
        for name in display_names
        for tf in df_topic_grn.loc[name].sort_values(ascending=False).head(top_k).index
    })
    g = sns.clustermap(
        df_plot.loc[display_names, plot_tfs],
        row_cluster=False,
        col_cluster=len(plot_tfs) > 1,
        linewidths=1.5,
        dendrogram_ratio=0.1,
        cmap="Spectral",
        vmin=-4,
        vmax=4,
        figsize=figsize,
        annot_kws={"size": 20},
    )
    if out_fig:
        g.figure.savefig(out_fig, dpi=300)
    plt.show()

    logger.info("=== Done plotting top regulators per topic ===")
    return df_topic_grn, selected_tf


def get_top_repressor_per_topic(
    grn_final,  # => shape (num_topics, num_tfs, num_genes)
    tf_names,  # list of TF names => length = num_tfs
    latent_all_torch,  # shape (num_cells, num_topics)
    selected_topics=None,
    top_k=5,
    clamp_value=1e-8,
    zscore=True,
    figsize=(25, 10),
    out_fig=None,
    display_topics=None,
    plot=True,
):
    """
    Identify and plot top repressor transcription factors per topic.

    Parameters
    ----------
    grn_final : np.ndarray or torch.Tensor
        An array of shape (num_topics, num_tfs, num_genes), representing a repressor GRN.
    tf_names : list of str
        List of TF names, length = num_tfs.
    latent_all_torch : np.ndarray or torch.Tensor
        scDORI latent topic activity of shape (num_cells, num_topics).
    selected_topics : list of int, optional
        Which topics to analyze. If None, all topics are used.
    top_k : int, optional
        Number of top TFs to select per topic. Default is 5.
    clamp_value : float, optional
        Small cutoff to avoid division by zero. Default is 1e-8.
    zscore : bool, optional
        If True, apply z-score normalization across topics in the final matrix. Default is True.
    figsize : tuple, optional
        Size for the Seaborn clustermap. Default is (25, 10).
    out_fig : str or Path, optional
        If provided, the figure is saved to this path; otherwise it is shown.

    display_topics : list of int, optional
        Subset to draw after scaling over selected_topics. Returned scores and
        selected TFs keep their original reference. None displays all rows.
    plot : bool, optional
        Draw the regulator heatmap. False returns the same scores without a figure.

    Returns
    -------
    tuple
        df_plot : pd.DataFrame
            The final DataFrame of shape (#topics, #TF).
        selected_tf : list of str
            A sorted list of TFs used in the final clustermap.
    """
    logger.info("=== Plotting top repressor regulators per topic ===")

    num_topics = grn_final.shape[0]
    selected_topics = _topic_indices(selected_topics, num_topics)

    topic_tf_grn = []
    topic_tf_grn_norm = []

    grn_final_np = grn_final
    if isinstance(grn_final, torch.Tensor):
        grn_final_np = grn_final.detach().cpu().numpy()

    for i in range(num_topics):
        grn_topic = np.abs(grn_final_np[i]).sum(axis=1)
        topic_tf_grn.append(grn_topic)
        total_activity = grn_topic.sum() + clamp_value
        topic_tf_grn_norm.append(grn_topic / total_activity)

    topic_tf_grn_norm_rep = np.array(topic_tf_grn_norm)

    df_topic_grn = pd.DataFrame(
        topic_tf_grn_norm_rep[selected_topics], columns=tf_names, index=[f"Topic_{k}" for k in selected_topics]
    )

    if zscore:
        df_topic_grn = df_topic_grn.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8), axis=0).fillna(0)

    selected_tf = set()
    for _i, row_name in enumerate(df_topic_grn.index):
        row = df_topic_grn.loc[row_name].sort_values(ascending=False)
        top_tfs = row.head(top_k).index.values
        selected_tf.update(top_tfs)
    selected_tf = list(selected_tf)
    selected_tf = sorted(selected_tf)

    df_plot = df_topic_grn[selected_tf]

    if not plot:
        return df_plot, selected_tf

    sns.set_style("darkgrid")
    plt.rcParams["figure.facecolor"] = "white"
    sns.set(font_scale=1.2)

    display_names = df_plot.index if display_topics is None else [
        f"Topic_{topic}" for topic in _topic_indices(display_topics, num_topics)
    ]
    if any(topic not in df_plot.index for topic in display_names):
        raise ValueError("display_topics must be included in selected_topics")
    plot_tfs = selected_tf if display_topics is None else sorted({
        tf
        for name in display_names
        for tf in df_topic_grn.loc[name].sort_values(ascending=False).head(top_k).index
    })
    g = sns.clustermap(
        df_plot.loc[display_names, plot_tfs],
        row_cluster=False,
        col_cluster=len(plot_tfs) > 1,
        linewidths=1.5,
        dendrogram_ratio=0.1,
        cmap="Spectral",
        vmin=-4,
        vmax=4,
        figsize=figsize,
        annot_kws={"size": 20},
    )
    if out_fig:
        g.figure.savefig(out_fig, dpi=300)
    plt.show()

    logger.info("=== Done plotting top repressor regulators per topic ===")
    return df_plot, selected_tf


def compute_activator_tf_activity_per_cell(
    grn_final,  # => shape (num_topics, num_tfs, num_genes)
    tf_names,  # list of TF names => length = num_tfs
    latent_all_torch,  # shape (num_cells, num_topics)
    selected_topics=None,
    clamp_value=1e-8,
    zscore=True,
):
    """
    Compute per-cell activity of activator TFs.

    Parameters
    ----------
    grn_final : np.ndarray or torch.Tensor
        Activator GRN of shape (num_topics, num_tfs, num_genes).
    tf_names : list of str
        List of TF names, length = num_tfs.
    latent_all_torch : np.ndarray or torch.Tensor
        scDORI latent topic activity of shape (num_cells, num_topics).
    selected_topics : list of int, optional
        Which topics to analyze. If None, all topics are used.
    clamp_value : float, optional
        Small constant to avoid division by zero. Default is 1e-8.
    zscore : bool, optional
        If True, apply z-score normalization across cells in the final matrix. Default is True.

    Returns
    -------
    np.ndarray
        A (num_cells, num_tfs) array of TF activity values.
    """
    logger.info("=== Computing TF activity per cell ===")

    num_topics = grn_final.shape[0]
    selected_topics = _topic_indices(selected_topics, num_topics)

    topic_tf_grn = []
    topic_tf_grn_norm = []

    grn_final_np = grn_final
    if isinstance(grn_final, torch.Tensor):
        grn_final_np = grn_final.detach().cpu().numpy()

    for i in range(num_topics):
        grn_topic = grn_final_np[i].sum(axis=1)
        topic_tf_grn.append(grn_topic)
        total_activity = grn_topic.sum() + clamp_value
        topic_tf_grn_norm.append(grn_topic / total_activity)

    topic_tf_grn_norm_act = np.array(topic_tf_grn_norm)

    theta = (
        latent_all_torch.detach().cpu().numpy()
        if isinstance(latent_all_torch, torch.Tensor)
        else np.asarray(latent_all_torch)
    )
    cell_tf_act = theta[:, selected_topics] @ topic_tf_grn_norm_act[selected_topics]

    if zscore:
        cell_tf_act = np.divide(
            cell_tf_act - cell_tf_act.mean(axis=0),
            cell_tf_act.std(axis=0),
            out=np.zeros_like(cell_tf_act),
            where=cell_tf_act.std(axis=0) > 0,
        )

    return cell_tf_act


def compute_repressor_tf_activity_per_cell(
    grn_final,  # => shape (num_topics, num_tfs, num_genes)
    tf_names,  # list of TF names => length = num_tfs
    latent_all_torch,  # shape (num_cells, num_topics)
    selected_topics=None,
    clamp_value=1e-8,
    zscore=True,
):
    """
    Compute per-cell activity of repressor TFs.

    Parameters
    ----------
    grn_final : np.ndarray or torch.Tensor
        Repressor GRN of shape (num_topics, num_tfs, num_genes).
    tf_names : list of str
        List of TF names, length = num_tfs.
    latent_all_torch : np.ndarray or torch.Tensor
        scDORI latent topic activity of shape (num_cells, num_topics).
    selected_topics : list of int, optional
        Which topics to analyze. If None, all topics are used.
    clamp_value : float, optional
        Small constant to avoid division by zero. Default is 1e-8.
    zscore : bool, optional
        If True, apply z-score normalization across cells in the final matrix. Default is True.

    Returns
    -------
    np.ndarray
        A (num_cells, num_tfs) array of TF activity values.
    """
    logger.info("=== Computing TF activity per cell ===")

    num_topics = grn_final.shape[0]
    selected_topics = _topic_indices(selected_topics, num_topics)

    topic_tf_grn = []
    topic_tf_grn_norm = []

    grn_final_np = grn_final
    if isinstance(grn_final, torch.Tensor):
        grn_final_np = grn_final.detach().cpu().numpy()

    for i in range(num_topics):
        grn_topic = np.abs(grn_final_np[i]).sum(axis=1)
        topic_tf_grn.append(grn_topic)
        total_activity = grn_topic.sum() + clamp_value
        topic_tf_grn_norm.append(grn_topic / total_activity)

    topic_tf_grn_norm_rep = np.array(topic_tf_grn_norm)

    theta = (
        latent_all_torch.detach().cpu().numpy()
        if isinstance(latent_all_torch, torch.Tensor)
        else np.asarray(latent_all_torch)
    )
    cell_tf_rep = theta[:, selected_topics] @ topic_tf_grn_norm_rep[selected_topics]

    if zscore:
        cell_tf_rep = np.divide(
            cell_tf_rep - cell_tf_rep.mean(axis=0),
            cell_tf_rep.std(axis=0),
            out=np.zeros_like(cell_tf_rep),
            where=cell_tf_rep.std(axis=0) > 0,
        )

    return cell_tf_rep
