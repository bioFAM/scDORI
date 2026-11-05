"""Load the expression multipliers belonging to an existing phase-2 fit."""

import hashlib
import json
from pathlib import Path

import numpy as np


def load_downstream_model(settings, device=None):
    """Restore the fitted model and its inputs for the full analysis notebook.

    Keep current data/output paths so a project can move between machines. Restore
    model and expression settings from the checkpoint's training snapshot, and
    verify cell, feature and batch order before using its saved expression values.
    Old fits may opt into recomputation with their original training split.
    """
    import copy
    from types import SimpleNamespace
    import torch
    from sklearn.preprocessing import OneHotEncoder
    from torch.utils.data import DataLoader, TensorDataset
    from .data_io import load_scdori_inputs
    from .downstream import load_best_model
    from .models import scDoRI
    from .topic_selection import load_grn_topics
    from .utils import set_seed
    from scdori.configuration import check_notebook_inputs, check_training_input_order, training_input_order

    settings = copy.deepcopy(settings)
    config = settings.training
    checkpoint = Path(config.weights_folder_grn) / "best_scdori_best_eval.pth"
    snapshot = checkpoint.parent / "notebook_config.json"
    if snapshot.is_file():
        saved = json.loads(snapshot.read_text()).get("training", {})
        for name in ("num_topics", "dim_encoder1", "dim_encoder2", "batch_col", "random_seed",
                     "tf_expression_mode", "tf_expression_scaling", "tf_expression_clamp",
                     "tf_expression_min_detection", "tf_expression_keep_tfs", "cells_per_topic",
                     "batch_size_cell_prediction", "atac_count_mode", "freeze_phase1_in_grn",
                     "update_encoder_in_grn", "update_peak_gene_in_grn", "update_topic_peak_in_grn",
                     "update_topic_tf_in_grn", "grn_topic_selection"):
            if name in saved:
                setattr(config, name, saved[name])
        if "tf_expression_min_detection" not in saved:  # fitted before this rule existed
            config.tf_expression_min_detection = 0.0
    check_notebook_inputs(settings, "downstream")
    set_seed(config.random_seed)
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    rna, atac, dist, act, rep = load_scdori_inputs(config)
    if "celltype" not in rna.obs:
        raise ValueError("Add cell-type or cluster labels to RNA obs['celltype'] before downstream analysis")
    rna.layers["counts"], atac.layers["counts"] = rna.X.copy(), atac.X.copy()
    if "num_cells" not in rna.obs:
        rna.obs["num_cells"] = 1
    num_cells = rna.obs["num_cells"].to_numpy().reshape(-1, 1)
    if not np.isfinite(num_cells).all() or (num_cells <= 0).any():
        raise ValueError("num_cells must contain positive finite cell counts")
    rna.var["index_int"] = range(rna.n_vars)
    tf_indices = np.flatnonzero(rna.var["gene_type"].to_numpy() == "TF")
    batch = rna.obs[config.batch_col].to_numpy()
    if rna.obs[config.batch_col].isna().any() or not np.array_equal(batch, atac.obs[config.batch_col].to_numpy()):
        raise ValueError("RNA and ATAC sample labels must match and contain no missing values")
    rna.obs["batch"], atac.obs["batch"] = batch, batch
    encoder = OneHotEncoder(handle_unknown="ignore").fit(batch.reshape(-1, 1))
    onehot = encoder.transform(batch.reshape(-1, 1)).toarray()
    check_training_input_order(checkpoint, training_input_order(rna, atac, tf_indices, encoder.categories_[0]))
    model = scDoRI(device=device, num_genes=rna.n_vars, num_peaks=atac.n_vars,
                   num_tfs=act.shape[1], num_topics=config.num_topics, num_batches=onehot.shape[1],
                   dim_encoder1=config.dim_encoder1, dim_encoder2=config.dim_encoder2).to(device)
    model = load_best_model(model, checkpoint, device)
    topics = load_grn_topics(model, checkpoint)
    excluded = topics["excluded_topics"] if topics else []
    try:
        values, record = load_saved_tf_expression(checkpoint.parent, model.num_topics, model.num_tfs,
            filename=settings.downstream.tf_expression_file, excluded_topics=excluded)
        expression = torch.as_tensor(values, device=device)
    except FileNotFoundError:
        if not settings.downstream.allow_tf_expression_recompute or settings.downstream.tf_expression_file is not None:
            raise
        from .train_grn import get_tf_expression
        with np.load(checkpoint.parent / "training_split.npz", allow_pickle=False) as split:
            ids = split["train"]
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids) or len(np.unique(ids)) != len(ids)
                or ids.min() < 0 or ids.max() >= rna.n_obs):
            raise ValueError("The saved training split contains invalid cell indices")
        loader = DataLoader(TensorDataset(torch.as_tensor(ids, dtype=torch.long)),
                            batch_size=config.batch_size_cell_prediction, shuffle=False)
        expression = get_tf_expression(config.tf_expression_mode, model, device, loader,
                                       rna, atac, num_cells, tf_indices, onehot, config)
        record = {"source": "recomputed_from_original_training_split", "recomputed": True,
                  "split": str(checkpoint.parent / "training_split.npz"),
                  "tf_expression_scaling": config.tf_expression_scaling,
                  "tf_expression_clamp": config.tf_expression_clamp,
                  "tf_expression_min_detection": config.tf_expression_min_detection,
                  "tf_expression_keep_tfs": list(config.tf_expression_keep_tfs)}
    output = Path(settings.downstream.grn_output)
    output.mkdir(parents=True, exist_ok=True)
    seed = settings.downstream.permutation_seed
    (output / "downstream_run.json").write_text(json.dumps({
        "checkpoint": str(checkpoint.resolve()), "tf_expression": record,
        "phase2_excluded_topics": list(map(int, excluded)),
        "activator_alpha": settings.downstream.activator_alpha,
        "repressor_alpha": settings.downstream.repressor_alpha,
        "num_permutations": config.num_permutations, "permutation_seed": seed,
        "rng_policy": "historical global RNG" if seed is None else "per-topic generators; repressor offset 100000",
    }, indent=2) + "\n")
    fixed = dist.clone()
    fixed[fixed > 0] = 1
    return SimpleNamespace(settings=settings, config=config, model=model, device=device, rna=rna, atac=atac,
        gene_peak_dist=dist, gene_peak_fixed=fixed, insilico_act=act, insilico_rep=rep, tf_indices=tf_indices,
        num_cells=num_cells, onehot_batch=onehot, tf_expression=expression, tf_expression_record=record,
        excluded_topics=excluded, checkpoint=checkpoint)


def load_saved_tf_expression(folder, num_topics, num_tfs, filename=None, excluded_topics=()):
    """Load saved TF-expression values with shape, range and excluded-topic checks.

    An explicit filename takes precedence. Automatic lookup accepts the notebook
    name and one earlier name in the checkpoint directory; differing copies are
    rejected. Shape checks cannot verify feature identities: retain the matching
    input-order record and original training data. No values are recomputed.
    """
    candidates = (
        [Path(filename)]
        if filename is not None
        else [Path(folder) / "topic_tf_expression.npy", Path(folder) / "gate_used.npy"]
    )
    paths = [p for p in candidates if p.is_file()]
    if not paths:
        raise FileNotFoundError(
            "No saved phase-2 TF-expression multipliers found. Set downstream.tf_expression_file "
            "to the matching topic_tf_expression.npy or gate_used.npy. "
            "For older fits, explicit recomputation requires the original training split."
        )
    arrays = []
    for path in paths:
        array = np.load(path, allow_pickle=False)
        if array.shape == (num_topics, 1, num_tfs):
            array = array[:, 0, :]
        if array.shape != (num_topics, num_tfs):
            raise ValueError(f"{path}: expected {(num_topics, num_tfs)}, found {array.shape}")
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all() or (array < 0).any() or (array > 1).any():
            raise ValueError(f"{path}: TF-expression multipliers must be finite numbers in [0, 1]")
        excluded = np.asarray(list(excluded_topics))
        if excluded.size and (excluded.ndim != 1 or excluded.dtype.kind not in "iu"):
            raise ValueError("Excluded topic IDs must be integers")
        excluded = excluded.astype(int)
        if excluded.size and ((excluded < 0).any() or (excluded >= num_topics).any()):
            raise ValueError("Excluded topic IDs are outside the fitted model")
        if excluded.size and np.any(array[excluded] != 0):
            raise ValueError(f"{path}: excluded phase-2 topics have nonzero expression multipliers")
        arrays.append(array)
    if any(not np.array_equal(arrays[0], array) for array in arrays[1:]):
        raise ValueError("Conflicting saved TF-expression files; specify downstream.tf_expression_file explicitly")
    record = {
        "source": "saved",
        "file": str(paths[0].resolve()),
        "sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        "shape": list(arrays[0].shape),
        "recomputed": False,
        "identity_check": "shape only; verify TF order against the training input-order record",
    }
    summary = paths[0].parent / "heldout_summary.json"
    snapshot = paths[0].parent / "notebook_config.json"
    if summary.is_file():
        record["recorded_expression_settings"] = json.loads(summary.read_text()).get("gate", {})
    elif snapshot.is_file():
        cfg = json.loads(snapshot.read_text()).get("training", {})
        record["recorded_expression_settings"] = {
            key: cfg.get(key) for key in ("tf_expression_mode", "tf_expression_scaling", "tf_expression_clamp",
                                          "tf_expression_min_detection", "tf_expression_keep_tfs")
        }
    return arrays[0], record
