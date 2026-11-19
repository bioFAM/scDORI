"""Selection of the topics modelled in the GRN phase (phase 2) and permutations.

The default ``grn_topic_selection="live"`` excludes collapsed topics, and "all" retains every topic.
"shuffle_null" keeps a topic only if the trained encoder gives it more weight in real cells than in cells whose
gene and peak values were shuffled within each cell (see :func:`shuffle_null_topics`). With a selection, only the
selected topics enter the phase-2 forward pass, GRN factor caches and permutation tests.
Excluded topics keep their original IDs, and their GRN weights, TF gate and ATAC GRNs are zero, so every output
keeps shape K.

The phase-1 model is untouched: theta is still a softmax over all K topics. The
term dropped from the phase-2 prediction is ``theta[:, excluded] @ C[excluded]``. It is negligible for collapsed
topics ("live"); a topic dropped by "shuffle_null" can carry more weight, which the record reports
(``max_excluded_theta_mass``).
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch

from .config import train_default

logger = logging.getLogger(__name__)

GRN_TOPICS_FILE = "grn_topics.json"
GRN_TOPICS_FORMAT = 1


def set_grn_topics(model, topics):
    """Restrict the GRN phase to ``topics`` (original IDs); ``None`` restores all topics."""
    if topics is None:
        model.grn_topics = None
        return model
    values = np.asarray(list(topics))
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError("GRN topics must be integer IDs")
    ids = np.unique(values.astype(np.int64))
    if len(ids) != len(values) or not len(ids) or ids[0] < 0 or ids[-1] >= model.num_topics:
        raise ValueError("GRN topics must be nonempty, unique and within range")
    model.grn_topics = None if len(ids) == model.num_topics else torch.as_tensor(ids)
    return model


def grn_topic_ids(model):
    """Topic IDs used in the GRN phase (all topics when no selection is active)."""
    selected = getattr(model, "grn_topics", None)
    return list(range(model.num_topics)) if selected is None else selected.tolist()


def excluded_grn_topics(model):
    """Topic IDs excluded from the GRN phase."""
    return sorted(set(range(model.num_topics)) - set(grn_topic_ids(model)))


@torch.no_grad()
def zero_excluded_grn(model):
    """Set the GRN weights of excluded topics to zero; they then receive no gradient or L1 change."""
    excluded = excluded_grn_topics(model)
    if excluded:
        model.tf_gene_topic_activator_grn[excluded] = 0
        model.tf_gene_topic_repressor_grn[excluded] = 0
    return excluded


def collapsed_topics(theta, max_theta=0.05, max_mean_fraction=0.1):
    """Collapsed topics: theta below ``max_theta`` in every cell and mean below ``max_mean_fraction / K``.

    Both conditions must hold. The defaults match ``TrainConfig.grn_dead_topic_max_theta`` and
    ``TrainConfig.grn_dead_topic_max_mean_fraction``.
    """
    theta = np.asarray(theta)
    K = theta.shape[1]
    return np.flatnonzero((theta.max(0) < max_theta) & (theta.mean(0) < max_mean_fraction / K))


def _dead_topic_thresholds(config):
    """(max theta, max mean fraction) for collapsed topics; missing settings use the current defaults."""
    return tuple(
        getattr(config, name, train_default(name))
        for name in ("grn_dead_topic_max_theta", "grn_dead_topic_max_mean_fraction")
    )


def shuffle_within_rows(matrix, rng):
    """Copy of ``matrix`` (cells x features) with each row's stored values moved to random feature positions.

    Each cell keeps its values and depth; which features carry them is random (positions drawn with
    ``rng.choice`` without replacement). Dense input is converted to CSR first.
    """
    import scipy.sparse as sp

    shuffled = sp.csr_matrix(matrix, copy=True)
    width = shuffled.shape[1]
    for row in range(shuffled.shape[0]):
        start, end = shuffled.indptr[row], shuffled.indptr[row + 1]
        shuffled.indices[start:end] = rng.choice(width, end - start, replace=False)
    shuffled.sort_indices()
    return shuffled


def shuffle_null_topics(model, device, rna, atac, num_cells, tf_indices, batches, cell_ids, theta, config):
    """Per-topic test of observed topic weights against shuffled cells, for ``grn_topic_selection="shuffle_null"``.

    ``theta`` holds the trained encoder's topic weights (eval mode) for the training cells ``cell_ids``. For each
    topic, the ``grn_null_top_cells`` cells with the highest weight are taken; their gene values are shuffled
    across genes and their peak values across peaks within each cell (:func:`shuffle_within_rows`), and the
    encoder is applied again. This is repeated ``grn_null_shuffles`` times with seeds ``grn_null_seed + k``. The
    topic's cutoff is the ``grn_null_quantile`` quantile of its weights in these shuffled cells; the topic is kept
    if the ``grn_null_observed_quantile`` quantile of its observed weights in the training cells is above it.

    Returns a dict with per-topic ``cutoff``, ``observed`` and ``keep`` arrays and the settings used.
    """
    import anndata as ad
    from torch.utils.data import DataLoader, TensorDataset

    from .dataloader import read_count_rows
    from .evaluation import get_latent_topics

    settings = {name: getattr(config, name, train_default(name)) for name in (
        "grn_null_top_cells", "grn_null_shuffles", "grn_null_quantile", "grn_null_observed_quantile", "grn_null_seed")}
    theta = np.asarray(theta)
    cell_ids = np.asarray(cell_ids)
    if theta.ndim != 2 or theta.shape[0] != len(cell_ids) or not len(cell_ids):
        raise ValueError("theta must be (training cells x topics) and aligned with cell_ids")
    top = min(int(settings["grn_null_top_cells"]), len(cell_ids))
    order = np.argsort(-theta, axis=0, kind="stable")[:top]  # top x topics: row positions in theta
    rows = np.unique(order)
    cells = cell_ids[rows]
    rna_rows, atac_rows = read_count_rows(rna, cells), read_count_rows(atac, cells)
    uns = {key: atac.uns[key] for key in ("scdori_atac_count_mode",) if key in atac.uns}
    position = np.searchsorted(rows, order)  # top x topics: positions in the shuffled block
    batch_size = int(getattr(config, "batch_size_cell_prediction", train_default("batch_size_cell_prediction")))
    null = []
    for k in range(int(settings["grn_null_shuffles"])):
        rng = np.random.default_rng(int(settings["grn_null_seed"]) + k)
        rna_k = ad.AnnData(shuffle_within_rows(rna_rows, rng).astype(np.float32))
        atac_k = ad.AnnData(shuffle_within_rows(atac_rows, rng).astype(np.float32), uns=uns)
        loader = DataLoader(TensorDataset(torch.arange(len(cells))), batch_size=batch_size, shuffle=False)
        null.append(get_latent_topics(model, device, loader, rna_k, atac_k, np.asarray(num_cells)[cells],
                                      tf_indices, np.asarray(batches)[cells]))
    null = np.stack(null)  # shuffles x shuffled cells x topics
    topics = range(theta.shape[1])
    cutoff = np.array([np.quantile(null[:, position[:, t], t], settings["grn_null_quantile"]) for t in topics])
    observed = np.quantile(theta, settings["grn_null_observed_quantile"], axis=0)
    return {"cutoff": cutoff, "observed": observed, "keep": observed > cutoff, "theta": theta,
            "shuffled_cells": int(len(cells)), **settings}


def max_excluded_theta_mass(theta, excluded):
    """Largest topic mass any cell places on excluded topics (what the GRN phase drops)."""
    excluded = list(excluded)
    return float(np.asarray(theta)[:, excluded].sum(1).max()) if excluded else 0.0


def resolve_grn_topics(model, config, theta=None, null=None):
    """Topics for the GRN phase from the config; ``None`` means all topics.

    ``theta`` (training cells x K) is needed for "live" without ``grn_live_topics_file``; ``null`` (the result
    of :func:`shuffle_null_topics`) is needed for "shuffle_null".
    """
    from .diagnostics import read_live_topics

    mode = getattr(config, "grn_topic_selection", train_default("grn_topic_selection"))
    if mode == "all":
        return None
    if mode == "shuffle_null":
        if null is None:
            raise ValueError('grn_topic_selection="shuffle_null" needs the shuffled-cell test (run it in phase 2)')
        live = np.flatnonzero(null["keep"])
        if not len(live):
            raise ValueError("No topic has more weight in real cells than in shuffled cells; check phase 1")
        return live
    if mode == "explicit":
        topics = list(getattr(config, "grn_topics", []))
        if not topics:
            raise ValueError('grn_topic_selection="explicit" requires grn_topics')
        return np.asarray(topics)
    if mode != "live":
        raise ValueError(f"Unknown grn_topic_selection: {mode!r}; use 'shuffle_null', 'live', 'all' or 'explicit'")
    path = getattr(config, "grn_live_topics_file", None)
    if path:
        checkpoint = getattr(config, "phase1_checkpoint", None)
        if not checkpoint:
            raise ValueError("Loading a live-topic file requires phase1_checkpoint for provenance validation")
        topics = read_live_topics(path, num_topics=model.num_topics, checkpoint=checkpoint)
        if theta is None:
            raise ValueError("Loading a live-topic file requires validation against current training-cell theta")
        dead = collapsed_topics(theta, *_dead_topic_thresholds(config))
        excluded = np.setdiff1d(np.arange(model.num_topics), topics)
        if not np.isin(excluded, dead).all():
            raise ValueError("Live-topic file excludes topics that are not collapsed on current training cells")
        return topics
    if theta is None:
        raise ValueError('grn_topic_selection="live" needs grn_live_topics_file or topic activations')
    dead = collapsed_topics(theta, *_dead_topic_thresholds(config))
    live = np.setdiff1d(np.arange(model.num_topics), dead)
    if not len(live):
        raise ValueError("Every topic is collapsed; phase 1 did not learn usable topics")
    return live


def apply_grn_topic_selection(model, config, theta_fn=None, null_fn=None):
    """Resolve, set and log the GRN topic selection; zero excluded GRN weights.

    ``theta_fn`` returns training-cell theta and is only called when needed; ``null_fn`` returns the
    :func:`shuffle_null_topics` result and is needed for "shuffle_null". Returns a JSON-serializable record
    (also written by :func:`write_grn_topics`).
    """
    mode = getattr(config, "grn_topic_selection", train_default("grn_topic_selection"))
    flags = ("update_encoder_in_grn", "update_peak_gene_in_grn", "update_topic_peak_in_grn", "update_topic_tf_in_grn")
    if mode != "all" and (
        not getattr(config, "freeze_phase1_in_grn", train_default("freeze_phase1_in_grn"))
        or any(getattr(config, flag, train_default(flag)) for flag in flags)
    ):
        raise ValueError("Topic selection requires fully frozen phase 1, including encoder and BatchNorm state")
    needs_theta = mode == "live" and not getattr(config, "grn_live_topics_file", None)
    theta = null = None
    device = next(model.parameters()).device
    if mode == "shuffle_null" and null_fn is not None:
        with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
            null = null_fn()
        theta = null["theta"]
    elif mode not in ("all", "shuffle_null") and theta_fn is not None:
        # Iterating a DataLoader draws from the global RNG; keep the random stream of phase 2 unchanged.
        with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
            theta = theta_fn()
    topics = resolve_grn_topics(model, config, theta, null=null)
    set_grn_topics(model, topics)
    excluded = zero_excluded_grn(model)
    record = {
        "format": GRN_TOPICS_FORMAT,
        "num_topics": model.num_topics,
        "selection": mode,
        "grn_topics": grn_topic_ids(model),
        "excluded_topics": excluded,
        "dead_max_theta": _dead_topic_thresholds(config)[0],
        "dead_max_mean_fraction": _dead_topic_thresholds(config)[1],
        "source": getattr(config, "grn_live_topics_file", None) or ("training-cell theta" if needs_theta else mode),
    }
    if null is not None:
        record["source"] = "shuffled training cells"
        record["dead_max_theta"] = record["dead_max_mean_fraction"] = None  # the collapsed-topic rule is not used
        keys = ("grn_null_top_cells", "grn_null_shuffles", "grn_null_quantile", "grn_null_observed_quantile",
                "grn_null_seed", "shuffled_cells")
        record["null_rule"] = {key: null[key] for key in keys}
        record["per_topic"] = [
            {"topic": t, "cutoff": float(c), "observed_quantile": float(o), "kept": bool(k)}
            for t, (c, o, k) in enumerate(zip(null["cutoff"], null["observed"], null["keep"]))
        ]
    checkpoint = getattr(config, "phase1_checkpoint", None)
    if checkpoint:
        from .diagnostics import _sha256

        record["phase1_checkpoint_sha256"] = _sha256(checkpoint)
    if theta is not None:
        record["max_excluded_theta_mass_train_cells"] = max_excluded_theta_mass(theta, excluded)
    if excluded:
        logger.info(
            "GRN phase uses %d of %d topics; excluded %s", len(record["grn_topics"]), model.num_topics, excluded
        )
        if null is not None:
            for t in excluded:
                row = record["per_topic"][t]
                logger.info("Topic %d dropped: observed quantile %.4g <= shuffled-cell cutoff %.4g",
                            t, row["observed_quantile"], row["cutoff"])
            logger.info("Largest topic mass of a training cell on dropped topics: %.4g",
                        record["max_excluded_theta_mass_train_cells"])
    model._grn_selection_record = record
    return record


def sequential_loader(loader):
    """Unshuffled loader over the same cells, so reading theta does not advance a shuffling generator."""
    # Preserve a subset sampler's population; using loader.dataset alone can
    # accidentally include validation/test cells. Do not consume its generator.
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset, SubsetRandomSampler

    sampler = loader.sampler
    if isinstance(sampler, SubsetRandomSampler):
        dataset = Subset(loader.dataset, list(sampler.indices))
    elif isinstance(sampler, SequentialSampler) or (
        isinstance(sampler, RandomSampler) and not sampler.replacement and sampler.num_samples == len(loader.dataset)
    ):
        dataset = loader.dataset
    else:
        raise ValueError("Topic selection requires an unambiguous training-cell sampler")
    return DataLoader(dataset, batch_size=loader.batch_size or 512, shuffle=False)


def write_grn_topics(folder, record):
    """Write the GRN topic record next to the phase-2 weights."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / GRN_TOPICS_FILE).write_text(json.dumps(record, indent=1))


def load_grn_topics(model, folder_or_file):
    """Restore a saved GRN topic selection onto ``model``; a missing file means all topics."""
    path = Path(folder_or_file)
    if path.suffix == ".pth":
        sidecar = path.with_suffix(".topics.json")
        path = sidecar if sidecar.is_file() else path.parent / GRN_TOPICS_FILE
    path = path / GRN_TOPICS_FILE if path.is_dir() else path
    if not path.is_file():
        set_grn_topics(model, None)
        model._grn_selection_record = None
        return None
    record = json.loads(path.read_text())
    if record.get("format") != GRN_TOPICS_FORMAT:
        raise ValueError(f"Unsupported GRN topic record format in {path}")
    if record["num_topics"] != model.num_topics:
        raise ValueError(f"{path} is for {record['num_topics']} topics, but the model has {model.num_topics}")
    set_grn_topics(model, record["grn_topics"])
    model._grn_selection_record = record
    return record
