"""Selection of the topics modelled in the GRN phase (phase 2) and permutations.

The default ``grn_topic_selection="live"`` excludes collapsed topics after phase 1.
Use "all" to retain every topic. With "live" or
"explicit", only the selected topics enter the phase-2 forward pass, GRN factor
caches and permutation tests. Excluded topics keep their original IDs, and their
GRN weights, TF gate and ATAC GRNs are zero, so every output keeps shape K.

The phase-1 model is untouched: theta is still a softmax over all K topics. The
term dropped from the phase-2 prediction is ``theta[:, excluded] @ C[excluded]``,
which is negligible only for collapsed topics (see ``max_excluded_theta_mass``).
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch

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


def collapsed_topics(theta, max_theta=0.1, max_mean_fraction=0.1):
    """Collapsed topics: theta below ``max_theta`` in every cell and mean below ``max_mean_fraction / K``."""
    theta = np.asarray(theta)
    K = theta.shape[1]
    return np.flatnonzero((theta.max(0) < max_theta) & (theta.mean(0) < max_mean_fraction / K))


def max_excluded_theta_mass(theta, excluded):
    """Largest topic mass any cell places on excluded topics (what the GRN phase drops)."""
    excluded = list(excluded)
    return float(np.asarray(theta)[:, excluded].sum(1).max()) if excluded else 0.0


def resolve_grn_topics(model, config, theta=None):
    """Topics for the GRN phase from the config; ``None`` means all topics.

    ``theta`` (training cells x K) is needed for "live" without ``grn_live_topics_file``.
    """
    from .diagnostics import read_live_topics

    mode = getattr(config, "grn_topic_selection", "all")
    if mode == "all":
        return None
    if mode == "explicit":
        topics = list(getattr(config, "grn_topics", []))
        if not topics:
            raise ValueError('grn_topic_selection="explicit" requires grn_topics')
        return np.asarray(topics)
    if mode != "live":
        raise ValueError(f"Unknown grn_topic_selection: {mode!r}; use 'all', 'live' or 'explicit'")
    path = getattr(config, "grn_live_topics_file", None)
    if path:
        checkpoint = getattr(config, "phase1_checkpoint", None)
        if not checkpoint:
            raise ValueError("Loading a live-topic file requires phase1_checkpoint for provenance validation")
        topics = read_live_topics(path, num_topics=model.num_topics, checkpoint=checkpoint)
        if theta is None:
            raise ValueError("Loading a live-topic file requires validation against current training-cell theta")
        dead = collapsed_topics(theta, config.grn_dead_topic_max_theta, config.grn_dead_topic_max_mean_fraction)
        excluded = np.setdiff1d(np.arange(model.num_topics), topics)
        if not np.isin(excluded, dead).all():
            raise ValueError("Live-topic file excludes topics that are not collapsed on current training cells")
        return topics
    if theta is None:
        raise ValueError('grn_topic_selection="live" needs grn_live_topics_file or topic activations')
    dead = collapsed_topics(theta, config.grn_dead_topic_max_theta, config.grn_dead_topic_max_mean_fraction)
    live = np.setdiff1d(np.arange(model.num_topics), dead)
    if not len(live):
        raise ValueError("Every topic is collapsed; phase 1 did not learn usable topics")
    return live


def apply_grn_topic_selection(model, config, theta_fn=None):
    """Resolve, set and log the GRN topic selection; zero excluded GRN weights.

    ``theta_fn`` returns training-cell theta and is only called when needed. Returns a
    JSON-serializable record (also written by :func:`write_grn_topics`).
    """
    mode = getattr(config, "grn_topic_selection", "all")
    flags = ("update_encoder_in_grn", "update_peak_gene_in_grn", "update_topic_peak_in_grn", "update_topic_tf_in_grn")
    if mode != "all" and (
        not getattr(config, "freeze_phase1_in_grn", False) or any(getattr(config, flag, False) for flag in flags)
    ):
        raise ValueError("Topic selection requires fully frozen phase 1, including encoder and BatchNorm state")
    needs_theta = mode == "live" and not getattr(config, "grn_live_topics_file", None)
    theta = None
    if mode != "all" and theta_fn is not None:
        # Iterating a DataLoader draws from the global RNG; keep the random stream of phase 2 unchanged.
        device = next(model.parameters()).device
        with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
            theta = theta_fn()
    topics = resolve_grn_topics(model, config, theta)
    set_grn_topics(model, topics)
    excluded = zero_excluded_grn(model)
    record = {
        "format": GRN_TOPICS_FORMAT,
        "num_topics": model.num_topics,
        "selection": mode,
        "grn_topics": grn_topic_ids(model),
        "excluded_topics": excluded,
        "dead_max_theta": getattr(config, "grn_dead_topic_max_theta", None),
        "dead_max_mean_fraction": getattr(config, "grn_dead_topic_max_mean_fraction", None),
        "source": getattr(config, "grn_live_topics_file", None) or ("training-cell theta" if needs_theta else mode),
    }
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
