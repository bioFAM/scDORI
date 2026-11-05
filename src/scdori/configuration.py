"""Shared, optional project configuration for the three public notebooks."""

import copy
import json
import logging
import os
import shutil
from dataclasses import dataclass, field, fields
from pathlib import Path

from scdori._core.config import trainConfig
from scdori.pp.config import ppConfig


def training_input_order(rna, atac, tf_indices, batch_categories):
    """Record the cell/feature/batch order needed to reuse a phase-1 checkpoint."""
    return {
        "cells": list(rna.obs_names.astype(str)),
        "genes": list(rna.var_names.astype(str)),
        "peaks": list(atac.var_names.astype(str)),
        "tfs": list(rna.var_names[tf_indices].astype(str)),
        "batches": [str(value) for value in batch_categories],
    }


def check_training_input_order(checkpoint, expected):
    """Reject reordered inputs for saved notebook checkpoints; warn for older ones."""
    path = Path(checkpoint).parent / "training_input_order.json"
    if not path.is_file():
        logging.getLogger(__name__).warning(
            "This older checkpoint has no saved cell/feature order. Verify that its original inputs and split are used."
        )
        return
    saved = json.loads(path.read_text())
    changed = [key for key in expected if saved.get(key) != expected[key]]
    if changed:
        raise ValueError(f"Checkpoint input order differs for {changed}; use its original inputs")


def check_checkpoint_model_config(checkpoint, config):
    """Reject checkpoints whose architecture the softmax/BatchNorm loader cannot reproduce."""
    folder = Path(checkpoint).parent
    for path in (folder / "notebook_config.json", folder / "config.json", folder.parent / "config.json"):
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        if path == folder.parent / "config.json" and not any(
            key in payload for key in ("phase1_run", "phase1_penalty_study")
        ):
            continue
        saved = payload.get("training", payload.get("effective_settings", {}))
        if not saved:
            continue
        unsupported = []
        for key, default in (
            ("theta_activation", "softmax"),
            ("theta_temperature", 1.0),
            ("decoder_norm", "batchnorm"),
        ):
            if saved.get(key, default) != default:
                unsupported.append(key)
        if saved.get("theta_temperature_final") not in (None, 1.0):
            unsupported.append("theta_temperature_final")
        for part in ("atac", "tf", "rna"):
            key = f"decoder_norm_{part}"
            if saved.get(key, "") not in ("", "batchnorm"):
                unsupported.append(key)
        if saved.get("label_weight", 0) != 0:
            unsupported.append("label_weight")
        if unsupported:
            raise ValueError(f"This checkpoint uses settings that scDORI cannot reproduce: {unsupported}. See {path}")
        changed = [
            key
            for key in ("num_topics", "dim_encoder1", "dim_encoder2", "batch_col")
            if key in saved and getattr(config, key) != saved[key]
        ]
        if changed:
            raise ValueError(f"Settings differ from the trained model for {changed}. See {path}")
        return saved
    return None


@dataclass
class DownstreamConfig:
    """Analysis options; these do not change model training."""

    topic_activity_threshold: float | None = 0.07
    exclude_topics: list[int] = field(default_factory=list)
    scale_selected_topics: bool = True
    grn_output: str = "results/downstream"
    run_gastrulation_examples: bool = False  # no longer used: example plots pick topics from each fit
    celltype_palette: str | None = None  # optional JSON {"label": "#colour"} for cell-type plots
    example_celltypes: list[str] = field(default_factory=list)
    example_topics: list[int] = field(default_factory=list)
    example_tfs: list[str] = field(default_factory=list)
    example_gene: str | None = None
    geneset_file: str | None = None  # .gmt, or a table with geneset and genesymbol columns
    geneset_name_contains: str | None = None  # keep only the sets whose name contains this text
    ora_min_targets: int = 5
    tf_expression_file: str | None = None
    allow_tf_expression_recompute: bool = False
    permutation_seed: int | None = None
    activator_alpha: float = 0.05
    repressor_alpha: float = 0.05


@dataclass
class NotebookConfig:
    """Independent configuration objects and their path provenance."""

    preprocessing: object
    training: object
    downstream: DownstreamConfig
    source: Path | None
    root: Path


def _options(config):
    names = {f.name for f in fields(config)}
    names.update(
        k
        for k, value in vars(type(config)).items()
        if not k.startswith("_") and isinstance(value, (str, int, float, bool, list, Path))
    )
    return {name: getattr(config, name) for name in sorted(names)}


def _apply(config, values, section):
    if not isinstance(values, dict):
        raise ValueError(f"{section} must be a JSON object")
    known = _options(config)
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"Unknown {section} setting: {key}")
        default = known[key]
        valid = (
            (type(default) is bool and type(value) is bool)
            or (type(default) is int and type(value) is int)
            or (type(default) is float and type(value) in (float, int))
            or (isinstance(default, (str, Path)) and isinstance(value, str))
            or (isinstance(default, list) and isinstance(value, list))
            or (default is None and key != "permutation_seed" and (value is None or isinstance(value, str)))
            or (key == "topic_activity_threshold" and value is None)
            or (key == "window_size" and (value is None or (type(value) is int and value >= 0)))
            or (key == "permutation_seed" and (value is None or type(value) is int))
        )
        if not valid:
            raise ValueError(f"Invalid type for {section}.{key}: {value!r}")
        if key in {"example_celltypes", "example_tfs"} and any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError(f"{section}.{key} must contain nonempty names")
        if key == "example_topics" and (
            any(type(item) is not int or item < 0 for item in value) or len(set(value)) != len(value)
        ):
            raise ValueError(f"{section}.{key} must contain unique nonnegative topic integers")
        setattr(config, key, copy.deepcopy(value))


def _path(value, root):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_notebook_config(config_file=None):
    """Load shared JSON settings, or preserve the existing Python config defaults.

    An explicit path or SCDORI_CONFIG_FILE takes precedence. Otherwise search
    the current directory and its ancestors for scdori_config.json. Relative
    paths in JSON resolve against that file, independently of Jupyter's cwd.
    Without JSON, Python config defaults resolve against the nearest source
    checkout root (or cwd), and are copied so notebook changes are isolated.
    """
    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents]
    requested = config_file if config_file is not None else os.environ.get("SCDORI_CONFIG_FILE")
    source = (
        _path(requested, cwd)
        if requested
        else next((p / "scdori_config.json" for p in candidates if (p / "scdori_config.json").is_file()), None)
    )
    root = (
        source.parent
        if source
        else next((p for p in candidates if (p / "src/scdori").is_dir() and (p / "pyproject.toml").is_file()), cwd)
    )
    data = json.loads(source.read_text()) if source else {}
    if not isinstance(data, dict) or set(data) - {"preprocessing", "training", "downstream"}:
        raise ValueError("Config sections must be preprocessing, training and/or downstream")
    pp, training, downstream = copy.deepcopy(ppConfig), copy.deepcopy(trainConfig), DownstreamConfig()
    for obj, section in ((pp, "preprocessing"), (training, "training"), (downstream, "downstream")):
        for name, value in _options(obj).items():
            setattr(obj, name, copy.deepcopy(value))
        _apply(obj, data.get(section, {}), section)
    pp.resolve_window_size(data.get("preprocessing", {}).get("window_size"))  # species is known only now
    # A common preprocessing location automatically feeds training unless overridden.
    if "preprocessing" in data:
        for target, origin in (("data_dir", "data_dir"), ("output_subdir", "output_subdir_name")):
            if target not in data.get("training", {}):
                setattr(training, target, getattr(pp, origin))
    for name in ("data_dir", "genome_dir", "motif_directory"):
        setattr(pp, name, str(_path(getattr(pp, name), root)))
    training.data_dir = _path(training.data_dir, root)
    for name in ("weights_folder_scdori", "weights_folder_grn", "best_scdori_model_path", "best_grn_model_path"):
        setattr(training, name, str(_path(getattr(training, name), root)))
    for name in ("phase1_checkpoint", "phase1_split_file", "grn_live_topics_file"):
        if getattr(training, name, None) is not None:
            setattr(training, name, str(_path(getattr(training, name), root)))
    downstream.grn_output = str(_path(downstream.grn_output, root))
    if downstream.celltype_palette is not None:
        downstream.celltype_palette = str(_path(downstream.celltype_palette, root))
    if downstream.geneset_file is not None:
        downstream.geneset_file = str(_path(downstream.geneset_file, root))
    if downstream.tf_expression_file is not None:
        downstream.tf_expression_file = str(_path(downstream.tf_expression_file, root))
    print(f"Configuration: {source if source else 'existing Python config.py defaults'}; path root: {root}")
    return NotebookConfig(pp, training, downstream, source, root)


def check_notebook_inputs(settings, stage):
    """Check paths and prerequisite commands without loading large matrices."""
    if stage == "preprocessing":
        cfg = settings.preprocessing
        needed = [
            Path(cfg.data_dir) / cfg.rna_adata_file_name,
            Path(cfg.data_dir) / cfg.atac_adata_file_name,
            Path(cfg.motif_directory) / f"{cfg.motif_database}_{cfg.species}.meme",
        ]
        if shutil.which("bedtools") is None:
            raise RuntimeError("Preprocessing requires BEDTools on PATH. See docs/installation.md.")
    elif stage in ("training", "downstream"):
        cfg = settings.training
        from ._core.topic_penalties import validate_topic_penalties

        validate_topic_penalties(cfg)
        if cfg.tf_expression_scaling not in {"within_topic", "mean_within_across"}:
            raise ValueError("tf_expression_scaling must be 'within_topic' or 'mean_within_across'")
        if not 0 <= cfg.tf_expression_clamp <= 1:
            raise ValueError("tf_expression_clamp must be between 0 and 1")
        if not 0 <= cfg.tf_expression_min_detection <= 1:
            raise ValueError("tf_expression_min_detection must be between 0 and 1, for example 0.02 for 2%")
        if not isinstance(cfg.tf_expression_keep_tfs, (list, tuple)):
            raise ValueError('tf_expression_keep_tfs must be a list of TF names, for example ["Noto"]')
        if stage == "training" and cfg.phase1_checkpoint is not None:
            check_checkpoint_model_config(cfg.phase1_checkpoint, cfg)
        folder = cfg.data_dir / cfg.output_subdir
        needed = [
            folder / getattr(cfg, name)
            for name in (
                "rna_metacell_file",
                "atac_metacell_file",
                "gene_peak_distance_file",
                "insilico_chipseq_act_file",
                "insilico_chipseq_rep_file",
            )
        ]
        if cfg.freeze_phase1_in_grn and any(
            getattr(cfg, name)
            for name in (
                "update_encoder_in_grn",
                "update_peak_gene_in_grn",
                "update_topic_peak_in_grn",
                "update_topic_tf_in_grn",
            )
        ):
            raise ValueError("Fully frozen phase 2 cannot also enable update_*_in_grn flags")
        if stage == "downstream":
            checkpoint = Path(cfg.weights_folder_grn) / "best_scdori_best_eval.pth"
            saved = check_checkpoint_model_config(checkpoint, cfg) or {}
            fitted_selection = saved.get("grn_topic_selection", "all") if saved else cfg.grn_topic_selection
            if fitted_selection != "all" and not any(
                path.is_file()
                for path in (checkpoint.with_suffix(".topics.json"), checkpoint.parent / "grn_topics.json")
            ):
                raise FileNotFoundError(
                    "This fitted model selected phase-2 topics; copy its grn_topics.json or checkpoint .topics.json sidecar"
                )
            for name in ("activator_alpha", "repressor_alpha"):
                if not 0 < getattr(settings.downstream, name) < 1:
                    raise ValueError(f"{name} must be strictly between 0 and 1")
            seed = settings.downstream.permutation_seed
            if seed is not None and (type(seed) is not int or seed < 0):
                raise ValueError("permutation_seed must be a nonnegative integer or null")
            if settings.downstream.tf_expression_file is not None:
                needed.append(Path(settings.downstream.tf_expression_file))
            needed.append(Path(cfg.weights_folder_grn) / "best_scdori_best_eval.pth")
            if settings.downstream.geneset_file:
                needed.append(Path(settings.downstream.geneset_file))
            snapshot = Path(cfg.weights_folder_grn) / "notebook_config.json"
            if snapshot.is_file():
                saved = json.loads(snapshot.read_text())["training"]
                changed = [
                    key
                    for key in ("num_topics", "dim_encoder1", "dim_encoder2", "batch_col")
                    if getattr(cfg, key) != saved[key]
                ]
                if changed:
                    raise ValueError(f"Downstream settings differ from the trained model: {changed}. See {snapshot}")
    else:
        raise ValueError(f"Unknown notebook stage: {stage}")
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing notebook inputs:\n"
            + "\n".join(missing)
            + "\nCheck scdori_config.json or the existing config.py settings."
        )
    print(f"{stage}: {len(needed)} required input files found")


def _thread_settings():
    """CPU thread limits in force. Clusters, and so the selected peaks, can change with the thread count."""
    limits = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    return {"cpu_count": os.cpu_count(), **{name: os.environ.get(name) for name in limits}}


def save_notebook_config(settings, stage, folder):
    """Record the effective configuration and thread settings alongside generated data or checkpoints."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "source_config": str(settings.source) if settings.source else None,
        "threads": _thread_settings(),
        **{name: _options(getattr(settings, name)) for name in ("preprocessing", "training", "downstream")},
    }
    (folder / "notebook_config.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
