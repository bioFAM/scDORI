import json
import logging
import math
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """
    Global configuration for the scDORI modeling pipeline.

    This file defines top-level constants and parameters controlling:

    1. Logging
    2. File paths for data and outputs
    3. Model architecture details (numbers of topics, hidden dimensions)
    4. Training phases and hyperparameters
    5. Loss weighting for different data modalities (ATAC, TF, RNA)
    6. Regularization and early stopping settings
    7. Significance testing cutoffs for TF-gene links
    8. UMAP parameters for visualization

    Attributes
    ----------
    logging_level : int
        The Python logging level (e.g., logging.INFO).
    data_dir : Path
        Base directory containing the processed anndata and other precomputed files.
    output_subdir : str
        Subdirectory name within data_dir for storing or accessing outputs.
    rna_metacell_file : str
        Name of the H5AD file containing RNA data.
    atac_metacell_file : str
        Name of the H5AD file containing ATAC data.
    batch_col : str
        Key for the batch column in the AnnData object.
    gene_peak_distance_file : str
        Filename for the NumPy array containing gene-peak distances.
    insilico_chipseq_act_file : str
        Filename for the in silico ChIP-seq activator embeddings.
    insilico_chipseq_rep_file : str
        Filename for the in silico ChIP-seq repressor embeddings.
    random_seed : int
        Random seed for reproducibility.
    batch_size_cell : int
        Batch size for training.
    theta_entropy_weight : float
        Optional phase-1 penalty on per-cell topic entropy, summed over the
        minibatch. Zero by default. Positive values encourage fewer topics per cell.
    usage_balance_weight : float
        Optional phase-1 penalty encouraging even mean topic use within each
        minibatch. Zero by default. Both topic penalties depend on batch size;
        validation and early stopping retain the existing objective without these terms.
    dim_encoder1 : int
        Dimension of the first encoder layer.
    dim_encoder2 : int
        Dimension of the second encoder layer.
    num_topics : int
        Number of latent topics for scDORI.
    batch_size_cell_prediction : int
        Batch size when making predictions in eval mode (e.g., forward passes only).
    epoch_warmup_1 : int
        Number of epochs to run "warmup_1" (ATAC+TF) before adding RNA.
    max_scdori_epochs : int or "auto"
        Maximum number of epochs for the scDORI phase 1 training (module 1,2,3). "auto" (default):
        ``max(1000, ceil(phase1_max_steps / steps_per_epoch))``, where ``steps_per_epoch`` is the number of
        optimizer steps (minibatches) per epoch, ``ceil(training cells / batch size)``. An integer overrides it.
    max_grn_epochs : int or "auto"
        Maximum number of epochs for the GRN training phase (module 4). "auto" (default):
        ``max(1000, ceil(grn_max_steps / steps_per_epoch))``. An integer overrides it.
    phase1_max_steps, phase1_patience_steps, grn_max_steps, grn_patience_steps : int
        Optimizer-step budgets behind the "auto" epoch limits and patience (defaults 25,000, 1,000, 25,000 and
        300). They come from phase-1 fits on 10 benchmark datasets (batch 128, 33-86 steps per epoch), which stopped
        after about 4,000-20,000 steps; small inputs such as a few hundred metacells have few steps per epoch and
        therefore get more epochs.
    freeze_phase1_in_grn : bool
        Freeze all phase 1 parameters and state; train only GRN and its batch correction.
        Enabled by default. Set False for the legacy phase 2, which can also update
        library networks and other parameters.
    cache_frozen_grn : bool
        Cache fixed features in fully frozen mode. False is the full-forward reference.
    phase2_train_nuisance : bool
        Experimental, default False. With ``freeze_phase1_in_grn``, phase 2 also trains the two phase-1 terms of
        the RNA-from-GRN likelihood: the RNA dispersion ``rna_alpha_nb`` and the RNA library-size network
        ``rna_library_factor`` (all its parameters; it runs in training mode, so its BatchNorm statistics update
        and its dropout is on). Everything else stays as in the default phase 2. Topic activities and the ATAC/TF
        losses stay cached; the RNA library factor and the phase-1 RNA loss (which also depends on the dispersion)
        are recomputed for each minibatch from the cached topic activities.
    learning_rate_nuisance : float or None
        Learning rate for the ``phase2_train_nuisance`` parameters (a second Adam parameter group). None
        (default) uses ``learning_rate_grn``.
    update_encoder_in_grn : bool
        Whether to unfreeze the encoder during the GRN phase.
    update_peak_gene_in_grn : bool
        Whether to unfreeze the peak-gene links in the GRN phase.
    update_topic_peak_in_grn : bool
        Whether to unfreeze the topic-peak links in the GRN phase.
    grn_topic_selection : str
        Topics modelled in the GRN phase: "live" (default: skip collapsed topics after phase 1), "all" (reference
        behaviour), "shuffle_null" (see below) or "explicit" (use ``grn_topics``). Skipped topics keep their
        original IDs; their GRN weights, TF gate and ATAC GRNs are zero. "shuffle_null": with the trained encoder in
        eval mode, take each topic's ``grn_null_top_cells`` training cells with the highest weight, shuffle their
        gene values across genes and their peak values across peaks within each cell, encode them again
        (``grn_null_shuffles`` times, seeds ``grn_null_seed + k``); the topic's cutoff is the
        ``grn_null_quantile`` quantile of its weights in the shuffled cells, and the topic is kept if the
        ``grn_null_observed_quantile`` quantile of its observed weights in the training cells is above the cutoff.
        The cutoffs and decisions are saved in ``grn_topics.json``. Topics that phase 1 left unused have weights
        near zero in every cell, and this comparison can keep them.
    grn_null_top_cells, grn_null_shuffles, grn_null_quantile, grn_null_observed_quantile, grn_null_seed
        Settings of "shuffle_null" (defaults 100, 5, 0.95, 0.99, 1000).
    grn_topics : list of int
        Topic IDs for ``grn_topic_selection="explicit"``.
    grn_live_topics_file : str or None
        ``live_topics.json`` written by ``phase1_diagnostics``. With "live", this file
        defines the topics; otherwise they are computed from the training cells.
    grn_dead_topic_max_theta, grn_dead_topic_max_mean_fraction : float
        With "live" and no file: a topic is collapsed if its theta stays below
        ``grn_dead_topic_max_theta`` (default 0.05) in every training cell **and** its mean theta is below
        ``grn_dead_topic_max_mean_fraction / num_topics`` (default 0.1 / num_topics). Both conditions must hold.
        Which topics collapse depends on the cell types and conditions in the data; the lower default keeps
        topics that reach 5% of a cell's topic mass anywhere.
    update_topic_tf_in_grn : bool
        Whether to unfreeze the topic-TF links in the GRN phase.
    eval_frequency : int
        How often (in epochs) to evaluate validation loss.
    phase1_patience : int or "auto"
        Early stopping patience for phase 1, counted in validation evaluations (one every ``eval_frequency``
        epochs); training stops after ``phase1_patience + 1`` evaluations without improvement. "auto" (default):
        ``max(30, ceil(phase1_patience_steps / (steps_per_epoch * eval_frequency)))``, i.e. 30 for typical data
        (v2.0.0 used 50). An integer overrides it.
    grn_val_patience : int or "auto"
        Early stopping patience for the GRN phase, counted in evaluations as for ``phase1_patience``. "auto"
        (default): ``max(5, ceil(grn_patience_steps / (steps_per_epoch * eval_frequency)))``. An integer
        overrides it.
    learning_rate_scdori : float
        Learning rate for scDORI phase 1 training (module 1,2 3).
    learning_rate_grn : float
        Learning rate for the GRN training phase.
    weight_atac_phase1 : float
        Loss weight for ATAC reconstruction in Phase 1:warmup_1.
    weight_tf_phase1 : float
        Loss weight for TF reconstruction in Phase 1:warmup_1.
    weight_rna_phase1 : float
        Loss weight for RNA reconstruction in Phase 1:warmup_1. set to 0.
    weight_rna_grn_phase1 : float
        Loss weight for GRN-based RNA reconstruction in Phase 1:warmup_1. set to 0.
    weight_atac_phase2 : float
        Loss weight for ATAC reconstruction in Phase 1:warmup_2.
    weight_tf_phase2 : float
        Loss weight for TF reconstruction in Phase 1:warmup_2.
    weight_rna_phase2 : float
        Loss weight for RNA reconstruction in Phase 1:warmup_2.
    weight_rna_grn_phase2 : float
        Loss weight for GRN-based RNA reconstruction in Phase 1:warmup_2. set to 0.
    weight_atac_grn : float
        Loss weight for ATAC reconstruction in the GRN phase.
    weight_tf_grn : float
        Loss weight for TF reconstruction in the GRN phase.
    weight_rna_grn : float
        Loss weight for RNA reconstruction in the GRN phase.
    weight_rna_from_grn : float
        Loss weight for the GRN-based RNA branch in the GRN phase.
    l1_penalty_topic_tf : float
        L1 regularization coefficient on the topic_tf_decoder.
    l2_penalty_topic_tf : float
        L2 regularization coefficient on the topic_tf_decoder.
    l1_penalty_topic_peak : float
        L1 regularization coefficient on the topic_peak_decoder.
    l2_penalty_topic_peak : float
        L2 regularization coefficient on the topic_peak_decoder.
    l1_penalty_gene_peak : float
        L1 regularization coefficient on the gene_peak_factor_learnt.
    l2_penalty_gene_peak : float
        L2 regularization coefficient on the gene_peak_factor_learnt.
    l1_penalty_grn_activator : float
        L1 regularization on GRN activator parameters (tf_gene_topic_activator_grn).
        Default 5e-3 since v2.1.0 (v2.0.0: 5e-5).
    l1_penalty_grn_repressor : float
        L1 regularization on GRN repressor parameters (tf_gene_topic_repressor_grn).
        Default 5e-3 since v2.1.0, matching activators (v2.0.0: 5e-5; 0.0 before v2.0.0).
    tf_expression_mode : str
        Either "True" (use actual TF expression) or "latent" (model's predicted TF expression).
    tf_expression_clamp : float
        Threshold for scaled TF expression in [0, 1], default 0.2.
    tf_expression_scaling : str
        "mean_within_across" (default): average
        min-max scores across TFs within a topic and across all topics for each TF.
        Use "within_topic" with cutoff 0.1 for the historical rule.
        This changes phase-2 inputs and requires retraining phase 2.
    tf_expression_min_detection : float
        Minimum fraction of a topic's top cells (``cells_per_topic``) in which a TF must have at
        least one RNA count to contribute to that topic, default 0.02 (4 of 200 cells); 0 turns it
        off. Used only with measured expression. Requires retraining phase 2.
    tf_expression_keep_tfs : list of str
        TFs the minimum-detection rule never removes. TFs requested with ``tfs_user`` in
        preprocessing are also never removed.
    cells_per_topic : int
        Number of cells sampled per topic to compute topic-level TF expression.
    weights_folder_scdori : str
        Folder to save model weights after the scDORI Phase 1.
    weights_folder_grn : str
        Folder to save model weights after the GRN phase.
    best_scdori_model_path : str
        Filename for saving the best scDORI model (Phase 1).
    best_grn_model_path : str
        Filename for saving the best GRN model.
    umap_n_neighbors : int
        Number of neighbors for UMAP.
    umap_min_dist : float
        Min dist parameter for UMAP.
    umap_random_state : int
        Random seed for UMAP.
    significance_cutoffs : List[float]
        List of thresholds for empirical p-value cutoffs in TF-gene link permutation tests.
    num_permutations : int
        Number of permutations used to compute TF-gene link significance.
    """

    # LOGGING
    logging_level: int = logging.INFO

    # DATA PATHS
    data_dir: Path = Path("data")
    output_subdir: str = "generated"
    rna_metacell_file: str = "rna_processed.h5ad"
    atac_metacell_file: str = "atac_processed.h5ad"
    # One-time CSC -> CSR conversion when loading; opt out to preserve layout.
    count_matrix_csr: bool = True
    # Notebook defaults keep the familiar Run All workflow; inspection is optional.
    notebook_run_phase2: bool = True
    run_phase1_diagnostics: bool = True
    diagnostics_detailed: bool = False
    diagnostics_structure: bool = True
    diagnostics_groupby: str = "celltype"
    phase1_checkpoint: str | None = None
    phase1_split_file: str | None = None
    batch_col: str = "sample"
    gene_peak_distance_file: str = "gene_peak_distance_exp.npy"
    insilico_chipseq_act_file: str = "insilico_chipseq_act.npy"
    insilico_chipseq_rep_file: str = "insilico_chipseq_rep.npy"

    # RANDOM SEED
    random_seed: int = 3141

    # BATCH / ARCHITECTURE
    batch_size_cell: int = 128
    dim_encoder1: int = 500
    dim_encoder2: int = 200
    num_topics: int = 40
    batch_size_cell_prediction: int = 512

    # PHASE1
    epoch_warmup_1: int = 5
    max_scdori_epochs: int | str = "auto"
    phase1_max_steps: int = 25000
    phase1_patience_steps: int = 1000

    # PHASE 2
    max_grn_epochs: int | str = "auto"
    grn_max_steps: int = 25000
    grn_patience_steps: int = 300
    freeze_phase1_in_grn: bool = True
    cache_frozen_grn: bool = True
    phase2_train_nuisance: bool = False
    learning_rate_nuisance: float | None = None
    update_encoder_in_grn: bool = False
    update_peak_gene_in_grn: bool = False
    update_topic_peak_in_grn: bool = False
    update_topic_tf_in_grn: bool = False
    grn_topic_selection: str = "live"
    grn_null_top_cells: int = 100
    grn_null_shuffles: int = 5
    grn_null_quantile: float = 0.95
    grn_null_observed_quantile: float = 0.99
    grn_null_seed: int = 1000
    grn_topics: list = field(default_factory=list)
    grn_live_topics_file: str | None = None
    grn_dead_topic_max_theta: float = 0.05
    grn_dead_topic_max_mean_fraction: float = 0.1

    # early stopping and evaluation
    eval_frequency: int = 1
    phase1_patience: int | str = "auto"
    grn_val_patience: int | str = "auto"

    # LR / LOSSES
    learning_rate_scdori: float = 0.005
    learning_rate_grn: float = 0.001

    # Phase 1 weights (warmup_1)
    weight_atac_phase1: float = 1.0
    weight_tf_phase1: float = 200.0
    weight_rna_phase1: float = 0.0
    weight_rna_grn_phase1: float = 0.0

    # Phase 1 weights (warmup_2)
    weight_atac_phase2: float = 1.0
    weight_tf_phase2: float = 200.0
    weight_rna_phase2: float = 20.0
    weight_rna_grn_phase2: float = 0.0

    # Phase 2 (GRN) weights
    weight_atac_grn: float = 1.0
    weight_tf_grn: float = 200.0
    weight_rna_grn: float = 20.0
    weight_rna_from_grn: float = 20.0

    # REGULARIZATION
    l1_penalty_topic_tf: float = 0.001
    l2_penalty_topic_tf: float = 0.000
    l1_penalty_topic_peak: float = 0.001
    l2_penalty_topic_peak: float = 0.001
    l1_penalty_gene_peak: float = 0.001
    l2_penalty_gene_peak: float = 0.005
    l1_penalty_grn_activator: float = 5e-3
    l1_penalty_grn_repressor: float = 5e-3

    # Optional softmax topic-usage penalties, summed over each training minibatch.
    # Zero preserves the existing objective. Suitable weights depend on batch size.
    theta_entropy_weight: float = 0.0
    usage_balance_weight: float = 0.0

    # TF EXPRESSION SETTINGS
    tf_expression_mode: str = "True"  # FIXME
    tf_expression_clamp: float = 0.2
    tf_expression_scaling: str = "mean_within_across"
    tf_expression_min_detection: float = 0.02
    tf_expression_keep_tfs: list = field(default_factory=list)
    cells_per_topic: int = 200

    # SAVE FOLDERS
    weights_folder_scdori: str = "results/phase1"
    weights_folder_grn: str = "results/phase2"
    best_scdori_model_path: str = "results/phase1/best_scdori_best_eval.pth"
    best_grn_model_path: str = "results/phase2/best_scdori_best_eval.pth"

    # UMAP PARAMETERS
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_random_state: int = 42

    # SIGNIFICANCE SETTINGS
    significance_cutoffs = [0.001, 0.005, 0.01, 0.05]
    num_permutations: int = 1000


def train_default(name):
    """Current default of the ``TrainConfig`` setting ``name``.

    Used when a configuration object lacks a setting, so that it behaves like ``TrainConfig()``.
    """
    for item in fields(TrainConfig):
        if item.name == name:
            return item.default_factory() if item.default is MISSING else item.default
    raise KeyError(f"TrainConfig has no setting {name!r}")


# Training length: "auto" epoch limits and patience are derived from optimizer-step budgets.
AUTO = "auto"
AUTO_LENGTH_SETTINGS = ("max_scdori_epochs", "phase1_patience", "max_grn_epochs", "grn_val_patience")
_LENGTH = {
    "phase1": ("max_scdori_epochs", "phase1_patience", "phase1_max_steps", "phase1_patience_steps", 1000, 30),
    "grn": ("max_grn_epochs", "grn_val_patience", "grn_max_steps", "grn_patience_steps", 1000, 5),
}


def resolve_training_length(config, steps_per_epoch, phase):
    """Epoch limit and early-stopping patience for ``phase`` ("phase1" or "grn").

    ``steps_per_epoch`` is the number of optimizer steps per epoch: ``len(train_loader)``, i.e.
    ``ceil(training cells / batch size)``. A setting equal to "auto" is computed from its step budget
    (``max_epochs = max(1000, ceil(max_steps / steps_per_epoch))`` and
    ``patience = max(minimum, ceil(patience_steps / (steps_per_epoch * eval_frequency)))``, minimum 30 for
    phase 1 and 5 for phase 2); an integer is used as given. Patience is counted in validation evaluations.

    Returns a dict with the resolved values, the inputs and where each value came from.
    """
    epochs_key, patience_key, steps_key, patience_steps_key, min_epochs, min_patience = _LENGTH[phase]
    steps = int(steps_per_epoch)
    if steps < 1:
        raise ValueError("steps_per_epoch must be a positive integer")
    every = getattr(config, "eval_frequency", train_default("eval_frequency"))
    if type(every) is not int or every < 1:
        raise ValueError("eval_frequency must be a positive integer")
    budget = {key: getattr(config, key, train_default(key)) for key in (steps_key, patience_steps_key)}
    for key, value in budget.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    record = {"phase": phase, "steps_per_epoch": steps, "eval_frequency": every, **budget}
    for key, minimum, value in (
        (epochs_key, 1, getattr(config, epochs_key, train_default(epochs_key))),
        (patience_key, 0, getattr(config, patience_key, train_default(patience_key))),
    ):
        if value == AUTO:
            if key == epochs_key:
                value = max(min_epochs, math.ceil(budget[steps_key] / steps))
            else:
                value = max(min_patience, math.ceil(budget[patience_steps_key] / (steps * every)))
            source = "auto"
        elif type(value) is int and value >= minimum:
            source = "config"
        else:
            raise ValueError(f'{key} must be "auto" or an integer >= {minimum}, got {value!r}')
        record[key] = value
        record[key + "_source"] = source
    record["max_epochs"], record["patience"] = record[epochs_key], record[patience_key]
    record["max_epochs_source"] = record[epochs_key + "_source"]
    record["patience_source"] = record[patience_key + "_source"]
    record["max_steps_resolved"] = record["max_epochs"] * steps
    return record


def record_training_length(record, folder):
    """Log the resolved training length and save it as ``training_length_<phase>.json`` in ``folder``."""
    logger.info(
        "%s training length: %d steps per epoch, at most %d epochs (%s), patience %d evaluations (%s)",
        record["phase"], record["steps_per_epoch"], record["max_epochs"], record["max_epochs_source"],
        record["patience"], record["patience_source"],
    )
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"training_length_{record['phase']}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


# Create a default configuration instance
trainConfig = TrainConfig()
