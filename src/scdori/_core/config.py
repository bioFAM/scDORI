import logging
from dataclasses import dataclass, field
from pathlib import Path


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
    max_scdori_epochs : int
        Maximum number of epochs for the scDORI phase 1 training (module 1,2,3).
    max_grn_epochs : int
        Maximum number of epochs for the GRN training phase (module 4).
    freeze_phase1_in_grn : bool
        Freeze all phase 1 parameters and state; train only GRN and its batch correction.
        Enabled by default. Set False for the legacy phase 2, which can also update
        library networks and other parameters.
    cache_frozen_grn : bool
        Cache fixed features in fully frozen mode. False is the full-forward reference.
    update_encoder_in_grn : bool
        Whether to unfreeze the encoder during the GRN phase.
    update_peak_gene_in_grn : bool
        Whether to unfreeze the peak-gene links in the GRN phase.
    update_topic_peak_in_grn : bool
        Whether to unfreeze the topic-peak links in the GRN phase.
    grn_topic_selection : str
        Topics modelled in the GRN phase: "all" (reference behaviour), "live"
        (default: skip collapsed topics after phase 1) or "explicit" (use ``grn_topics``). Skipped topics keep
        their original IDs; their GRN weights, TF gate and ATAC GRNs are zero.
    grn_topics : list of int
        Topic IDs for ``grn_topic_selection="explicit"``.
    grn_live_topics_file : str or None
        ``live_topics.json`` written by ``phase1_diagnostics``. With "live", this file
        defines the topics; otherwise they are computed from the training cells.
    grn_dead_topic_max_theta, grn_dead_topic_max_mean_fraction : float
        With "live" and no file: a topic is collapsed if its theta stays below
        ``grn_dead_topic_max_theta`` in every training cell and its mean theta is below
        ``grn_dead_topic_max_mean_fraction / num_topics``.
    update_topic_tf_in_grn : bool
        Whether to unfreeze the topic-TF links in the GRN phase.
    eval_frequency : int
        How often (in epochs) to evaluate validation loss.
    phase1_patience : int
        Early stopping patience (in epochs) for phase 1 training (module 1,2 3).
    grn_val_patience : int
        Early stopping patience (in epochs) for the GRN phase.
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
    l1_penalty_grn_repressor : float
        L1 regularization on GRN repressor parameters (tf_gene_topic_repressor_grn).
        Defaults to 5e-5, matching activators; use 0.0 to reproduce the previous default.
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
    max_scdori_epochs: int = 1000

    # PHASE 2
    max_grn_epochs: int = 1000
    freeze_phase1_in_grn: bool = True
    cache_frozen_grn: bool = True
    update_encoder_in_grn: bool = False
    update_peak_gene_in_grn: bool = False
    update_topic_peak_in_grn: bool = False
    update_topic_tf_in_grn: bool = False
    grn_topic_selection: str = "live"
    grn_topics: list = field(default_factory=list)
    grn_live_topics_file: str | None = None
    grn_dead_topic_max_theta: float = 0.1
    grn_dead_topic_max_mean_fraction: float = 0.1

    # early stopping and evaluation
    eval_frequency: int = 1
    phase1_patience: int = 50
    grn_val_patience: int = 5

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
    l1_penalty_grn_activator: float = 0.00005
    l1_penalty_grn_repressor: float = 0.00005

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


# Create a default configuration instance
trainConfig = TrainConfig()
