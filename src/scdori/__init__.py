import logging

from ._lazy import resolve_export
try:
    from ._version import __version__, __version_tuple__
except ImportError:  # A source tree that was never installed has no generated version file.
    __version__ = "unknown"
    __version_tuple__ = ()

_EXPORTS = {
    "load_downstream_model": ("._core.downstream_inputs", "load_downstream_model"),
    "preview_phase2_tfs": ("._core.tf_preview", "preview_phase2_tfs"),
    "read_gene_sets": ("._core.exploration", "read_gene_sets"),
    "regions_for_link": ("._core.exploration", "regions_for_link"),
    "apply_label_colors": ("._core.exploration", "apply_label_colors"),
    "choose_example_topics": ("._core.exploration", "choose_example_topics"),
    "example_groups": ("._core.exploration", "example_groups"),
    "example_tf": ("._core.exploration", "example_tf"),
    "linked_peak_table": ("._core.exploration", "linked_peak_table"),
    "normalized_counts_for_plot": ("._core.exploration", "normalized_counts_for_plot"),
    "top_binding_tfs": ("._core.exploration", "top_binding_tfs"),
    "top_groups_for_topics": ("._core.exploration", "top_groups_for_topics"),
    "top_targets": ("._core.exploration", "top_targets"),
    "top_topics_for_group": ("._core.exploration", "top_topics_for_group"),
    "top_topics_for_tf": ("._core.exploration", "top_topics_for_tf"),
    "compare_tf_rankings": ("._core.interpretation", "compare_tf_rankings"),
    "tf_importance_scores": ("._core.interpretation", "tf_importance_scores"),
    "filter_tf_display": ("._core.tf_display", "filter_tf_display"),
    "topic_expression_support": ("._core.tf_display", "topic_expression_support"),
    "prepare_count_matrix": ("._sparse", "prepare_count_matrix"),
    "pp": (".pp", None),
    "pl": (".pl", None),
    "_core": ("._core", None),
    "compute_activator_tf_activity_per_cell": ("._core", "compute_activator_tf_activity_per_cell"),
    "compute_atac_grn_activator_with_significance": ("._core", "compute_atac_grn_activator_with_significance"),
    "compute_atac_grn_repressor_with_significance": ("._core", "compute_atac_grn_repressor_with_significance"),
    "compute_neighbors_umap": ("._core", "compute_neighbors_umap"),
    "compute_repressor_tf_activity_per_cell": ("._core", "compute_repressor_tf_activity_per_cell"),
    "compute_significant_grn": ("._core", "compute_significant_grn"),
    "compute_topic_gene_matrix": ("._core", "compute_topic_gene_matrix"),
    "compute_topic_peak_umap": ("._core", "compute_topic_peak_umap"),
    "get_latent_topics": ("._core", "get_latent_topics"),
    "get_tf_expression": ("._core", "get_tf_expression"),
    "get_top_activators_per_topic": ("._core", "get_top_activators_per_topic"),
    "get_top_repressor_per_topic": ("._core", "get_top_repressor_per_topic"),
    "initialize_scdori_parameters": ("._core", "initialize_scdori_parameters"),
    "load_best_model": ("._core", "load_best_model"),
    "load_scdori_inputs": ("._core", "load_scdori_inputs"),
    "save_model_weights": ("._core", "save_model_weights"),
    "save_regulons": ("._core", "save_regulons"),
    "scDoRI": ("._core", "scDoRI"),
    "set_seed": ("._core", "set_seed"),
    "train_model_grn": ("._core", "train_model_grn"),
    "train_scdori_phases": ("._core", "train_scdori_phases"),
    "trainConfig": ("._core", "trainConfig"),
    "phase1_diagnostics": ("._core", "phase1_diagnostics"),
    "Phase1Diagnostics": ("._core", "Phase1Diagnostics"),
    "read_live_topics": ("._core", "read_live_topics"),
    "load_grn_topics": ("._core", "load_grn_topics"),
    "set_grn_topics": ("._core", "set_grn_topics"),
    "plot_downstream_targets": (".pl", "plot_downstream_targets"),
    "plot_topic_activation_heatmap": (".pl", "plot_topic_activation_heatmap"),
    "compute_gene_peak_distance_matrix": (".pp", "compute_gene_peak_distance_matrix"),
    "compute_hvgs_and_tfs": (".pp", "compute_hvgs_and_tfs"),
    "compute_in_silico_chipseq": (".pp", "compute_in_silico_chipseq"),
    "compute_motif_scores": (".pp", "compute_motif_scores"),
    "create_dir_if_not_exists": (".pp", "create_dir_if_not_exists"),
    "create_extended_gene_bed": (".pp", "create_extended_gene_bed"),
    "create_metacells": (".pp", "create_metacells"),
    "download_genome_references": (".pp", "download_genome_references"),
    "filter_protein_coding_genes": (".pp", "filter_protein_coding_genes"),
    "intersect_cells": (".pp", "intersect_cells"),
    "keep_promoters_and_select_hv_peaks": (".pp", "keep_promoters_and_select_hv_peaks"),
    "load_anndata": (".pp", "load_anndata"),
    "load_gtf": (".pp", "load_gtf"),
    "load_motif_database": (".pp", "load_motif_database"),
    "ppConfig": (".pp", "ppConfig"),
    "remove_mitochondrial_genes": (".pp", "remove_mitochondrial_genes"),
    "run_bedtools_intersect": (".pp", "run_bedtools_intersect"),
    "save_processed_datasets": (".pp", "save_processed_datasets"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    return resolve_export(name, _EXPORTS, globals())


def __dir__():
    return sorted(set(globals()) | set(__all__))


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

if not logging.root.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s\t%(message)s"))
    _logger.addHandler(_handler)
    _logger.propagate = False
