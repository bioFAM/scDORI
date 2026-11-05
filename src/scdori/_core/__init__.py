from scdori._lazy import resolve_export

_EXPORTS = {
    "trainConfig": (".config", "trainConfig"),
    "load_scdori_inputs": (".data_io", "load_scdori_inputs"),
    "save_model_weights": (".data_io", "save_model_weights"),
    "compute_activator_tf_activity_per_cell": (".downstream", "compute_activator_tf_activity_per_cell"),
    "compute_atac_grn_activator_with_significance": (".downstream", "compute_atac_grn_activator_with_significance"),
    "compute_atac_grn_repressor_with_significance": (".downstream", "compute_atac_grn_repressor_with_significance"),
    "compute_neighbors_umap": (".downstream", "compute_neighbors_umap"),
    "compute_repressor_tf_activity_per_cell": (".downstream", "compute_repressor_tf_activity_per_cell"),
    "compute_significant_grn": (".downstream", "compute_significant_grn"),
    "compute_topic_gene_matrix": (".downstream", "compute_topic_gene_matrix"),
    "compute_topic_peak_umap": (".downstream", "compute_topic_peak_umap"),
    "get_top_activators_per_topic": (".downstream", "get_top_activators_per_topic"),
    "get_top_repressor_per_topic": (".downstream", "get_top_repressor_per_topic"),
    "load_best_model": (".downstream", "load_best_model"),
    "save_regulons": (".downstream", "save_regulons"),
    "get_latent_topics": (".train_grn", "get_latent_topics"),
    "initialize_scdori_parameters": (".models", "initialize_scdori_parameters"),
    "scDoRI": (".models", "scDoRI"),
    "get_tf_expression": (".train_grn", "get_tf_expression"),
    "train_model_grn": (".train_grn", "train_model_grn"),
    "train_scdori_phases": (".train_scdori", "train_scdori_phases"),
    "set_seed": (".utils", "set_seed"),
    "phase1_diagnostics": (".diagnostics", "phase1_diagnostics"),
    "Phase1Diagnostics": (".diagnostics", "Phase1Diagnostics"),
    "read_live_topics": (".diagnostics", "read_live_topics"),
    "load_grn_topics": (".topic_selection", "load_grn_topics"),
    "set_grn_topics": (".topic_selection", "set_grn_topics"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    return resolve_export(name, _EXPORTS, globals())


def __dir__():
    return sorted(set(globals()) | set(__all__))
