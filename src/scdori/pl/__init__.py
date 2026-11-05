from scdori._lazy import resolve_export

_EXPORTS = {
    "plot_tf_score_views": (".tf_scores", "plot_tf_score_views"),
    "plot_downstream_targets": (".plotting", "plot_downstream_targets"),
    "plot_topic_activation_heatmap": (".plotting", "plot_topic_activation_heatmap"),
    "plot_phase1_diagnostics": (".diagnostics", "plot_phase1_diagnostics"),
    "plot_phase1_overview": (".diagnostics", "plot_phase1_overview"),
    "plot_topic_programs": (".diagnostics", "plot_topic_programs"),
    "topic_status_table": (".diagnostics", "topic_status_table"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    return resolve_export(name, _EXPORTS, globals())


def __dir__():
    return sorted(set(globals()) | set(__all__))
