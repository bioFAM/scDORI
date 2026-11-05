# API

```{eval-rst}
.. module:: scdori
```

```{eval-rst}
.. currentmodule:: scdori

.. toctree::
    :hidden:

    core.md
    preprocessing.md
    plotting.md
```

## Core

```{eval-rst}
.. autosummary::

    compute_activator_tf_activity_per_cell
    compute_atac_grn_activator_with_significance
    compute_atac_grn_repressor_with_significance
    compute_neighbors_umap
    compute_repressor_tf_activity_per_cell
    compute_significant_grn
    compute_topic_gene_matrix
    compute_topic_peak_umap
    get_latent_topics
    get_tf_expression
    get_top_activators_per_topic
    get_top_repressor_per_topic
    initialize_scdori_parameters
    load_best_model
    load_downstream_model
    load_grn_topics
    load_scdori_inputs
    phase1_diagnostics
    Phase1Diagnostics
    prepare_count_matrix
    preview_phase2_tfs
    read_live_topics
    save_model_weights
    save_regulons
    scDoRI
    set_grn_topics
    set_seed
    train_model_grn
    train_scdori_phases
    trainConfig
```

## Exploring a fitted model

```{eval-rst}
.. autosummary::

    apply_label_colors
    choose_example_topics
    compare_tf_rankings
    example_groups
    example_tf
    filter_tf_display
    linked_peak_table
    normalized_counts_for_plot
    read_gene_sets
    regions_for_link
    tf_importance_scores
    top_binding_tfs
    top_groups_for_topics
    top_targets
    top_topics_for_group
    top_topics_for_tf
    topic_expression_support
```

## Preprocessing

```{eval-rst}
.. autosummary::

    compute_gene_peak_distance_matrix
    compute_hvgs_and_tfs
    compute_in_silico_chipseq
    compute_motif_scores
    create_dir_if_not_exists
    create_extended_gene_bed
    create_metacells
    download_genome_references
    filter_protein_coding_genes
    intersect_cells
    keep_promoters_and_select_hv_peaks
    load_anndata
    load_gtf
    load_motif_database
    pp.retain_requested_peaks
    ppConfig
    remove_mitochondrial_genes
    run_bedtools_intersect
    save_processed_datasets
```

## Plotting

```{eval-rst}
.. autosummary::

    pl.plot_phase1_diagnostics
    pl.plot_phase1_overview
    pl.plot_tf_score_views
    pl.plot_topic_programs
    pl.topic_status_table
    plot_downstream_targets
    plot_topic_activation_heatmap
```

