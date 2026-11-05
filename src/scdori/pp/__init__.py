from scdori._lazy import resolve_export

_EXPORTS = {
    "ppConfig": (".config", "ppConfig"),
    "compute_in_silico_chipseq": (".correlation", "compute_in_silico_chipseq"),
    "metacell_correlation_matrices": (".correlation", "metacell_correlation_matrices"),
    "create_dir_if_not_exists": (".data_io", "create_dir_if_not_exists"),
    "load_anndata": (".data_io", "load_anndata"),
    "save_processed_datasets": (".data_io", "save_processed_datasets"),
    "download_genome_references": (".download", "download_genome_references"),
    "intersect_cells": (".filtering", "intersect_cells"),
    "remove_mitochondrial_genes": (".filtering", "remove_mitochondrial_genes"),
    "compute_hvgs_and_tfs": (".gene_selection", "compute_hvgs_and_tfs"),
    "filter_protein_coding_genes": (".gene_selection", "filter_protein_coding_genes"),
    "load_gtf": (".gene_selection", "load_gtf"),
    "create_metacells": (".metacells", "create_metacells"),
    "compute_motif_scores": (".motif_scanning", "compute_motif_scores"),
    "load_motif_database": (".motif_scanning", "load_motif_database"),
    "run_bedtools_intersect": (".motif_scanning", "run_bedtools_intersect"),
    "keep_promoters_and_select_hv_peaks": (".peak_selection", "keep_promoters_and_select_hv_peaks"),
    "retain_requested_peaks": (".peak_selection", "retain_requested_peaks"),
    "compute_gene_peak_distance_matrix": (".utils", "compute_gene_peak_distance_matrix"),
    "create_extended_gene_bed": (".utils", "create_extended_gene_bed"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    return resolve_export(name, _EXPORTS, globals())


def __dir__():
    return sorted(set(globals()) | set(__all__))
