# Preprocessing

Specify the ATAC count convention when starting preprocessing. For example,
Signac fragment counts must not undergo the insertion-to-fragment conversion:

```python
rna_metacell, atac_metacell = create_metacells(
    data_rna, data_atac, atac_count_mode="fragments"
)
```

`create_metacells`, `select_highly_variable_peaks_by_std`, and
`keep_promoters_and_select_hv_peaks` accept `atac_count_mode="insertions"` or
`"fragments"`. If omitted, they read
`data_atac.uns["scdori_atac_count_mode"]`, falling back to the previous
insertion-count behavior for untagged inputs.

Metacell output is always tagged as fragment means. The tag survives H5AD
save/load and is honored by training, including the frozen phase 2 cache.
Fractional means are preserved; they are not converted a second time. Peak
selection scores converted values where necessary but leaves the selected
matrix in its original units and records those units.

For previously prepared fragment-count data used directly for training, set
the convention before constructing loaders or calling training functions:

```python
atac_metacell.uns["scdori_atac_count_mode"] = "fragments"
```

Only add this tag when the input actually contains fragments or their means.
Values alone cannot reliably identify the convention of older untagged files.

Metacell RNA `.X` averages **raw input counts**, and the layer `mean_log` averages
each cell's log-normalized counts, which TF binding correlates by default.
Normalization and logging run on separate copies, leaving input counts intact even
on failure. Existing count layers are left unchanged. Output `.obs["num_cells"]` records
the number of cells per metacell. Every RNA cell must have an ATAC measurement;
missing cells are rejected before clustering instead of being treated as zeros.
These corrections can change results when preprocessing is rerun.

```{eval-rst}
.. currentmodule:: scdori
.. autosummary::
    :toctree: ../generated

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
    ppConfig
    remove_mitochondrial_genes
    run_bedtools_intersect
    pp.retain_requested_peaks
    save_processed_datasets
```
