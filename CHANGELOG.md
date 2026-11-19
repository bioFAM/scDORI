# Changelog

All notable changes to this project will be documented in this file.

## v2.1.0

This release changes several defaults based on our benchmarking (TF binding, phase-2 penalties, training length and
the downstream topic threshold). Each entry below says how to restore the v2.0.0 behaviour. The example notebooks keep
their v2.0.0 outputs.

### TF binding (in-silico ChIP-seq) uses a new method

- Preprocessing combines motif matches and TF–peak correlations into the binding matrices in a new way:
  - motif evidence is the best FIMO match per peak and TF, as −log10 p (`compute_motif_pvalues`), instead of the
    sum of match scores;
  - each match gets a weight that rises linearly from 0 at `motif_match_pvalue_threshold` (1e-3) to 1 at the
    TF's top `insilico_weight_top_pct` percent of matches (default 5), and is 1 above;
  - the metacell correlation is averaged over `insilico_n_clusterings` Leiden clusterings (default 10, seeds
    `insilico_first_seed` = 101 to 110, resolution `leiden_resolution`) of one Harmony neighbour graph
    (`averaged_metacell_correlation`);
  - the cutoff is the `correlation_percentile` (99) of the TF's correlations with peaks without a motif match;
    if a TF has fewer than 100 such peaks, its `n_bg_peaks_for_corr` (5,000) weakest matches are used;
  - activator = weight × correlation and repressor = −weight × |correlation| where the correlation passes the
    cutoff (`compute_in_silico_chipseq_weighted`). Values are not rescaled per TF, so they lie in [−1, 1] and are
    usually well below 1 in size; v2.0.0 scaled each TF to [0, 1].
- FIMO reports p = 0 for a perfect match to some motifs. These are set to the smallest p-value FIMO can report
  for that motif (`motif_pvalue_floor`), which is the value the benchmarked implementation used.
- FIMO skips N bases when scoring, so a window made only of N scores exactly 0, and long low-information motifs
  report such windows as hits. These hits are dropped; all other hits, including real hits with a score of 0 or
  below, are kept. This departs from the benchmarked implementation, which counted a (peak, TF) as no hit whenever
  its best score was 0 and so also dropped real negative-score hits in peaks next to N runs.
- The ten clusterings make preprocessing slower: the whole preprocessing notebook of the gastrulation example
  (56,861 cells, 64 cores) took 53 minutes, against about 13 minutes with v2.0.0.
- The v2.0.0 functions (`compute_motif_scores`, `metacell_correlation_matrices`, `compute_in_silico_chipseq`) are
  unchanged, for scripts that need the v2.0.0 binding. Switching a fit to the new binding needs preprocessing and
  phase 2 again, but not phase 1.

### Faster shuffled background for the chromatin networks

- `compute_atac_grn_activator_with_significance` and `compute_atac_grn_repressor_with_significance` now
  compute the shuffled background one chromosome at a time. Gene-peak links never cross chromosomes, so each
  chromosome's genes and peaks are multiplied with the matching rows of the same shuffled binding matrix, and
  the percentile is taken on the GPU with `torch.quantile`. The shuffles are drawn exactly as in v2.0.0, and
  the backgrounds are no longer copied to host memory.
- Values differ from v2.0.0 only by floating-point rounding, so a link can change only if it lies within that
  rounding of the cutoff.
- The chromosome of each peak is read from the ATAC data (`var['chr']`, or peak names written `chr:start-end`
  or `chr-start-end`); `load_downstream_model` provides it, and `peak_chromosomes` can pass it directly. Every
  label is its own block, including contigs such as `chr1_KI270706v1_random` and names without `chr`.
- If a peak's chromosome cannot be read, a gene is linked to peaks on two chromosomes, or a background contains
  NaN, the v2.0.0 computation is used, with a warning.

### Changed defaults

- The phase-2 L1 penalties `l1_penalty_grn_activator` and `l1_penalty_grn_repressor` default to 5e-3. To restore
  v2.0.0, set both to 5e-5. `freeze_phase1_in_grn` stays `true`.
- Phase 2 still skips almost-unused topics by default (`grn_topic_selection: "live"`), but a topic now counts as
  collapsed only if its weight stays below 0.05 in every training cell (was 0.1) and its mean weight is below
  `0.1 / num_topics` (unchanged); both conditions must hold. Which topics collapse depends on the cell types and
  conditions in the data; the lower default keeps more topics. To restore the v2.0.0 rule, set
  `grn_dead_topic_max_theta: 0.1`.
- Downstream plots and summaries keep a topic if its mean weight exceeds 0.05 in at least one cell type
  (`downstream.topic_activity_threshold`, was 0.07).
- Topic gene scores (`compute_topic_gene_matrix`, `gene_importance_scores`) have a spread floor: each gene is
  divided by `sqrt(variance + (sd_floor × s0)² + 1e-5)` instead of `sqrt(variance + 1e-5)`, where `s0` is the
  median gene standard deviation across the reference topics and `sd_floor` defaults to 0.5. Without it, a gene
  that scores near zero in every topic got a top score from a tiny difference: in the gastrulation example
  notebook, 14 of the 65 leading genes listed are detected in fewer than 1% of cells in every cell type (none with
  the floor), and 25 of 65 and 22 of 55 in two fits with other seeds (2 and 2 with the floor). Genes with low but
  cell-type-specific expression are kept. The scores also feed the gene-set test and the exported topic–gene
  table; the networks and TF scores do not change. To restore v2.0.0, pass `sd_floor=0`.
- `genes_user` and `tfs_user` are empty by default. The Python defaults used to add the gene Myt1l and the TFs
  Tbx5, Myt1l, Cdx2 and Sox2 when no JSON config was used; the supplied `scdori_config.json` already left both
  empty.
- Training length follows the number of optimizer steps. `max_scdori_epochs`, `phase1_patience`,
  `max_grn_epochs` and `grn_val_patience` default to `"auto"`: with `steps_per_epoch = ceil(training cells /
  batch size)`, the epoch limits are `max(1000, ceil(25000 / steps_per_epoch))` and the patience values
  `max(30, ceil(1000 / steps_per_epoch))` (phase 1) and `max(5, ceil(300 / steps_per_epoch))` (phase 2). The step
  budgets (`phase1_max_steps`, `phase1_patience_steps`, `grn_max_steps`, `grn_patience_steps`) are settings. For
  data of usual size this gives 1,000 epochs and a phase-1 patience of 30 (v2.0.0: 50); a few hundred metacells
  get several thousand epochs. An integer in the config overrides `"auto"`,
  and each phase saves the values it used (`training_length_phase1.json`, `training_length_grn.json`). To restore
  v2.0.0, set `phase1_patience: 50`, `grn_val_patience: 5` and both epoch limits to 1000.
- Downstream analysis recomputes the TF-expression gate from all cells, training and validation, with the fit's
  TF-expression rule, as v1.0.0 did. v2.0.0 reused the gate from the training cells saved after phase 2; phase 2
  itself is still trained with the training-cell gate. `downstream_run.json` records which cells the gate used.
  To use a saved gate, set `downstream.tf_expression_file`; `downstream.allow_tf_expression_recompute` is no
  longer used.
- A configuration object that lacks a setting now uses the current default for it. Before, such objects fell
  back to the v1.0.0 behaviour for `freeze_phase1_in_grn`, `grn_topic_selection`, `tf_expression_scaling` and
  `tf_expression_min_detection`. Configs read from JSON were not affected.

### New

- `grn_topic_selection: "shuffle_null"`: another way to choose the phase-2 topics. For each topic, its 100 training
  cells with the highest weight are shuffled within each cell (gene values across genes, peak values across peaks)
  and encoded again, 5 times with a fixed seed; the topic is kept if the 99th percentile of its weight in the
  training cells is above the 95th percentile of its weight in the shuffled cells. Cutoffs and decisions are saved
  in `grn_topics.json`. Topics that phase 1 leaves unused have weights near zero in every cell, and this comparison
  can keep them, so `"live"` stays the default.
- `phase2_train_nuisance` (default `false`): an experimental phase-2 option that also trains the RNA dispersion
  and the RNA library-size network, the only phase-1 terms the network fit depends on, with their own learning
  rate (`learning_rate_nuisance`, default `learning_rate_grn`). Topic activities stay fixed and cached.

### Fixed

- The gene-set test in the downstream notebook used decoupler's default background of 20,000 genes, although only
  the model's gene panel (4,000 genes in the example) is ranked, so every enrichment was overstated. It now passes
  `n_background` = the number of panel genes. The scores of all sets are now also saved
  (`topic_geneset_enrichment_all_sets.csv`); `topic_geneset_enrichment.csv` keeps only the sets shown in the map.

- `remove_mitochondrial_genes` now ignores case in the prefix as well as in the gene names. An upper-case prefix
  such as `"MT-"` used to remove no genes; the default `"mt-"` was not affected.

## v2.0.0

Fitted results differ from v1.0.0. Refit your data rather than comparing networks across versions.

### New defaults

| Setting | v1.0.0 | v2.0.0 | To restore the v1.0.0 setting |
| --- | --- | --- | --- |
| Phase 2 | phase 1 frozen by default but no caching | phase 1 frozen, features cached | `freeze_phase1_in_grn: false` |
| Topics fitted in phase 2 | all | topics used after phase 1 (`grn_topic_selection: "live"`) | `"all"` |
| TF-expression rule | `within_topic`, cutoff 0.1 | `mean_within_across`, cutoff 0.2 | `tf_expression_scaling: "within_topic"`, `tf_expression_clamp: 0.1` |
| Minimum TF detection | none | RNA seen in 2% of a topic's top cells (`tf_expression_min_detection: 0.02`) | `tf_expression_min_detection: 0` |
| Repressor L1 | 0 | `5e-5`, same as activators | `l1_penalty_grn_repressor: 0.0` |


These switches restore v1.0.0 settings, not v1.0.0 results, because the fixes below cannot be switched off. Explicit settings and existing checkpoints are respected.

### Fixed

- Regularization takes effect: L1 and L2 penalties were computed on detached tensors, so they never influenced training (#19).
- ATAC metacell means are converted from insertions to fragments once, not twice during training.
- A saved phase 1 no longer overrides newly preprocessed TF binding when phase 2 starts.
- A fresh install could not finish preprocessing on datasets above 8,192 cells: `pynndescent` 0.7 is not compatible with the supported `scanpy`, so it is now pinned below 0.7.
- The supplied settings now find the motif files that ship with scDORI, in `assets/motif_database`.
- A failed or interrupted genome download no longer leaves a file that later runs accept as finished.
- Default genome downloads work again. The host `hgdownload.cse.ucsc.edu` no longer matches its certificate, so downloads from it failed; the defaults now use `hgdownload.soe.ucsc.edu`, which serves the same files.
- Cells with no TF counts no longer produce NaN scores (#17).
- TF expression follows the model's float precision (#15).
- Fixed peak–gene masks and TF-binding priors stay frozen after reloading a checkpoint (#13).
- ATAC insertion and fragment counts are recorded and respected through save and reload (#20).
- `get_top_activators_per_topic` and `get_top_repressor_per_topic` no longer raise `KeyError` when TFs tie.

### Changed

- Phase 2 is about ten times faster: frozen phase-1 calculations are cached.
- TFs are ranked by the mean of two min-max scores on within-topic shares, as in the [paper](https://www.nature.com/articles/s41586-026-11177-0), per topic and per cell type. Cell-type maps are coloured by this score. The earlier z-scores are still saved.
- All notebooks read one shared `scdori_config.json`, and choose examples from your own data instead of fixed topic numbers.
- Gene-set enrichment needs a `.gmt` file for your species. v1.0.0 converted mouse gene names to human ones by changing case, which is not an orthologue mapping.

### Added

- Review after phase 1: training curves, cell map, topic use by cell type, and topic programs.
- `tf_expression_keep_tfs` and `tf_preview.look_up()` to keep a named TF and see why a TF was left out.
- `preprocessing.peaks_user` to retain chosen peaks, and a report of names that could not be included.
- `preprocessing.correlation_normalization` to choose the metacell values used for TF binding. The default is the v1.0.0 rule.
- `regions_for_link` to show which regions carry one TF–gene link, and public imports for the functions the notebooks use.
- [Best practices](docs/best_practices.md): how far results change between fits, and how to repeat a fit.

## v1.0.0

First release.
