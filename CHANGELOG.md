# Changelog

All notable changes to this project will be documented in this file.

## v2.0.0

Fitted results differ from v1.0.0. Refit your data rather than comparing networks across versions.

### New defaults

| Setting | v1.0.0 | v2.0.0 | To restore the v1.0.0 setting |
| --- | --- | --- | --- |
| Phase 2 | phase 1 fine-tuned | phase 1 frozen, features cached | `freeze_phase1_in_grn: false` |
| Topics fitted in phase 2 | all | topics used after phase 1 (`grn_topic_selection: "live"`) | `"all"` |
| TF-expression rule | `within_topic`, cutoff 0.1 | `mean_within_across`, cutoff 0.2 | `tf_expression_scaling: "within_topic"`, `tf_expression_clamp: 0.1` |
| Minimum TF detection | none | RNA seen in 2% of a topic's top cells (`tf_expression_min_detection: 0.02`) | `tf_expression_min_detection: 0` |
| Repressor L1 | 0 | `5e-5`, same as activators | `l1_penalty_grn_repressor: 0.0` |
| Peak–gene window for human | 80 kb | 150 kb on each side; mouse stays at 80 kb | `window_size: 80000` |

These switches restore v1.0.0 settings, not v1.0.0 results, because the fixes below cannot be switched off. Explicit settings and existing checkpoints are respected.

### Fixed

- Regularization takes effect: L1 and L2 penalties were computed on detached tensors, so they never influenced training (#19).
- The cells used to measure each topic's TF expression are chosen by identity, not by position in a shuffled data loader.
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
