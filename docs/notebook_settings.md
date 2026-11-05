# Settings for your analysis

All three notebooks read **`scdori_config.json`**. For a new project, copy
`configs/scdori.example.json` to that name at the repository root. The file has
three sections: `preprocessing`, `training` and `downstream`. You can start with
the supplied analysis settings and change them as you learn more about your data.

## Keep genes, TFs and peaks you choose

Add names to the `preprocessing` section if you want to keep particular features
even when they are not highly variable:

```json
"genes_user": ["Gypa", "Alas2"],
"tfs_user": ["Tal1", "Gata1"],
"peaks_user": ["chr1:1000-1200"]
```

These mouse genes and the placeholder peak coordinate are examples; use names
from your own data. Genes need to survive the initial protein-coding and
mitochondrial filters. TFs also need matching motifs. Put a TF in `tfs_user` if
you want its network to be fitted; listing it only as a gene does not ensure this.
Peaks use exact names from the ATAC file.

Requested features take priority over the feature-count targets. Missing names
are listed in the preprocessing report. A requested peak outside the gene windows
can contribute to the ATAC model, while distance settings still determine which
genes it can be linked to. Changing these lists requires preprocessing and
training again.

## Distance between peaks and genes

`preprocessing.window_size` sets how far from a gene, on each side, a peak can be linked to it. By default this
is 150 kb for human and 80 kb for mouse and other species. A value you set yourself is kept. A wider window can
find more distal regions, but it adds spurious links and uses more memory. Changing it means repeating
preprocessing and training.

## Keep the generated folder with your fit

A phase 1 fit only works with the peaks it was trained on, and a rerun of
preprocessing does not always select the same peaks, because the clusters behind
peak selection can change. In the example data, runs limited to 8 or 16 threads
changed the clusters and up to 817 of 90,000 peaks, and two 8-thread runs differed
from each other. Runs using all 64 cores selected the same clusters and peaks. So
keep `data/generated` with your fit rather than regenerating it.
`notebook_config.json` in that folder records the thread settings used, but those
settings alone may not reproduce a run.

## Values used for TF binding

`preprocessing.correlation_normalization` sets which metacell values are correlated
to predict TF binding:

| Setting | RNA | ATAC |
| --- | --- | --- |
| `"mean_log"` (default) | Mean of each cell's log-normalized counts | Mean fragment counts |
| `"none"` | Mean raw counts | Mean fragment counts |
| `"log_normalized"` | Summed counts, scaled and log-transformed | Summed counts, scaled and log-transformed |

These settings predict partly different binding sites. Tested against measured
ChIP-seq on the example, none was clearly better. Changing this setting needs
preprocessing and phase 2 again, but not phase 1.

## Before training

The most useful settings in the `training` section are:

| Setting | What it changes |
| --- | --- |
| `num_topics` | Number of programs the model can learn; the template starts with 40. |
| `batch_size_cell` | Number of cells processed together; smaller batches can reduce memory use. |
| `learning_rate_scdori` | How large the parameter updates are during phase 1. |
| `learning_rate_grn` | The corresponding update size for phase 2. |
| `max_scdori_epochs`, `max_grn_epochs` | Maximum passes through the training cells in each phase. |
| `random_seed` | Starting randomness for a new fit. A new seed with a new phase 1 gives new topic numbers and reorders top links; see [Best practices](best_practices.md). |

A recommended starting point is `num_topics` = expected number of cell types + 10
(see the [hyperparameter guide](training_guide.rst)). Topics need not match cell types
one for one. Look at validation loss and the cell and gene patterns when deciding
whether to try another number. Batch size
can affect the fit as well as memory use. Compare one change at a time, keeping
the data and validation cells the same.

For a new fit, choose new `weights_folder_scdori` and `weights_folder_grn`
folders, set `phase1_checkpoint` to `null`, and restart the kernel.

## Look at phase 1 before continuing

**Run All** normally trains both phases and shows the phase-1 plots between them.
To look first, set `RUN_PHASE2 = False` near the top of training.

The plots show training curves, how much each topic is used, a cell map, topic
use by cell type, and genes and TFs in one topic. Change `TOPIC_TO_VIEW` to inspect
another program. A low average topic weight can describe a rare population, so
look at the cell-type patterns as well.

When ready, uncomment `RUN_PHASE2 = True` under **Decide whether to continue** and run
from that cell onwards. Phase 1 is reused. To try different phase-1 settings,
leave it false, update your config and start a new fit as described above.

You can omit the plots with `INSPECT_PHASE1 = False` if you have already reviewed
the fit. To return to a saved phase 1, put its checkpoint path in
`training.phase1_checkpoint`; keep its saved training split and input-order files
alongside it.

## Make phase 2 faster

The default `grn_topic_selection="live"` skips topics that were almost unused
after phase 1. This reduces the number of networks to fit. Topic numbers stay the
same; phase 1 is unchanged. Set it to `"all"` to fit every topic.

A topic is flagged almost unused only when its highest weight is below `0.1`
and its average weight is below `0.1 / num_topics` in the training cells.
A topic shared weakly across many cells can therefore remain included.

The phase-1 weights stay fixed by default, as in the original workflow. The new
code also caches their calculations. `cache_frozen_grn=false` turns caching off
if you want a comparison. Skipping topics needs `freeze_phase1_in_grn=true`.
The older training path is available with `freeze_phase1_in_grn=false` and
`grn_topic_selection="all"`; its `update_*_in_grn` flags control which phase-1
weights can change.

## TF expression in phase 2

For each topic, scDORI averages TF expression in the cells that use that topic
most. It then compares each TF in two ways:

1. With other TFs in the same topic.
2. With its own expression in other topics.

Each comparison is scaled from 0 to 1. The default takes their mean, then sets
values below `0.2` to zero. The remaining values multiply TF contributions to
the network. This helps include lower-expression TFs that stand out in a
particular program, alongside more abundant TFs.

These are the new defaults in `training`:

```json
"tf_expression_scaling": "mean_within_across",
"tf_expression_clamp": 0.2,
"tf_expression_min_detection": 0.02,
"tf_expression_keep_tfs": []
```

The cutoff was used in the gastrulation example; you can compare other values
for your data. A lower cutoff allows more contributions. To use the earlier
rule, set `tf_expression_scaling="within_topic"`, `tf_expression_clamp=0.1` and
`tf_expression_min_detection=0`. Explicit values in an existing config are kept.

### Minimum detection: was this TF's RNA seen at all?

The two comparisons above are relative. A TF with almost no measured RNA can
still score highly in its best topic, because it is compared with itself. So
scDORI also asks a direct question: in how many of this topic's top cells did we
see at least one RNA count of this TF?

`tf_expression_min_detection` sets the answer it requires, as a fraction of those
cells. With the default `0.02` and `cells_per_topic=200`, a TF must be seen in at
least 4 of the 200 cells. Below that, it is left out of that topic's network. It
can still contribute to any other topic where it passes. Set `0` to turn this off.

This rule asks about evidence, not biology. Low detection does not show that a TF
is absent: many TFs have few transcripts, and single-cell RNA misses most of them.
Check the TFs you care about, especially if:

- **The cell type is rare.** With fewer cells than `cells_per_topic`, a topic's top
  cells must include other cell types, which lowers the share. Increasing
  `cells_per_topic` does not help; it makes this worse.
- **The TF is known to matter from other evidence.** In the gastrulation example,
  Noto is seen in 63% of the 132 notochord cells, but in at most 1% of the top
  cells of the topics it would otherwise enter, so this rule removes it.

To keep a TF regardless, add it to `tf_expression_keep_tfs`, for example
`["Noto"]`. TFs requested with `tfs_user` in preprocessing are also always kept.
A kept TF skips only this rule; it still has to pass the expression cutoff. The
TF preview in the training notebook lists every TF left out, and
`tf_preview.look_up("Noto")` shows the reason for one TF in each topic. This rule
needs measured expression (`tf_expression_mode="True"`) and is not used with
`"latent"`. If your inputs are metacells, the fraction counts metacells, not cells.

### Changing these settings

Changing the expression rule, cutoff or minimum detection needs another phase-2
fit. Use the same saved phase 1 and a new phase-2 output folder for a comparison.
Finished analyses use their saved expression values, so updating scDORI does not
change an existing network. Old fits without these saved values need their
original expression settings and training split when recomputing them; scDORI
versions before 2.0 had no minimum detection, so use
`tf_expression_min_detection=0` for those fits.

The expression comparison uses all fitted topics before skipped topics are set
to zero. A constant row or column contributes zero to its comparison. The same
expression values apply to both activator and repressor contributions.

## Regularisation

Regularisation discourages large model weights. In phase 2, both
`l1_penalty_grn_activator` and `l1_penalty_grn_repressor` default to `5e-5`.
You can compare smaller or larger values from the same saved phase 1.

Phase-1 penalties act on the topic–peak, peak–gene and topic–TF weights. They
are different from the phase-2 penalties. Smaller weights do not necessarily
mean fewer topics are used. The optional `theta_entropy_weight` and
`usage_balance_weight` remain zero by default; a first analysis does not need them.

## Choose what to explore downstream

For the completed gastrulation fit, the detailed example uses these entries in
`downstream`:

```json
"example_celltypes": ["Erythroid3"],
"example_topics": [20],
"example_tfs": ["Gata1"],
"example_gene": "Gypa"
```

Topic numbers belong to a particular fit. For your data, choose from the topic
plots, or leave the example lists empty and the gene `null` to select examples
from the fit. One topic makes a useful worked example. Two can show how the same
TF's targets differ between programs; add a second topic to `example_topics`
when it has relevant links. This only needs downstream to run again.

For a broader comparison in the same example, use
`example_celltypes=["Erythroid3", "Endothelium", "Cardiomyocytes"]` and
`example_topics=[]`. Each cell type then gets its two strongest topics. Use
`example_tfs=["Gata1", "Fli1", "Mef2c"]` to compare these TFs; the detailed
target panels follow the first TF with links. Put another TF first to follow it.

The matrix section in downstream gives labelled
topic–gene, topic–peak and peak–gene tables, plus short queries for a topic's
activators or repressors, its network, and a gene's linked regions in a chosen
cell type. These queries reuse the fit and do not require training again.

You can supply a JSON cell-type colour map through `downstream.celltype_palette`,
for example `{"Erythroid3": "#e64b35"}`. The same colours are used in training
and downstream plots.

The `topic_activity_threshold` starts at `0.07`: keep a topic when its mean weight
exceeds this in at least one cell type. This is for choosing plots, not a
significance test. Set it to `null` to include all topics retained for phase 2.
`exclude_topics` allows manual exclusions. Changing this set can change relative
scores when `scale_selected_topics=true`.

The TF section separately compares contributions within topics, across topics
and their mean. Activators and repressors have their own scores. These describe
the fitted networks, rather than the expression values used during training.
`TF_DISPLAY_TOPICS` selects rows after scaling, so it can show one or two topics
without changing their scores. Full score tables are saved in `tf_score_views/`.
The tables of leading activators and repressors per cell type use the same mean
score, across TFs and across cell types.

### Name topics with gene sets

Downstream can compare each topic's leading genes with published gene sets, using
decoupler's over-representation test. Point `downstream.geneset_file` to a `.gmt`
file of gene symbols for your species, such as the cell-type signatures from
[MSigDB](https://www.gsea-msigdb.org/gsea/downloads.jsp) (`m8` for mouse, `c8` for
human). A tab-separated table with `geneset` and `genesymbol` columns also works.
Without this file the section is skipped.

```json
"geneset_file": "m8.all.v2026.2.Mm.symbols.gmt",
"geneset_name_contains": "DESCARTES_ORGANOGENESIS"
```

`geneset_name_contains` keeps only the sets whose name contains that text. Use it
to leave out unrelated tissues: the full mouse collection also holds adult sets,
such as uterus, which can lead topics in embryo data.

Read the result as a suggested label. It means a topic's leading genes overlap a
signature, not that the topic is that cell type. In the gastrulation example,
erythroid and cardiac topics get clear labels, while one signature, ependymal
cells, leads several unrelated topics. Those organogenesis sets also come from a
later stage than gastrulation. Most MSigDB sets are free under CC BY 4.0, with
further restrictions on some; see the
[licence terms](https://www.gsea-msigdb.org/gsea/msigdb_license_terms.jsp).

## Network comparisons with a shuffled background

`downstream.activator_alpha=0.05` keeps activator chromatin support above the 95th
percentile of the shuffled background; repressors use the opposite tail with
`repressor_alpha=0.05`. These cutoffs are not false-discovery rates for the final
networks.

`training.num_permutations` controls how many shuffles to use. More shuffles take
longer but can make the comparison more stable. Set `downstream.permutation_seed`
to an integer for repeatable comparisons. Use a new `downstream.grn_output` folder
when comparing settings, then rerun downstream; training is not needed.
