# Best practices

Advice for getting results you can rely on, once you have chosen your settings with the [hyperparameter guide](training_guide.rst). The numbers come from the gastrulation example (56,861 cells). They show what to expect, but your dataset may differ.

## Check the topics first

Review phase 1 before you train phase 2. The TF networks are built per topic, so they can only be as clear as the topics are.

A rare cell type may not have a topic of its own. Before you read its leading TFs, look at its row in `topic_celltype_mean.csv`. In the example, primordial germ cells reach at most 0.13 in any topic, and that topic belongs mainly to somitic and caudal mesoderm (0.20). Their leading activators are therefore Grhl2, Tfap2a and Hox genes, not germline factors. Treat such a row as a description of the topics the cells share, not as that cell type's regulators.

## Check that your TF is in the fit

A TF can only appear in the results if it is in the TF panel and passes the expression rule. In the training notebook, `tf_preview.look_up("Gata1")` shows, per topic, why a TF was kept or left out.

- **Left out for low detection:** add it to `TFS_TO_ALWAYS_KEEP`. You do not need to repeat preprocessing or phase 1.
- **Not in the panel:** add it to `preprocessing.tfs_user` and repeat preprocessing and training.

## Read links as a ranked list

Start with the strongest links, not with everything that passes the cutoff. Weak links are the least repeatable, as the numbers below show. The cutoff also lets through about 5% of links with no real support, so treat every link as a candidate (see [Reading the gene and TF scores](interpretation.md)).

## Repeat the fit before you report individual links

Training has random elements, so two fits never give exactly the same links. If specific links matter to your conclusions, fit more than once and keep the links that recur. There are two ways to repeat, and they answer different questions.

### Repeat phase 2 from your saved phase 1

This shows how much the links move because of training order alone. Topics stay the same, so you can compare links directly. For each repeat, change these settings and run the training and downstream notebooks:

| Setting | Value |
| --- | --- |
| `training.phase1_checkpoint` | Your saved phase 1 |
| `training.random_seed` | A new number for each repeat |
| `training.weights_folder_grn` | A new folder for each repeat |
| `downstream.grn_output` | A new folder for each repeat |
| `downstream.permutation_seed` | The same integer for every repeat |

In our run of the example, one repeat took about 94 minutes on one GPU: 77 for training and 17 for the downstream notebook with 1,000 permutations.

Then keep the links found in every run:

```python
from pathlib import Path
import numpy as np

runs = ["results/repeat1/downstream", "results/repeat2/downstream", "results/repeat3/downstream"]
networks = [np.load(Path(run) / "grn_activator__0.05.npy") for run in runs]  # topics x TFs x genes
present = np.stack([network != 0 for network in networks])

common = present.all(axis=0)
common_network = np.where(common, np.mean(networks, axis=0), 0)
print(f"{common.sum():,} links in all {len(runs)} runs; {present.any(axis=0).sum():,} in any")
```

Use `grn_repressor__0.05.npy` for repressive links. Three repeats of phase 2 from one phase 1, at the release settings, gave the following. A link is a (topic, TF, gene) combination.

| Measure | Activating links | Repressive links |
| --- | --- | --- |
| Links in one run | about 163,000 | about 112,000 |
| Overlap between two runs (Jaccard index) | 0.71–0.72 | 0.55–0.56 |
| Links in all three runs | 123,471 | 64,710 |
| Share of one run's links that are in all three | 76% | 58% |
| Share of the run's total link weight they carry | 99.7–99.8% | 99.7–99.8% |

Keeping the common links therefore removes many links but very little weight. The links that change are the weak ones, and the top links stay consistent:

| Strongest links in one run | Also in another run's group of the same size (activating, repressive) | A link at all in another run (activating, repressive) |
| --- | --- | --- |
| Top 100 | 95%, 91% | 100%, 100% |
| Top 1,000 | 96%, 96% | over 99.9%, over 99.9% |
| Top 10,000 | 97%, 98% | over 99.9%, 99.9% |
| Strongest 25% | 98%, 98% | 99.9%, 99.8% |
| Strongest 50% | 98%, 91% | 99.7%, 93% |
| All links | 83%, 71% | 83%, 71% |

The target lists shown in the notebooks behave similarly, but not every list is stable:

| Targets of a TF in a topic, share found in another run | Median (activating, repressive) | Least stable tenth of lists, at most (activating, repressive) |
| --- | --- | --- |
| Top 10 | 100%, 90% | 80%, 30% |
| Top 30 | 97%, 93% | 63%, 33% |

For example, Gata1 in the erythroid topic keeps 28–29 of its top 30 activating targets, and Nkx2-5 in the topic shared by cardiomyocytes and endothelium keeps all 30. Lists with almost no weight are the least stable: Sox10's repressive list in one topic has a total weight of about 2e-8, and only 8–12 of its top 30 targets recur. Do not read an order into near-zero weights. Marker TF support did not change between the three repeats, and test loss varied by 0.98 of about 1,936.

**If you can only run once,** rank by weight. Keeping the same number of strongest links from a single run recovered 92% of the common activating links and 82% of the common repressive links.

### Train phase 1 again as well

This is a stronger test, because the topics themselves are fitted again. Leave `training.phase1_checkpoint` empty, set a new `training.random_seed`, and use new folders for phase 1 and phase 2. One whole fit took 3 to 5 hours on one GPU.

We trained two new fits on the example, changing only the seed, with the same cells for training and validation. A seed is repeatable: training one seed twice on the same machine gave a byte-identical phase 1.

Topic numbers do not correspond between fits, so do not compare links topic by topic. Erythroid cells were Topic 37 in one new fit and Topic 14 in the other, and the fits selected 20 and 22 topics. Matching topics by which cells use them, 15 of 20 had a clear partner in the other fit. Compare TF–gene pairs across topics instead:

```python
tf_gene = np.stack([(network != 0).any(axis=0) for network in networks])  # TFs x genes
common_tf_gene = tf_gene.all(axis=0)
```

A new phase 1 changes the ranking much more than a new phase 2 does. Most top pairs still exist in the other fit, but few stay at the top:

| Strongest TF–gene pairs in one fit (activating, repressive) | Repeat of phase 2: also in the other run's top group | New phase 1: also in the other fit's top group | New phase 1: a link at all in the other fit |
| --- | --- | --- | --- |
| Top 100 | 95%, 91% | 7%, 8% | 77%, 78% |
| Top 1,000 | 96%, 96% | 19%, 21% | 81%, 80% |
| Top 10,000 | 98%, 98% | 35%, 39% | 81%, 79% |
| All pairs | 93%, 87% | 65%, 64% | 65%, 64% |

The other measures between the two new fits point the same way:

| Between the two new fits | Activating | Repressive |
| --- | --- | --- |
| Overlap of TF–gene pairs (Jaccard index; 0.86 and 0.77 between repeats of phase 2) | 0.48 | 0.47 |
| Five leading TFs per cell type, median shared | 3 of 5 | 2 of 5 |
| Top 30 targets of a TF in a cell type's main topic, median shared | 12 of 30 | 9 of 30 |

Broad results held better than individual links. Activity of 16 of 19 marker TFs ranked an expected cell type in its top three in all three fits. Gata1, Foxo3 and Klf1 led erythroid cells in all three, and Nkx2-5, Mef2c, Tbx5 and Esrrg led cardiomyocytes. Some cell types shared none of their five leading TFs.

**Fits can also differ in how clear they are.** In the second new fit, erythroid cells shared their main topic with parietal endoderm, and that topic's leading genes were not erythroid markers:

| | Example fit | New fit 1 | New fit 2 |
| --- | ---: | ---: | ---: |
| Erythroid3's weight in its main topic | 0.54 | 0.54 | 0.43 |
| Leading genes of that topic | Alas2, Spta1, Gypa, Hbb-bs, Hbb-bh1 | Pde6b, Alas2, Mptx2, Hbb-bt, Trhr2 | Pde6b, Chrne, Olfr569, Cnga1, Clca3a2 |
| Gata1's top 30 activating targets in the erythroid gene set (about 1 expected by chance) | 18 | 14 | 3 |
| Nkx2-5's top 30 activating targets in the cardiac gene set (about 1 expected by chance) | 18 | 17 | 15 |
| Validation loss of the phase-2 TF–gene model (lower is better) | 1,937 | 1,947 | 1,931 |

Three fits are too few to say how often this happens, but they lead to some practical advice:

- **If your conclusions rest on particular targets or one topic, train phase 1 with at least two or three seeds** and report what recurs.
- **Check the topics of each fit.** Do the cell types you care about have their own topic, and do its genes make sense? Do not choose by validation loss alone: here the lowest loss came from the fit with the shared erythroid topic.
- **Say how you chose a fit.** Keeping the fit that best matches what you expected can bias your results.
- **Treat the example as favourable, not typical.** The fit shown in these notebooks has the clearest erythroid program of the three. Its phase 1 was trained earlier, with the same settings and seed 3141.

## Common links are stable, not proven

All of these fits use the same cells and the same TF-binding input, so a link can recur in every fit and still be wrong. We have not tested whether common links agree better with ChIP-seq or perturbation data than the strongest links of a single run. Check links you intend to report against independent evidence.

## Compare fits only when their inputs match

Differences in preprocessing move links more than repeats do. With the same phase 1, detection floor and seed, changing only how TF binding was calculated gave an overlap of 0.45 for activating links and 0.36 for repressive links. Marker TF support and test loss did not change meaningfully, but individual links did.

Keep the generated folder with your fit, because repeating preprocessing may not give the same peaks (see [the settings guide](notebook_settings.md)).
