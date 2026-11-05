# Reading the gene and TF scores

The downstream notebook looks at a fitted model in several ways. This page
explains what the scores mean when you want to look more closely.

## Topics and cells

A cell can use several topics. Its topic weights add up to one. Averaging these
weights within a cell type shows which programs that group uses most; it is not
the percentage of cells assigned to each topic.

The notebook initially keeps topics with mean weight above `0.07` in at least
one cell type. A rare population within a broad label can be missed by that rule.
You can lower the threshold, or set it to `None` in the notebook to include all
topics retained for phase 2.

## Genes in a topic

The gene score starts from the model's topic–peak and peak–gene weights. The
peak–gene links are used during training to predict RNA. We compare each gene across topics, then use a softmax to make a positive set of
scores for each topic. This is the normalization used in the original downstream
notebook. It helps genes specific to a program stand out against genes that score
highly in many topics.

The ranking describes a learned program, rather than measured expression.
Across-topic scaling can make a small difference look large, so gene expression
and the unscaled weights provide useful context.

## TFs within and across topics

We sum a TF's contributions to its targets in each topic, using the final networks after the
shuffled-background comparison. Activators and repressors are kept separate. For repressors,
larger magnitudes mean stronger negative contributions. Repressor scores use the reciprocal of a
region's accessibility in the topic, so regions closed in that topic count most: this is how scDORI
represents repression.

We then **divide each TF's contribution by the topic's total**, so a topic with more or larger links
does not dominate. From these shares come two comparisons, each scaled from 0 to 1, and their mean:

| View | How to read it |
| --- | --- |
| Within topic | How does this TF compare with the other TFs in this topic? |
| Across topics | Where does this TF contribute most, compared with its other topics? |
| Mean of both | The score used for ranking, as in the scDORI paper. |

A high value is a relative ranking, not a probability. A TF can score highly in a topic whose total
is very small, so check the original contributions, saved as `raw`, before reading a score as strong
regulation.

The z-score view from the original notebook is also saved. It uses the same shares, but compares them
across topics with a z-score instead of min-max scaling. Positive values mean above that TF's average
share; negative values mean below it. For repressors, a negative value means less repression than
average, not activation.

The tables in `tf_score_views/` save all views. The ranking-comparison tables show TFs that enter the
top list under one view but not another.

## TFs across cell types

The same score ranks TFs in each cell type: how a TF compares with the other TFs in that cell type,
and how that cell type compares with the TF's other cell types. It picks the leading TFs and colours
the two cell-type maps, so darker means a higher score. A dark band across many cell types marks a TF
that is strong but not specific. This applies the formula from the [paper](https://www.nature.com/articles/s41586-026-11177-0) to cell types.

The original notebook used a z-score for each TF across cells, averaged per cell type. It shows where
a TF is most active compared with its own average, but not how large its contribution is. It is saved
as `celltype_tf_activity_zscore.csv` and `celltype_tf_repressor_activity_zscore.csv`, and the single-TF
cell maps still use it.

A rare cell type may have no topic of its own. Check its row in `topic_celltype_mean.csv`: if it has no
strong topic, its leading TFs describe programs it shares with other cells.

## Showing fewer topics

Scores are calculated using a set of comparison topics. Selecting a smaller
set with `TF_DISPLAY_TOPICS` afterwards only changes the figure. Removing a topic
from the comparison itself can change the scores.

`SCALE_SELECTED_TOPICS=True` uses the topics selected by the activity threshold
and exclusions. `False` uses all topics retained for phase 2. Topics skipped in
phase 2 are excluded in either case. Keep this set the same when comparing plots.

One topic can illustrate a gene program. Two topics are useful for comparing a
TF's targets between cell states. Their numbers belong to the fit being explored,
so choose them from its topic plots.

## TF expression and network ranking are different

Training uses TF expression to decide how much each TF can contribute in each
topic. The new default averages comparisons within and across topics, then sets
values below `0.2` to zero. The same values apply to both signs.

The downstream scores above instead describe contributions in the finished
networks. They cannot restore a TF contribution excluded during training. To
change the expression rule, fit phase 2 again. Existing fits use their saved
expression values and original settings.

## The shuffled-background cutoff

The notebook asks whether accessible TF-binding sites near a gene provide
support that stands out from shuffled TF-binding scores. Activators use the
upper tail and repressors the lower tail.

An `activator_alpha` of `0.05` means the upper 95th percentile of that background;
`repressor_alpha=0.05` uses the opposite tail. These are background cutoffs,
not false-discovery rates for all final links. Smaller values are stricter.
More permutations and another random seed can help assess whether borderline
links are stable; [Best practices](best_practices.md) shows how to repeat a fit
and keep the links that recur. Both the cutoff and permutation count are saved
with the results.

## Regions linked to a gene

Peak–gene weights are shared across topics. The example chooses regions with
positive link weights and accessibility in the selected cells. The table shows
both quantities so weak links can be distinguished from stronger candidates.

The cell-level region score compares the selected regions with matched
background regions. It may be negative. Compare where its signal occurs with
the gene's expression pattern; the two maps have different units and colour scales.

## Exported score definitions

| Saved score | Calculation |
| --- | --- |
| Gene across-topic z-score | Centre each gene across topics and divide by `sqrt(population variance + 1e-5)`. |
| Gene downstream softmax | Apply softmax across genes within each topic after the above standardisation. |
| TF `raw` | Sum the final signed gene contributions, separately by sign; report positive repressor magnitudes. |
| TF `within_group_share` | Divide by total TF contribution in the topic, with `1e-8` for numerical stability. |
| TF `within_group_minmax` | Scale each topic's TF shares between its minimum and maximum. |
| TF `across_groups_minmax` | Scale each TF's shares between its minimum and maximum across reference topics. |
| TF `minmax_consensus` | Mean of the two min–max views. |
| TF `share_across_groups_zscore` | Standardise each TF's shares across topics using sample standard deviation, with `1e-8`. |

Constant min–max comparisons and single-topic z-scores are set to zero. The gene
z-score and softmax views give the same gene order within a topic.
