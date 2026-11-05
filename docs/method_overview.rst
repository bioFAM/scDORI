.. _method_overview:

How scDORI learns topics and networks
=====================================

scDORI combines RNA and ATAC measurements from the same cells. It learns topics:
patterns of accessible DNA and gene expression that cells can share. Each cell
uses a mixture of topics, allowing a program to vary across a population rather
than assigning every cell to a single group. The paper describes the model in full
:cite:p:`Saraswat_2026`.

.. image:: _static/Supp_fig_2_1_v6_crop.png
   :align: center
   :width: 100%
   :alt: The two training phases and four modules of scDORI

Phase 1: learn the topics
-------------------------

The encoder converts each cell's RNA and ATAC profiles into topic weights. The
weights sum to one for each cell. Three parts of the model then use those topics
to describe the measured data:

- **Module 1** learns which accessible peaks belong to each topic.
- **Module 2** connects peaks to nearby genes and uses predicted accessibility
  to describe gene expression.
- **Module 3** learns the TF-expression pattern of each topic.

Training begins with ATAC and TF expression, then adds the remaining RNA
expression. The notebook shows cell maps, topic use and gene/TF programs after
this phase. Cell-type labels help interpret the plots but are not required to
fit these topics.

Phase 2: learn regulatory networks
----------------------------------

**Module 4** combines the learned topics and peak–gene weights with the predicted
TF-binding information prepared during preprocessing. It learns activating and
repressive TF–gene contributions for each topic and uses them to describe RNA
expression.

Phase-1 weights stay fixed by default. Their calculations are cached for reuse,
and almost-unused topics are skipped to make training faster. The notebook lets
you retain all topics instead.

TF expression contributes to each topic's network. The default averages two
scaled comparisons: a TF against other TFs in that topic, and the same TF across
topics. Values below ``0.2`` are set to zero, and a TF must also have RNA in 2% of the
topic's top cells. This can retain lower-expression TFs that are more specific to a
program. The same expression values are used for
activating and repressive contributions.

After training
--------------

The downstream notebook explores gene and peak programs, compares chromatin
support with shuffled binding scores, and combines this support with the learned
TF–gene weights. It then shows TF rankings, targets and gene-linked regions.
A TF can have different targets in different topics; the notebook supports
comparing these contexts as well as following one topic in detail.
