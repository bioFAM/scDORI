.. _training_hyperparameters:

Hyperparameter configuration guide
==================================

This guide helps you choose feature-selection and training settings. For a first
analysis, use the three notebooks with the supplied settings. The training notebook
shows the topics after phase 1, so you can decide whether the learned programs are
useful before spending time fitting their networks.

Settings are in the ``preprocessing`` and ``training`` sections of
``scdori_config.json``. The :doc:`notebook_settings` page lists their names and the
steps to rerun for each choice. :doc:`best_practices` covers checking and repeating
a fit once you have chosen them.

Input feature selection
-----------------------

``num_genes``, ``num_tfs`` and ``num_peaks`` set how many features enter training.

- Although you may increase the number of input features, your final selection
  should reflect the available GPU memory. A GPU with about 12--15 GB supports
  about 4,000 genes, 70,000 peaks and 300 TFs. Larger feature sets need
  proportionally more memory.
- With a larger GPU you may expand the feature set for finer granularity.

**Manual inclusion of specific genes, TFs and peaks**

- Force the inclusion of genes, TFs or peaks with ``genes_user``, ``tfs_user`` and
  ``peaks_user`` (all empty by default). This is useful for known regulators or
  candidate markers from prior work or literature.
- Ensure that all TFs in ``tfs_user`` have entries in your motif database. TFs
  without motif matches are excluded during motif scanning. The preprocessing
  report lists any names that could not be included.

Motif file settings
-------------------

- The motif database must be in MEME format. The example uses cisBP, but you may
  use other databases with the same structure.
- Motif scanning uses `FIMO <https://meme-suite.org/meme/doc/fimo.html>`_
  :cite:p:`Grant_2011` through `tangermeme <https://tangermeme.readthedocs.io/en/latest/>`_
  :cite:p:`Schreiber_2026`.
- ``motif_match_pvalue_threshold`` controls the stringency of motif matching. A
  stricter cutoff (lower p-value) gives fewer but more confident matches.

TF–peak correlation: empirical filtering
----------------------------------------

- scDORI computes Pearson correlations between TF expression and peak accessibility
  at the metacell level to improve TF–peak specificity (in-silico ChIP-seq).
- Since v2.1.0, each correlation is weighted by the strength of the best motif
  match and averaged over ten Leiden clusterings; the values are not rescaled. See
  :doc:`notebook_settings` for the details.
- ``correlation_percentile`` prunes weak associations.
- ``correlation_normalization`` sets which metacell values are correlated. The
  default is the mean of each cell's log-normalized RNA and the mean ATAC fragment
  count.
- While not yet natively supported, advanced users can assign different correlation
  thresholds for activators (positive) and repressors (negative) in the source code.

Metacell construction
---------------------

- Metacells are formed by high-resolution Leiden clustering on Harmony-corrected
  PCA embeddings.
- For robust correlation analysis, aim for at least 50 metacell clusters. The
  example has 134.
- With fewer than 50 clusters, consider increasing ``leiden_resolution``.

Enhancer–gene distance window
-----------------------------

- scDORI links peaks to genes within a fixed genomic window around each gene, set by
  ``window_size``. By default, this is 150 kb on each side for human and 80 kb for
  mouse. A value you set yourself is kept.
- Expanding the window may increase sensitivity but also raises the chance of
  spurious associations.
- Reducing it favours specificity but may exclude distal regulatory elements.

**Gene annotation**

- Default GTF files are used for ``hg38`` and ``mm10``.
- To use your own annotation, set ``gtf_url`` to a URL.

Choosing the number of topics
-----------------------------

- A recommended starting point is ``#topics = expected number of cell types + 10``.
- Topics with low activity are suppressed by softmax normalization, and almost-unused
  topics are skipped in phase 2 by default.
- More topics can model finer programs but use more memory, because each topic has
  its own network.
- This is a starting point rather than a rule: related cells can share a program, and
  a cell can use several programs. When comparing topic counts, look at validation
  loss, the gene programs, rare populations and agreement between independently
  trained fits.

Regularisation
--------------

- L1 and L2 penalties apply to the topic–peak, peak–gene and topic–TF matrices, and
  to the 3D topic–TF–gene networks.
- Higher regularisation promotes sparsity and easier biological interpretation.
- The phase-2 activator and repressor L1 penalties both default to ``5e-3`` (v2.0.0:
  ``5e-5``). Consider different strengths for the two signs. To compare phase-2 penalties,
  reuse the same saved phase 1 and write each new fit to its own output folder.

Training length
---------------

Training length is set from the number of optimizer steps (minibatches), because an
epoch is a very different amount of training for a few hundred metacells and for tens of
thousands of cells. With the default ``"auto"`` settings:

.. code-block:: text

   steps_per_epoch    = ceil(training cells / batch size)
   max_scdori_epochs  = max(1000, ceil(phase1_max_steps / steps_per_epoch))         phase1_max_steps = 25,000
   phase1_patience    = max(30, ceil(phase1_patience_steps / steps_per_epoch))      phase1_patience_steps = 1,000
   max_grn_epochs     = max(1000, ceil(grn_max_steps / steps_per_epoch))            grn_max_steps = 25,000
   grn_val_patience   = max(5, ceil(grn_patience_steps / steps_per_epoch))          grn_patience_steps = 300

Patience counts validation checks; with ``eval_frequency`` above 1, the step budget is
divided by ``steps_per_epoch * eval_frequency``. Training stops after ``patience + 1``
checks without improvement. The step budgets come from phase-1 fits on 10 of our
benchmark datasets (batch size 128, 33 to 86 steps per epoch), which stopped after
about 4,000 to 20,000 steps. Resulting values with batch size 128:

.. code-block:: text

   training cells  steps  max_scdori_epochs  phase1_patience  max_grn_epochs  grn_val_patience
        400      4   6,250    250   6,250    75
      4,224     33   1,000     31   1,000    10
     11,008     86   1,000     30   1,000     5
     45,488    356   1,000     30   1,000     5

The first row is a few hundred metacells; the last is the example (56,861 cells, 80%
for training). An integer in the config replaces any of these values, and the step
budgets are settings too. ``resolve_training_length(config, steps_per_epoch, phase)``
gives the values for your data before training, and each phase writes them to ``training_length_phase1.json`` or
``training_length_grn.json`` in its output folder. On the example, with the earlier
fixed patience of 50, phase 1 stopped after 131 and 212 epochs for two seeds, and
phase 2 stopped after 118 to 197 epochs.

A best checkpoint near the epoch limit is a reason to inspect the curve. If it is still
improving substantially, a longer run may help. Small late improvements alone do not
show that the biology is better.

Batch size
----------

Changing batch size can change the fit as well as memory use. Keep it fixed when
comparing other settings. The model sums ATAC losses over the batch but averages the
RNA and TF losses over cells, so batch size changes their relative influence.

Faster phase 2
--------------

The original defaults already fixed phase-1 weights. The fully frozen mode also fixes
training-dependent behaviour such as dropout and batch normalisation, so calculations
can be cached. New fits use this mode and caching, and skip almost-unused topics. All
three choices are configurable.

Caching is rebuilt when phase 2 starts. It needs memory for the stored cell topics and
TF–gene factors, but avoids repeating those calculations at each batch. On the example,
a cached phase-2 epoch took about 20 to 25 seconds, against about 54 seconds for a
phase-1 epoch and about 6 minutes for an uncached phase-2 epoch. Checkpoints save model
weights and supporting metadata, rather than a complete optimizer state for exact
continuation of an interrupted run.

TF expression
-------------

The default compares each TF with other TFs within a topic and with its own expression
across topics, averages the scaled values, and sets values below ``0.2`` to zero. A TF
must also have RNA in 2% of the topic's top cells (``tf_expression_min_detection``). The
original rule remains available as ``within_topic`` with cutoff ``0.1``. Changing this
rule needs a new phase-2 fit; completed fits reuse their saved expression values
downstream.
