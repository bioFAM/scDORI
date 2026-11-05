.. _welcome:

Welcome to scDORI
=================

scDORI uses paired single-cell RNA and ATAC data to learn gene programs and their
regulatory networks. Each cell can use several programs, called topics. It is described in
`Saraswat et al., Nature 2026 <https://www.nature.com/articles/s41586-026-11177-0>`_; please cite this paper if you use scDORI.

The analysis follows three notebooks:

1. :doc:`notebooks/preprocessing` selects features and prepares TF-binding and
   peak–gene information.
2. :doc:`notebooks/training` learns the topics, shows their cell and gene patterns,
   and fits the regulatory networks.
3. :doc:`notebooks/downstream` explores the programs, TF targets, repressors and
   regions linked to genes.

Use the same ``scdori_config.json`` for all three. The :doc:`installation` page
explains setup and the required inputs. The notebooks explain each analysis step
where it is used.
