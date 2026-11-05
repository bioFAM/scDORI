# Set up scDORI

The workflow uses three notebooks: **preprocessing → training → downstream**.
This page helps you install scDORI and prepare the settings they share. Once that
is done, each notebook explains the analysis as you go.

## Install and configure

Use Python 3.10–3.12; `python3 --version` shows yours, and if `python3.12` is not
available, replace it below with the Python you have. The commands below create a
separate environment so the installation does not change packages used by your
other projects.

```bash
git clone https://github.com/bioFAM/scDORI.git
cd scDORI
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Next, install PyTorch. For an NVIDIA GPU, choose your system and CUDA version in
the [PyTorch installer](https://pytorch.org/get-started/locally/) and run its command
in this environment. For CPU use:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If you have a GPU, check that PyTorch can use it. This should print `True`:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

A plain `pip install torch` can install a build too new for your NVIDIA driver. It
installs without any error and this check prints `False`. If so, choose an older CUDA
version in the PyTorch installer.

Then install scDORI and the packages used by all three notebooks:

```bash
python -m pip install -e '.[all]'
python -m pip check
```

Preprocessing also needs **BEDTools and wget**; it uses wget to download the
genome reference files. If you use conda, these can be installed with
`conda install -c conda-forge -c bioconda bedtools wget`.
On a shared cluster they may already be available as modules; ask your local
support team if `bedtools --version` is not found. These tools are not needed
when starting from already processed scDORI inputs.

For a new project, create your settings file and open Jupyter:

```bash
cp configs/scdori.example.json scdori_config.json
python -m jupyter lab
```

Keep your existing `scdori_config.json` when returning to a project. Open the
notebooks inside `docs/notebooks/`, using the Python environment installed above.

## Prepare your inputs

For preprocessing, provide:

- **RNA and ATAC `.h5ad` files** from the same cells, with raw counts in `.X`.
  Cell and feature names should be unique. Use gene symbols for RNA features and
  peak names such as `chr1:1000-1200` for ATAC.
- **Sample labels**, in a column of the cell metadata. Set `batch_col` to that
  column name in `training` and `batch_key` to the same name in `preprocessing`.
  Both RNA and ATAC should contain this column. For example, if it is called
  `donor`, use `preprocessing.batch_key="donor"` and `training.batch_col="donor"`.
  For a single sample, add a `sample` column with the same constant label to both
  inputs and keep the supplied settings. Cell-type labels in RNA `.obs['celltype']` are used for downstream
  plots, rather than training the model.
- **A TF motif database in MEME format** for your species. scDORI includes cisBP
  files for mouse and human in `assets/motif_database`, and the supplied settings use
  them. To use your own, its TF names should match your RNA gene symbols and its
  filename should be `<motif_database>_<species>.meme`, for example `cisbp_mouse.meme`.
- **A matching genome assembly.** The notebook downloads its configured reference
  files if they are not already in `genome_dir`.

Set these paths and species choices in the `preprocessing` section of
`scdori_config.json`. Relative paths start from the folder containing this file.
The first notebook cell prints which config it is using.

Cell calling and quality control should already be complete. Here, raw counts
means unnormalised counts for those retained cells, not unfiltered barcodes.

ATAC counts may represent fragments or insertions. Record the convention from
your data preparation in `atac.uns['scdori_atac_count_mode']` as `"fragments"`
or `"insertions"` before saving your inputs. Without this metadata, scDORI uses
its historical insertion-count convention.

## Run the three notebooks

| Notebook | What it does |
| --- | --- |
| [Preprocessing](notebooks/preprocessing.ipynb) | Select genes, TFs and peaks; prepare binding scores and peak–gene information. |
| [Training](notebooks/training.ipynb) | Learn topics, show their cell and gene patterns, then learn the regulatory networks. |
| [Downstream](notebooks/downstream.ipynb) | Explore topics, gene sets, TFs, their targets and linked DNA regions. |

Use the same config throughout. For a first run, you can keep the analysis
settings at their supplied values. On the gastrulation example (56,861 cells) with one
RTX 4090, preprocessing took about 13 minutes, training 3 to 4.5 hours and the
downstream notebook about 20 minutes. A run with shortened training used about 34 GB of
disk beyond the input files, so leave more than that free. A GPU with 12–15 GB of memory supports about
4,000 genes, 70,000 peaks and 300 TFs; larger feature sets need proportionally more. Training runs both phases by default; its
opening cells also explain how to pause after phase 1.

**Already have processed data?** Start with training. Set the five input filenames
and their directory in the `training` section: processed RNA, processed ATAC,
activator and repressor TF-binding matrices, and the peak–gene distance matrix.
Keep the original cell and feature order. TF genes must be labelled `TF` in RNA
`.var['gene_type']`, in the order used by the binding matrices.

**Using your earlier Python configs?** They still work when no JSON config is
selected: `src/scdori/pp/config.py` for preprocessing and
`src/scdori/_core/config.py` for training. Restart the notebook kernel after
editing those files.

## Common setup questions

- **GPU not available in training:** in this environment, run
  `python -c "import torch; print(torch.cuda.is_available())"`. If it prints
  `False`, check your GPU allocation and PyTorch installation.
- **A different config is loaded:** set `CONFIG_FILE` in the first notebook cell
  to your config's path.
- **A TF is missing:** check its RNA name and motif name. Include TFs of interest
  in `preprocessing.tfs_user` before preprocessing.
- **Choosing settings:** the [hyperparameter guide](training_guide.rst) covers features,
  topics, regularisation and training length.
- **Changing a setting:** restart the relevant notebook kernel and run from the
  top. Use a new output folder for a different fit. The notebooks explain which
  choices need training again and which only change plots.

Download or clone the full repository so notebook images and local help links
stay together. In JupyterLab, Markdown pages can be opened with **Markdown Preview**.
