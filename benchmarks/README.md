# Benchmarks and checks

These scripts measure training and check the notebooks. You do not need them to run scDORI.

| Script | What it does |
| --- | --- |
| `validate_notebook_workflow.py` | Runs the three notebooks on small synthetic data. Continuous integration uses it. |
| `check_notebook_links.py` | Checks the relative help links reachable from the three notebooks, offline. |
| `benchmark_frozen_grn.py` | Compares cached and full phase 2 on processed real data, from the same phase-1 weights. |
| `benchmark_sparse_layout.py` | Measures CSC and CSR row reads and memory. See [sparse count layout](../docs/sparse_count_layout.md). |

## Check the notebooks

The check needs BEDTools on your `PATH`, or pass its location with `--bedtools`.

```bash
python benchmarks/validate_notebook_workflow.py --out notebook-check
```

It exercises preprocessing, training, permutation filtering and regulon export. It shows that the
software works together. It says nothing about the biology of a real dataset, and it does not
test downloading genomes or other remote resources.

## How the phase-2 cache works

By default phase 1 stays fixed during phase 2, so its per-cell calculations are computed once. To
compare with a full forward pass under the same freezing, set `cache_frozen_grn` to false. To
fine-tune phase 1 as v1.0.0 did, set `freeze_phase1_in_grn` to false.

The cache holds cell topics, RNA library factors and constant reconstruction loss terms on the CPU.
It is indexed by cell ID, so shuffled batches and train and validation subsets stay aligned. The
ATAC-derived activator and repressor factors stay on the model's device. Observed RNA is still
loaded as the training target. For float32, the cache takes about
`4 * (N * (K + 4) + 2 * K * T * G)` bytes for `N` cells, `K` topics, `T` TFs and `G` genes.

Caches exist only inside a training call. They are rebuilt after loading a checkpoint or in a new
call, and they are not saved with the model.
