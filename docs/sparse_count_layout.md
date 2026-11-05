# Sparse counts and cell minibatches

Training repeatedly selects rows of cells from RNA and ATAC counts. CSC stores
counts by column and can make these reads expensive on large single-cell data.
CSR stores counts by row. Changing CSC to CSR preserves count values and dtype;
it does not change ATAC units, normalization, cell/feature order, model parameters
or the training objective. Dense inputs remain dense.

Dense minibatch buffers now use a consistent row-major memory order for CSC,
CSR and dense sources, including cached RNA targets and topic TF-expression
calculations. Without this, identical counts can take different floating-point
reduction/kernel paths. Paired tests compare CSC and CSR within the updated
implementation; they do not promise bitwise reproduction of historical training
trajectories or deterministic results across different GPU/software versions.

`load_scdori_inputs(config)` now converts in-memory CSC `.X` to CSR once when
loading each dataset. Both the training and downstream notebooks use this
loader. `save_processed_datasets` also saves aligned CSC inputs as CSR by default,
so newly prepared data avoid the conversion on later loads. The preprocessing
function works on copies and does not change the caller's input objects.

## Configuration and memory

The shared notebook configuration has two independent settings:

```json
{
  "preprocessing": {"count_matrix_csr": true},
  "training": {"count_matrix_csr": true}
}
```

The first controls the format saved by the preprocessing notebook; the second
controls loading for both training and downstream analysis. Set both to `false`
to preserve the existing format throughout the notebook workflow. Old Python
config objects without this attribute still work and default to conversion.
For direct preprocessing calls, pass `count_matrix_csr=False` to
`save_processed_datasets` to preserve layout. Opting out does not convert an
already-CSR file back to CSC.

Conversion temporarily holds both sparse representations in CPU memory. For
float32 values and int32 indices, the new representation needs approximately
`8 * nnz + 4 * (n_cells + 1)` bytes, plus working memory. Actual requirements
depend on the value/index dtypes. The log reports the input sparse buffer size
and conversion time, **not measured peak RAM**. Check memory headroom for large
ATAC matrices. No whole-matrix dense copy is created. Saving aligned data already
involves copies, so the total saving-process peak can be larger still.

## Direct AnnData users

Call the public helper once before training or repeated downstream encoding:

```python
from scdori import prepare_count_matrix

rna = prepare_count_matrix(rna)
atac = prepare_count_matrix(atac)
# Now pass these objects to train_scdori_phases, train_model_grn,
# get_latent_topics or get_tf_expression as usual.
```

The helper modifies `.X` in place on ordinary in-memory AnnData objects and
returns the same object. A CSC AnnData view instead produces an independent
copy, including its other slots, so always assign the return value. Parent
objects are left intact. Both CSC sparse matrices and CSC sparse arrays are
supported. Existing CSR, dense and other sparse formats are returned unchanged.
Layers and `.raw` are not converted.

Backed objects are returned unchanged without reading `.X`; their existing row
reader remains in use. This optimization is for in-memory CSC data and makes no
backed-mode speed claim. The helper is not called inside minibatch loops.
Direct callers can omit preparation or pass `use_csr=False` to preserve layout.
