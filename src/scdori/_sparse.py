"""Prepare sparse counts for repeated cell-row access."""

import logging
from time import perf_counter

import scipy.sparse as sp

logger = logging.getLogger(__name__)


def prepare_count_matrix(adata, *, use_csr=True):
    """Convert in-memory CSC ``.X`` to CSR once for cell minibatches.

    Counts, dtype, annotations, layers and ATAC count-mode metadata are preserved.
    Dense, CSR, other sparse formats and backed objects are returned unchanged;
    backed inputs are never materialized. Both SciPy sparse matrices and arrays
    are supported. Set ``use_csr=False`` to preserve the input representation.

    This modifies an ordinary AnnData object's ``.X`` in place and returns that
    object. For a CSC AnnData view, it returns an independent copy instead, to
    avoid modifying the parent; always use the return value. Copying a view also
    copies its other slots. Conversion temporarily holds both CSC and CSR sparse
    buffers in host RAM. The logged buffer sizes are not process peak memory.

    Call once before training or downstream encoding, never per minibatch.
    """
    if not use_csr or adata.isbacked:
        return adata
    counts = adata.X
    if not sp.issparse(counts) or counts.format != "csc":
        return adata
    nbytes = counts.data.nbytes + counts.indices.nbytes + counts.indptr.nbytes
    logger.info(
        "Converting count matrix %s from CSC to CSR for row access (%.1f MiB sparse buffers; "
        "temporary additional host memory required).",
        counts.shape,
        nbytes / 2**20,
    )
    start = perf_counter()
    converted = counts.tocsr()
    if adata.is_view:
        adata = adata.copy()
    adata.X = converted
    logger.info("Count matrix converted to CSR in %.3f seconds", perf_counter() - start)
    return adata
