"""Shared ATAC count conventions for preprocessing and model inputs."""

import numpy as np
import scipy.sparse as sp

ATAC_COUNT_MODE_KEY = "scdori_atac_count_mode"


def resolve_atac_count_mode(adata, mode=None):
    """Use an explicit convention, then saved metadata, then legacy insertions."""
    if mode is None:
        mode = adata.uns.get(ATAC_COUNT_MODE_KEY, "insertions")
    if mode not in ("insertions", "fragments"):
        raise ValueError("atac_count_mode must be 'insertions' or 'fragments'")
    return mode


def as_fragment_counts(matrix, mode):
    """Convert insertions on a copy; preserve fragment means and sparse storage."""
    if mode not in ("insertions", "fragments"):
        raise ValueError("atac_count_mode must be 'insertions' or 'fragments'")
    if mode == "fragments":
        return matrix
    if sp.issparse(matrix):
        result = matrix.astype(np.result_type(matrix.dtype, np.float32), copy=True)
        result.sum_duplicates()
        result.data = (result.data + 1) // 2
        result.eliminate_zeros()
        return result
    values = np.asarray(matrix)
    values = values.astype(np.result_type(values.dtype, np.float32), copy=False)
    return (values + 1) // 2
