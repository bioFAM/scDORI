import numpy as np
import scipy.sparse as sp
import torch

from scdori._counts import as_fragment_counts, resolve_atac_count_mode


def read_count_rows(adata, indices):
    """Read count rows without annotation copies, preserving order and duplicates."""
    if adata.isbacked:
        # HDF5 needs sorted, unique indices; reconstruct the requested sequence.
        unique, inverse = np.unique(indices, return_inverse=True)
        return adata[unique, :].X[inverse]
    return adata.X[indices]


def create_minibatch(
    device,
    index_matrix,
    rna_anndata,
    atac_anndata,
    num_cells,
    tf_indices,
    encoding_batch_onehot,
    *,
    atac_count_mode=None,
):
    """
    Create a minibatch of required input tensors using integer indices of cells.

    Parameters
    ----------
    device : torch.device
        The device (CPU or CUDA) to which the data should be moved.
    index_matrix : torch.Tensor
        A 1D tensor containing integer indices of the cells in the minibatch.
    rna_anndata : anndata.AnnData
        AnnData object for RNA data. The .X matrix should contain RNA counts or expression values.
    atac_anndata : anndata.AnnData
        AnnData object for ATAC data. The .X matrix should contain accessibility counts.
    num_cells : np.ndarray
        A NumPy array (N x 1) indicating the number of cells represented by each row (if using metacells).
        For single-cell level data, this may be an array of ones.
    tf_indices : np.ndarray
        Indices corresponding to transcription factors (TFs) in the RNA AnnData.
    encoding_batch_onehot : np.ndarray
        A one-hot encoded matrix representing batch information for each cell (cells x num_batches).

    atac_count_mode : {"insertions", "fragments"}, optional
        Uses saved ``scdori_atac_count_mode`` metadata when available, otherwise
        legacy insertion counts. Fragment counts and means are preserved.

    Returns
    -------
    tuple
        A tuple containing:
        - input_matrix (torch.Tensor): Concatenated RNA and ATAC input of shape (B, g + p),
          where B is batch size, g is the number of genes, p is the number of peaks.
          Values are floats on the given device.
        - tf_exp (torch.Tensor): RNA expression values for TFs, shape (B, num_tfs).
        - library_size_value (torch.Tensor): Log-scale library sizes for RNA and ATAC, shape (B, 2).
        - num_cells_value (torch.Tensor): Number of cells per row in the minibatch (B, 1).
        - input_batch (torch.Tensor): One-hot batch-encoding, shape (B, num_batches).

    Notes
    -----
    - This function converts sparse arrays to dense if necessary.
    - Insertion counts use (x + 1) // 2; fragment counts are unchanged.
    """
    index_train = index_matrix.detach().cpu().numpy()
    mode = resolve_atac_count_mode(atac_anndata, atac_count_mode)
    # Avoid constructing AnnData views (and slicing annotations) in the hot path.
    atac_input = read_count_rows(atac_anndata, index_train)
    atac_input = as_fragment_counts(atac_input, mode)
    rna_input = read_count_rows(rna_anndata, index_train)

    if sp.issparse(atac_input):
        atac_input = atac_input.toarray(order="C")
    if sp.issparse(rna_input):
        rna_input = rna_input.toarray(order="C")

    # CSC defaults to column-major dense arrays, CSR to row-major. Use the same
    # memory order so reductions and tensor kernels do not depend on sparse layout.
    atac_input = np.ascontiguousarray(atac_input)
    rna_input = np.ascontiguousarray(rna_input)

    library_size_atac = atac_input.sum(axis=1).reshape(-1, 1) + 1e-8
    library_size_rna = rna_input.sum(axis=1).reshape(-1, 1) + 1e-8
    library_size = np.concatenate((np.log(library_size_rna), np.log(library_size_atac)), axis=1)

    input_data = np.concatenate((rna_input, atac_input), axis=1)
    input_matrix = torch.from_numpy(input_data).to(device, dtype=torch.float)

    input_batch = torch.from_numpy(encoding_batch_onehot[index_train, :]).to(device, dtype=torch.float)

    tf_exp = torch.from_numpy(rna_input[:, tf_indices]).to(device, dtype=torch.float)

    library_size_value = torch.from_numpy(library_size).to(device, dtype=torch.float)
    num_cells_value = torch.from_numpy(num_cells[index_train, :]).to(device, dtype=torch.float)

    return (input_matrix, tf_exp, library_size_value, num_cells_value, input_batch)
