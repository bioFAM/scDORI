import numpy as np
import torch
from tqdm import tqdm

from .dataloader import create_minibatch


def get_latent_topics(
    model,
    device,
    data_loader,
    rna_anndata,
    atac_anndata,
    num_cells,
    tf_indices,
    encoding_batch_onehot,
    *,
    return_indices=False,
):
    """
    Extract the softmaxed topic activations (theta) for each cell in the dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The scDORI model containing an encoder for generating topic distributions.
    device : torch.device
        The PyTorch device (e.g., 'cpu' or 'cuda') used for computations.
    data_loader : torch.utils.data.DataLoader
        A DataLoader that yields batches of cell indices.
    rna_anndata : anndata.AnnData
        RNA single-cell data in AnnData format.
    atac_anndata : anndata.AnnData
        ATAC single-cell data in AnnData format.
    num_cells : np.ndarray
        Number of cells in each row (e.g., if using metacells). set to ones for single-cell data.
    tf_indices : np.ndarray
        Indices of transcription factor genes in the RNA data.
    encoding_batch_onehot : np.ndarray
        One-hot encoding of batch information (cells x num_batches).

    return_indices : bool, optional
        Also return absolute cell indices aligned with the output rows.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (n_cells, n_topics) representing the softmaxed
        topic activations for each cell in the order given by the DataLoader.
    """
    modes = {module: module.training for module in model.modules()}
    model.eval()
    all_thetas, all_indices = [], []
    try:
        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc="Extracting latent topics"):
                cell_indices = batch_data[0].cpu()
                inputs, _, libraries, cells, _ = create_minibatch(
                    device, cell_indices, rna_anndata, atac_anndata, num_cells, tf_indices, encoding_batch_onehot
                )
                theta, _ = model.encode(
                    inputs[:, : model.num_genes],
                    inputs[:, model.num_genes :],
                    libraries[:, :1],
                    libraries[:, 1:],
                    cells,
                )
                all_thetas.append(theta.cpu().numpy())
                all_indices.append(cell_indices.numpy())
    finally:
        for module, training in modes.items():
            module.training = training
    if not all_thetas:
        raise ValueError("Cannot extract topics from an empty data loader")
    topics = np.concatenate(all_thetas)
    if return_indices:
        return topics, np.concatenate(all_indices)
    return topics
