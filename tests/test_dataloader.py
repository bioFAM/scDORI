import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from scdori._core.dataloader import create_minibatch


@pytest.mark.parametrize("backed", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("mode", ["insertions", "fragments"])
def test_batch_preserves_order_duplicates_units_and_inputs(tmp_path, backed, sparse, mode):
    rna_values = np.arange(28, dtype=np.float64).reshape(7, 4)
    atac_values = np.arange(35, dtype=np.float64).reshape(7, 5) / 4
    rna = ad.AnnData(sp.csr_matrix(rna_values) if sparse else rna_values.copy())
    atac = ad.AnnData(sp.csr_matrix(atac_values) if sparse else atac_values.copy())
    atac.uns["scdori_atac_count_mode"] = mode
    if backed:
        rna.write_h5ad(tmp_path / "rna.h5ad")
        atac.write_h5ad(tmp_path / "atac.h5ad")
        rna = ad.read_h5ad(tmp_path / "rna.h5ad", backed="r")
        atac = ad.read_h5ad(tmp_path / "atac.h5ad", backed="r")
    ids = torch.tensor([5, 1, 5, 0])
    tfs = np.array([3, 1])
    cells = np.arange(1, 8, dtype=np.float32)[:, None]
    batches = np.eye(2, dtype=np.float32)[np.arange(7) % 2]
    try:
        x, tf, lib, nc, batch = create_minibatch("cpu", ids, rna, atac, cells, tfs, batches)
        a = atac_values[ids.numpy()]
        if mode == "insertions":
            a = (a + 1) // 2
        r = rna_values[ids.numpy()]
        np.testing.assert_array_equal(x.numpy(), np.concatenate((r, a), axis=1).astype(np.float32))
        np.testing.assert_array_equal(tf.numpy(), r[:, tfs].astype(np.float32))
        np.testing.assert_allclose(lib.numpy(), np.log(np.stack((r.sum(1), a.sum(1)), axis=1) + 1e-8), rtol=1e-6)
        np.testing.assert_array_equal(nc.numpy(), cells[ids.numpy()])
        np.testing.assert_array_equal(batch.numpy(), batches[ids.numpy()])
        for data, original in ((rna, rna_values), (atac, atac_values)):
            actual = data.X[:] if backed else data.X
            actual = actual.toarray() if sp.issparse(actual) else actual
            np.testing.assert_array_equal(actual, original)
    finally:
        if backed:
            rna.file.close()
            atac.file.close()
