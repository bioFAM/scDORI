"""scanpy switches to pynndescent at 8,192 cells, so small test data never reach this code path."""

import numpy as np
import pytest


def test_neighbors_work_with_the_transformer_used_for_real_data():
    anndata = pytest.importorskip("anndata")
    sc = pytest.importorskip("scanpy")
    pytest.importorskip("pynndescent")
    cells = anndata.AnnData(np.random.default_rng(0).normal(size=(300, 10)).astype("float32"))
    sc.pp.neighbors(cells, n_neighbors=15, use_rep="X", transformer="pynndescent", random_state=0)
    assert cells.obsp["distances"].shape == (300, 300)
    assert (np.asarray((cells.obsp["distances"] > 0).sum(axis=1)).ravel() > 0).all()
