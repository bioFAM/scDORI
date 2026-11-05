import copy
import json
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, TensorDataset

from scdori import prepare_count_matrix
from scdori._core.config import TrainConfig
from scdori._core.data_io import load_scdori_inputs
from scdori._core.dataloader import create_minibatch
from scdori._core.models import scDoRI
from scdori._core.train_grn import get_tf_expression, train_model_grn
from scdori._core.train_scdori import train_scdori_phases
from scdori.configuration import load_notebook_config
from scdori.pp.data_io import save_processed_datasets


@pytest.mark.parametrize("constructor", [sp.csc_matrix, sp.csc_array])
@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_conversion_preserves_values_metadata_and_other_slots(constructor, dtype):
    # Includes an empty row/column and non-integral counts.
    values = np.array([[0, 0, 0], [1.5, 0, 3], [0, 0, 7]], dtype=dtype)
    data = ad.AnnData(
        constructor(values),
        obs=pd.DataFrame({"num_cells": [1, 3, 2]}, index=["c", "b", "a"]),
        var=pd.DataFrame(index=["z", "x", "y"]),
    )
    data.uns["scdori_atac_count_mode"] = "fragments"
    data.layers["original"] = data.X.copy()
    data.obsm["embedding"] = np.ones((3, 2))
    data.raw = data.copy()
    original = data.copy()
    layer, raw = data.layers["original"], data.raw
    result = prepare_count_matrix(data)
    assert result is data
    assert result.X.format == "csr" and result.X.dtype == dtype
    assert isinstance(result.X, sp.sparray) == isinstance(original.X, sp.sparray)
    np.testing.assert_array_equal(result.X.toarray(), values)
    pd.testing.assert_frame_equal(result.obs, original.obs)
    pd.testing.assert_frame_equal(result.var, original.var)
    assert result.uns == original.uns
    assert result.layers["original"] is layer and result.raw is raw
    np.testing.assert_array_equal(result.obsm["embedding"], original.obsm["embedding"])
    matrix = result.X
    assert prepare_count_matrix(result).X is matrix  # no repeated conversion


@pytest.mark.parametrize("constructor", [np.asarray, sp.csr_matrix, sp.csr_array, sp.csc_matrix, sp.csc_array])
def test_opt_out_and_already_prepared_inputs_are_noops(constructor):
    data = ad.AnnData(constructor(np.eye(3)))
    matrix = data.X
    assert prepare_count_matrix(data, use_csr=False) is data
    assert data.X is matrix
    if not sp.issparse(matrix) or matrix.format == "csr":
        assert prepare_count_matrix(data) is data
        assert data.X is matrix


def test_csc_view_copies_without_touching_parent():
    parent = ad.AnnData(sp.csc_matrix(np.arange(20).reshape(5, 4)))
    matrix = parent.X
    view = parent[[3, 1, 3], [2, 0]]
    expected = view.X.toarray()
    actual = prepare_count_matrix(view)
    assert actual is not view and view.is_view and not actual.is_view
    assert parent.X is matrix and parent.X.format == "csc"
    np.testing.assert_array_equal(actual.X.toarray(), expected)


def test_backed_preparation_does_not_even_access_x():
    class Backed:
        isbacked = True

        @property
        def X(self):
            raise AssertionError("Backed counts must not be read during preparation")

    data = Backed()
    assert prepare_count_matrix(data) is data


@pytest.mark.parametrize("backed", [False, True])
@pytest.mark.parametrize("constructor", [sp.csc_matrix, sp.csc_array])
@pytest.mark.parametrize("mode", ["insertions", "fragments"])
def test_csc_and_csr_minibatches_match(tmp_path, backed, constructor, mode):
    rng = np.random.default_rng(19)
    rna = ad.AnnData(constructor(rng.poisson(2, (7, 4)).astype(np.float32)))
    atac = ad.AnnData(constructor(rng.poisson(2, (7, 5)).astype(np.float32) / 2))
    atac.uns["scdori_atac_count_mode"] = mode
    csr_rna, csr_atac = prepare_count_matrix(rna.copy()), prepare_count_matrix(atac.copy())
    if backed:
        rna.write_h5ad(tmp_path / "rna.h5ad")
        atac.write_h5ad(tmp_path / "atac.h5ad")
        rna = ad.read_h5ad(tmp_path / "rna.h5ad", backed="r")
        atac = ad.read_h5ad(tmp_path / "atac.h5ad", backed="r")
        assert prepare_count_matrix(rna) is rna
        assert prepare_count_matrix(atac) is atac
    args = (np.arange(1, 8, dtype=np.float32)[:, None], np.array([3, 1]), np.eye(2)[np.arange(7) % 2])
    ids = torch.tensor([5, 1, 5, 0])
    try:
        expected = create_minibatch("cpu", ids, rna, atac, *args)
        actual = create_minibatch("cpu", ids, csr_rna, csr_atac, *args)
        for left, right in zip(expected, actual, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        if backed:
            assert rna.isbacked and rna.X.format == "csc"
    finally:
        if backed:
            rna.file.close()
            atac.file.close()


@pytest.mark.parametrize("use_csr", [True, False])
def test_saving_and_loading_layout_and_config(tmp_path, use_csr):
    data = ad.AnnData(sp.csc_matrix(np.arange(12).reshape(4, 3)))
    data.uns["scdori_atac_count_mode"] = "fragments"
    # Existing cell-intersection/order behavior must be preserved.
    save_processed_datasets(data, data[[2, 0, 3]].copy(), tmp_path, count_matrix_csr=use_csr)
    expected = data[[0, 2, 3]].X.toarray()
    assert data.X.format == "csc"
    for modality in ("rna", "atac"):
        loaded = ad.read_h5ad(tmp_path / f"{modality}_processed.h5ad")
        assert loaded.X.format == ("csr" if use_csr else "csc")
        np.testing.assert_array_equal(loaded.X.toarray(), expected)
        assert list(loaded.obs_names) == ["0", "2", "3"]
        assert loaded.uns["scdori_atac_count_mode"] == "fragments"
    for name in ("distance", "act", "rep"):
        np.save(tmp_path / f"{name}.npy", np.ones((3, 3)))
    config = SimpleNamespace(
        data_dir=tmp_path,
        output_subdir="",
        rna_metacell_file="rna_processed.h5ad",
        atac_metacell_file="atac_processed.h5ad",
        gene_peak_distance_file="distance.npy",
        insilico_chipseq_act_file="act.npy",
        insilico_chipseq_rep_file="rep.npy",
    )
    # Old user configs without the new setting get the default conversion.
    loaded = load_scdori_inputs(config)
    assert loaded[0].X.format == loaded[1].X.format == "csr"
    np.testing.assert_array_equal(loaded[0].X.toarray(), expected)
    config.count_matrix_csr = False
    assert load_scdori_inputs(config)[0].X.format == ("csr" if use_csr else "csc")
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"preprocessing": {"count_matrix_csr": use_csr}, "training": {"count_matrix_csr": use_csr}})
    )
    settings = load_notebook_config(path)
    assert settings.preprocessing.count_matrix_csr is use_csr
    assert settings.training.count_matrix_csr is use_csr


@pytest.mark.parametrize("mode", ["insertions", "fragments"])
@pytest.mark.parametrize("phase2", ["legacy", "frozen_uncached", "frozen_cached"])
def test_paired_training_losses_gradients_and_checkpoints(tmp_path, mode, phase2, monkeypatch):
    # Run the actual training functions, with identical RNG and batch sequences.
    from scdori._core import train_scdori

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(27)
    rng = np.random.default_rng(27)
    initial = scDoRI(torch.device("cpu"), 7, 11, 3, 4, 2, 9, 5)
    with torch.no_grad():
        initial.tf_binding_matrix_repressor.mul_(-1)
    rna = ad.AnnData(sp.csc_matrix(rng.poisson(3, (12, 7)).astype(np.float32)))
    atac = ad.AnnData(sp.csc_matrix(rng.poisson(3, (12, 11)).astype(np.float32) / 2))
    atac.uns["scdori_atac_count_mode"] = mode
    cells = np.arange(1, 13, dtype=np.float32)[:, None]
    tfs = np.array([1, 3, 5])
    batches = np.eye(2, dtype=np.float32)[np.arange(12) % 2]
    train = DataLoader(TensorDataset(torch.arange(8)), batch_size=4, shuffle=True)
    val = DataLoader(TensorDataset(torch.arange(8, 12)), batch_size=4)
    results, eval_losses = [], []
    original_eval = train_scdori.compute_eval_loss_scdori

    def record_eval(*args, **kwargs):
        losses = original_eval(*args, **kwargs)
        eval_losses.append(losses)
        return losses

    monkeypatch.setattr(train_scdori, "compute_eval_loss_scdori", record_eval)
    try:
        for layout in ("csc", "csr"):
            model = copy.deepcopy(initial)
            r = prepare_count_matrix(rna.copy(), use_csr=layout == "csr")
            a = prepare_count_matrix(atac.copy(), use_csr=layout == "csr")
            config = TrainConfig(
                max_scdori_epochs=2,
                epoch_warmup_1=1,
                max_grn_epochs=2,
                eval_frequency=1,
                cells_per_topic=3,
                batch_size_cell_prediction=5,
                weights_folder_scdori=str(tmp_path / layout / "phase1"),
                weights_folder_grn=str(tmp_path / layout / "phase2"),
                freeze_phase1_in_grn=phase2 != "legacy",
                grn_topic_selection="all",
                cache_frozen_grn=phase2 == "frozen_cached",
            )
            torch.manual_seed(91)
            train_scdori_phases(model, "cpu", train, val, r, a, cells, tfs, batches, config)
            phase1_grad = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            topic_tf = get_tf_expression("actual", model, "cpu", train, r, a, cells, tfs, batches, config)
            train_model_grn(model, "cpu", train, val, r, a, cells, tfs, batches, config)
            phase2_grad = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            results.append((model.state_dict(), phase1_grad, phase2_grad, {"topic_tf": topic_tf}))
        np.testing.assert_array_equal(eval_losses[:2], eval_losses[2:])
        for left, right in zip(*results, strict=True):
            assert left.keys() == right.keys()
            for key in left:
                assert torch.isfinite(left[key]).all(), key
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0, msg=key)
        for phase in ("phase1", "phase2"):
            left = torch.load(tmp_path / "csc" / phase / "best_scdori_best_eval.pth", weights_only=True)
            right = torch.load(tmp_path / "csr" / phase / "best_scdori_best_eval.pth", weights_only=True)
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0, msg=key)
    finally:
        torch.set_num_threads(old_threads)
