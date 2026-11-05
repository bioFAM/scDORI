"""The minimum-detection rule for TFs in phase 2 (``tf_expression_min_detection``)."""

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from scdori._core.config import TrainConfig
from scdori._core.tf_preview import TFExpressionPreview
from scdori._core.train_grn import (
    get_tf_expression,
    scale_topic_tf_expression,
    topic_tf_detection,
    topic_tf_detection_filter,
    topic_tf_expression_means,
)
from scdori.configuration import check_notebook_inputs

CELLS = 200


def counts(detected):
    """``CELLS`` cells; TF ``j`` has one count in the first ``detected[j]`` cells."""
    matrix = np.zeros((CELLS, len(detected)), dtype=np.float32)
    for j, n in enumerate(detected):
        matrix[:n, j] = 1
    matrix[:, 0] += 1  # every cell has RNA, so TF totals stay positive
    names = [f"TF{j}" for j in range(len(detected))]
    return ad.AnnData(sp.csr_matrix(matrix), var=pd.DataFrame(index=names), obs=pd.DataFrame(index=map(str, range(CELLS))))


def config(**kwargs):
    return SimpleNamespace(**{"tf_expression_min_detection": 0.02, "tf_expression_keep_tfs": [], **kwargs})


SELECTED = np.arange(CELLS)[:, None]


def test_default_is_two_percent_and_can_be_turned_off():
    assert TrainConfig().tf_expression_min_detection == 0.02
    assert TrainConfig().tf_expression_keep_tfs == []
    _, keep = topic_tf_detection_filter(SELECTED, counts([200, 3]), [0, 1], config(tf_expression_min_detection=0))
    assert keep.all()


def test_detection_counts_cells_with_at_least_one_rna_count():
    rna = counts([200, 3, 4, 0])
    np.testing.assert_array_equal(topic_tf_detection(SELECTED, rna, [0, 1, 2, 3])[0], [1, 0.015, 0.02, 0])


def test_exactly_at_the_floor_is_kept_and_below_is_removed():
    # 2% of 200 cells is 4 cells. The boundary must hold exactly, with no floating-point surprise.
    _, keep = topic_tf_detection_filter(SELECTED, counts([200, 3, 4]), [0, 1, 2], config())
    np.testing.assert_array_equal(keep[0], [True, False, True])


@pytest.mark.parametrize("floor, cells", [(0.05, 10), (0.07, 14), (0.015, 3), (0.1, 20)])
def test_boundary_holds_for_other_floors(floor, cells):
    _, keep = topic_tf_detection_filter(
        SELECTED, counts([200, cells - 1, cells]), [0, 1, 2], config(tf_expression_min_detection=floor))
    np.testing.assert_array_equal(keep[0], [True, False, True])


def test_detection_uses_each_topics_own_cells():
    rna = counts([200, 100])  # TF1 is seen in the first 100 cells only
    selected = np.column_stack([np.arange(100), np.arange(100, 200)])
    detection, keep = topic_tf_detection_filter(selected, rna, [0, 1], config())
    np.testing.assert_array_equal(detection[:, 1], [1, 0])
    np.testing.assert_array_equal(keep[:, 1], [True, False])


def test_listed_and_requested_tfs_are_kept_and_nothing_else_changes():
    rna = counts([200, 0, 0, 0])
    _, keep = topic_tf_detection_filter(SELECTED, rna, [0, 1, 2, 3], config(tf_expression_keep_tfs=["TF1"]))
    np.testing.assert_array_equal(keep[0], [True, True, False, False])
    rna.uns["scdori_requested_features"] = {"tfs_retained": ["TF2"]}
    _, keep = topic_tf_detection_filter(SELECTED, rna, [0, 1, 2, 3], config(tf_expression_keep_tfs=["TF1"]))
    np.testing.assert_array_equal(keep[0], [True, True, True, False])
    rna.uns["scdori_requested_features"] = {"tfs_retained": np.array([], dtype=str)}  # as read back from h5ad
    _, keep = topic_tf_detection_filter(SELECTED, rna, [0, 1, 2, 3], config())
    np.testing.assert_array_equal(keep[0], [True, False, False, False])


def test_a_misspelt_tf_to_keep_is_an_error_not_silently_ignored():
    with pytest.raises(ValueError, match="not in the fitted TF panel.*Nto"):
        topic_tf_detection_filter(SELECTED, counts([200, 0]), [0, 1], config(tf_expression_keep_tfs=["Nto"]))


def test_keeping_names_only_the_panel_positions_asked_for():
    # tf_indices need not start at zero or cover every feature.
    rna = counts([200, 0, 0])
    detection, keep = topic_tf_detection_filter(SELECTED, rna, [2, 1], config(tf_expression_keep_tfs=["TF1"]))
    assert detection.shape == (1, 2)
    np.testing.assert_array_equal(keep[0], [False, True])


def test_config_from_before_this_setting_keeps_every_tf():
    _, keep = topic_tf_detection_filter(SELECTED, counts([200, 0]), [0, 1], SimpleNamespace())
    assert keep.all()


def test_invalid_floor_is_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        topic_tf_detection_filter(SELECTED, counts([200, 0]), [0, 1], config(tf_expression_min_detection=2))


def test_gate_is_the_unfloored_gate_with_entries_zeroed_and_scores_unchanged(monkeypatch):
    """The floor never rescales: surviving entries equal the gate without the floor, exactly."""
    rna = counts([200, 3, 50, 4, 150])
    theta = np.random.default_rng(0).random((CELLS, 3))
    ids = np.arange(CELLS)
    tfs = [0, 1, 2, 3, 4]
    monkeypatch.setattr("scdori._core.train_grn.get_latent_topics", lambda *a, **k: (theta, ids))
    monkeypatch.setattr("scdori._core.topic_selection.excluded_grn_topics", lambda model: [])
    model = SimpleNamespace(topic_tf_decoder=torch.zeros(1, dtype=torch.float32))
    settings = {"tf_expression_clamp": 0.2, "tf_expression_scaling": "mean_within_across", "cells_per_topic": 40}
    args = (model, "cpu", None, rna, None, None, tfs, None)
    off = get_tf_expression("True", *args, SimpleNamespace(**settings, tf_expression_min_detection=0.0)).numpy()
    on = get_tf_expression("True", *args, SimpleNamespace(**settings, tf_expression_min_detection=0.25)).numpy()

    means, selected = topic_tf_expression_means(theta, ids, rna, tfs, 40)
    np.testing.assert_array_equal(off, scale_topic_tf_expression(torch.as_tensor(means), 0.2).numpy())
    keep = topic_tf_detection(selected, rna, tfs) >= 0.25
    np.testing.assert_array_equal(on, off * keep)
    assert ((off > 0) & (on == 0)).any() and (on > 0).any()


def preview():
    labels = {"index": ["Topic_0", "Topic_1", "Topic_2"], "columns": ["Noto", "Gata1"]}
    scores = pd.DataFrame([[0.9, 0.0], [0.3, 0.8], [0.6, 0.7]], **labels)
    weights = pd.DataFrame([[0.0, 0.0], [0.3, 0.8], [0.0, 0.0]], **labels)
    detection = pd.DataFrame([[0.015, 0.5], [0.1, 0.9], [0.01, 0.5]], **labels)
    return TFExpressionPreview(weights, ["Topic_2"], detection, scores, 0.02, 200)


def test_preview_lists_what_was_left_out_and_why():
    p = preview()
    p.scores_before_detection.loc["Topic_2"] = 0  # a skipped topic reports no scores
    removed = p.removed
    assert removed[["Topic", "TF", "Top cells with RNA"]].values.tolist() == [["Topic_0", "Noto", 3]]
    assert p.summary.loc["Topic_0", "Left out: seen in too few cells"] == 1
    assert p.summary.loc["Topic_1", "Left out: seen in too few cells"] == 0
    assert p.look_up("Noto")["Status"].to_dict() == {
        "Topic_0": "Left out: seen in too few top cells", "Topic_1": "Included", "Topic_2": "Topic skipped in phase 2"}
    assert p.look_up("Gata1").loc["Topic_0", "Status"] == "Left out: below the expression cutoff"
    with pytest.raises(ValueError, match="tfs_user"):
        p.look_up("Absent")


def test_preview_without_measured_expression_has_no_detection_column():
    p = TFExpressionPreview(pd.DataFrame([[0.5, 0.0]], index=["Topic_0"], columns=["A", "B"]), [])
    assert list(p.removed.columns)[:2] == ["Topic", "TF"] and p.removed.empty
    assert "Left out: seen in too few cells" not in p.summary.columns
    assert p.look_up("B").loc["Topic_0", "Status"] == "Left out: below the expression cutoff"


def test_notebook_check_rejects_a_floor_given_in_percent():
    from scdori.configuration import load_notebook_config

    settings = load_notebook_config()
    settings.training.tf_expression_min_detection = 2
    with pytest.raises(ValueError, match="tf_expression_min_detection"):
        check_notebook_inputs(settings, "training")
    settings.training.tf_expression_min_detection = 0.02
    settings.training.tf_expression_keep_tfs = "Noto"
    with pytest.raises(ValueError, match="tf_expression_keep_tfs"):
        check_notebook_inputs(settings, "training")
