import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip("seaborn")

from scdori._core import exploration as ex


def test_palette_file_is_used_and_missing_labels_are_filled(tmp_path, capsys):
    a = ad.AnnData(np.zeros((4, 1)), obs=pd.DataFrame({"celltype": ["b", "a", "c", "a"]}, index=list("wxyz")))
    palette = tmp_path / "palette.json"
    palette.write_text(json.dumps({"celltype_colors": {"a": "#111111", "b": "#222222"}, "stage_colors": {}}))
    colors = ex.apply_label_colors(a, "celltype", palette)
    assert list(a.obs["celltype"].cat.categories) == ["a", "b", "c"]
    assert a.uns["celltype_colors"][:2] == ["#111111", "#222222"] and colors["c"].startswith("#")
    assert "['c']" in capsys.readouterr().out
    # without a palette, colours already stored for every label are kept
    assert ex.apply_label_colors(a, "celltype") == colors


def test_topics_celltypes_tfs_and_targets_are_chosen_from_results():
    table = pd.DataFrame(
        {"A": [0.6, 0.3, 0.1], "B": [0.5, 0.1, 0.4], "C": [0.2, 0.7, 0.1]}, index=["Topic_0", "Topic_3", "Topic_5"]
    )
    assert ex.top_topics_for_group(table, "B") == ["Topic_0", "Topic_5"]
    assert ex.example_groups(table, n=2) == ["C", "A"]  # strongest topic use, distinct topics
    assert ex.top_groups_for_topics(table, ["Topic_0", "Topic_3"], n=1) == ["A", "C"]
    grn = np.zeros((6, 3, 4))
    grn[0, 0, :2] = [0.9, 0.1]  # TF "x": one strong topic
    grn[3, 1, 1:3] = [0.3, 0.2]  # TF "y": strong in topics 3 and 5
    grn[5, 1, 3] = 0.4
    tfs, genes = ["x", "y", "z"], ["g0", "g1", "g2", "g3"]
    assert ex.example_tf(grn, tfs, [0, 3, 5], n_topics=2) == "y"
    assert ex.example_tf(grn, tfs, [0, 3, 5], n_topics=1) == "x"
    assert ex.top_topics_for_tf(grn, tfs, "y", [0, 3, 5]).index.tolist() == [3, 5]
    assert ex.top_targets(grn, tfs, genes, "y", 3, n=5).index.tolist() == ["g1", "g2"]
    links = np.array([[0.0, 0.5, 0.9], [0.2, 0.0, 0.0]])
    assert ex.top_linked_peaks(links, ["g0", "g1"], ["p0", "p1", "p2"], "g0").index.tolist() == ["p2", "p1"]
    peaks = pd.DataFrame({"Topic_0": [1.0, 0.0, 0.5]})
    assert ex.top_binding_tfs(peaks, np.array([[0.9, 0.0], [0.1, 1.0], [0.5, 0.2]]), ["t1", "t2"]) == {
        "Topic_0": ["t1"]
    }


def test_region_selection_requires_accessibility_in_requested_groups_and_aligns_peaks():
    means = pd.DataFrame({"erythroid": [0, 3, 2], "other": [10, 0, 0]}, index=["p0", "p2", "p1"])
    table = ex.linked_peak_table([[9, .5, .2]], ["g"], ["p0", "p1", "p2"], "g", means, ["erythroid"])
    assert table.index.tolist() == ["p1", "p2"]  # strongest global link is inaccessible here
    assert table.mean_accessibility.tolist() == [2, 3]
    assert ex.linked_peak_table([[9, .5, .2]], ["g"], ["p0", "p1", "p2"], "g", means,
                                ["erythroid"], min_link_weight=.5).empty
    with pytest.raises(ValueError, match="centred"):
        ex.linked_peak_table([[1, 2, 3]], ["g"], means.index, "g", means - 1, ["erythroid"])


@pytest.mark.parametrize("layout", [np.array, sparse.csr_matrix, sparse.csc_matrix])
def test_plotting_copy_preserves_original_counts_even_with_aliased_layer(layout):
    counts = np.array([[1., 3., 0.], [2., 0., 0.]])
    a = ad.AnnData(layout(counts))
    a.layers["counts"] = a.X  # alias used by older notebook cells
    prepared = ex.normalized_counts_for_plot(a)
    def dense(x):
        return x.toarray() if sparse.issparse(x) else x
    np.testing.assert_array_equal(dense(a.X), counts)
    np.testing.assert_array_equal(dense(a.layers["counts"]), counts)
    np.testing.assert_array_equal(dense(prepared.layers["counts"]), counts)
    # Scanpy's default: each cell scaled to the median library size (4 and 2 counts here).
    np.testing.assert_allclose(dense(prepared.X).sum(axis=1), 3)
    target = ex.normalized_counts_for_plot(a, target_sum=100)
    np.testing.assert_allclose(dense(target.X).sum(axis=1), 100)


def test_signed_repressors_are_ranked_separately_and_respect_topic_context():
    grn = np.array([[[-.1, -.9, .5]], [[-5., 0., 0.]]])
    before = grn.copy()
    assert ex.top_topics_for_tf(grn, ["tf"], "tf", [0], sign=-1).index.tolist() == [0]
    targets = ex.top_targets(grn, ["tf"], ["a", "b", "c"], "tf", 0, sign=-1)
    assert targets.index.tolist() == ["b", "a"]
    np.testing.assert_allclose(targets.values, [.9, .1])
    np.testing.assert_array_equal(grn, before)


def test_topic_region_ranking_uses_aligned_weights_before_top_n_and_keeps_groups_separate():
    links = np.array([[9., 2., 1.]])
    access = pd.DataFrame({"A": [1., 0., 3.], "B": [0., 8., 0.]}, index=["p0", "p1", "p2"])
    weights = pd.Series([1., .01, .8], index=["p2", "p0", "p1"])
    before = (links.copy(), access.copy(), weights.copy())
    common = (links, ["g"], ["p0", "p1", "p2"], "g", access)
    assert ex.linked_peak_table(*common, ["A"], n=1).index.tolist() == ["p0"]
    a = ex.linked_peak_table(*common, ["A"], n=1, topic_peak_weights=weights)
    b = ex.linked_peak_table(*common, ["B"], n=1, topic_peak_weights=weights)
    assert a.index.tolist() == ["p2"] and b.index.tolist() == ["p1"]
    assert a.loc["p2", "topic_contribution"] == 1
    assert b.loc["p1", "topic_contribution"] == pytest.approx(1.6)
    np.testing.assert_array_equal(links, before[0])
    pd.testing.assert_frame_equal(access, before[1])
    pd.testing.assert_series_equal(weights, before[2])


@pytest.mark.parametrize("weights", [
    pd.Series([1.], index=["missing"]),
    pd.Series([1., .5], index=["p", "p"]),
    pd.Series([-1.], index=["p"]),
    pd.Series([1.1], index=["p"]),
    pd.Series([np.nan], index=["p"]),
])
def test_topic_region_ranking_rejects_incomplete_or_invalid_peak_weights(weights):
    with pytest.raises(ValueError, match="Topic peak weights"):
        ex.linked_peak_table([[1]], ["g"], ["p"], "g", pd.DataFrame({"A": [1.]}, index=["p"]),
                             ["A"], topic_peak_weights=weights)
