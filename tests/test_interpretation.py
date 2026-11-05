import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scdori._core.interpretation import compare_tf_rankings, gene_importance_scores, tf_importance_scores, top_features


def test_gene_scores_match_original_downstream_helper():
    from scdori._core.downstream import compute_topic_gene_matrix
    from scdori._core.models import scDoRI

    torch.manual_seed(71)
    model = scDoRI(torch.device("cpu"), 7, 11, 3, 4, 2, 9, 5)
    with torch.no_grad():
        raw = pd.DataFrame(model.topic_gene_matrix().numpy())
    scores = gene_importance_scores(raw)
    expected = compute_topic_gene_matrix(model, "cpu")
    np.testing.assert_allclose(scores["downstream_softmax"], expected, rtol=1e-5, atol=2e-6)
    np.testing.assert_array_equal(
        np.argsort(scores["across_topics_zscore"], axis=1), np.argsort(scores["downstream_softmax"], axis=1)
    )


def test_specific_gene_outweighs_shared_high_baseline_and_constant_is_zero():
    raw = pd.DataFrame(
        [[1000, 1, 7], [1001, 9, 7], [999, 1, 7], [1000, 1, 7]], columns=["broad", "specific", "constant"]
    )
    scores = gene_importance_scores(raw)
    assert scores["across_topics_zscore"].loc[1, "specific"] > scores["across_topics_zscore"].loc[1, "broad"]
    assert (scores["across_topics_zscore"]["constant"] == 0).all()
    np.testing.assert_allclose(scores["downstream_softmax"].sum(axis=1), 1)


def test_tf_minmax_scores_use_within_group_shares_as_in_the_paper():
    # The first topic is 50 times larger overall, which must not decide the across-topic comparison.
    raw = pd.DataFrame([[100.0, 100.0], [1.0, 3.0]], columns=["a", "b"])
    scores = tf_importance_scores(raw)
    share = raw.div(raw.sum(axis=1), axis=0)
    across = (share - share.min()) / (share.max() - share.min())
    within = share.sub(share.min(axis=1), axis=0).div(share.max(axis=1) - share.min(axis=1), axis=0).fillna(0)
    np.testing.assert_allclose(scores["across_groups_minmax"], across, atol=1e-6)
    np.testing.assert_allclose(scores["minmax_consensus"], (across + within) / 2, atol=1e-6)
    # On raw sums, TF b would look strongest in the large topic; on shares it is strongest in the second.
    assert scores["across_groups_minmax"].loc[1, "b"] == pytest.approx(1)
    assert scores["across_groups_minmax"].loc[0, "b"] == pytest.approx(0)
    pd.testing.assert_frame_equal(scores["raw"], raw)


def test_tf_scores_match_historical_row_share_then_pandas_zscore():
    raw = pd.DataFrame([[100, 1, 0], [100, 20, 0], [5, 0, 0]], columns=["broad", "specific", "absent"])
    scores = tf_importance_scores(raw)
    share = raw.div(raw.sum(axis=1) + 1e-8, axis=0)
    expected = share.apply(lambda column: (column - column.mean()) / (column.std() + 1e-8))
    np.testing.assert_allclose(scores["share_across_groups_zscore"], expected)
    assert (scores["minmax_consensus"].to_numpy() >= 0).all()
    assert (scores["minmax_consensus"].to_numpy() <= 1).all()
    assert (scores["raw"]["absent"] == 0).all()
    assert (scores["minmax_consensus"]["absent"] == 0).all()
    np.testing.assert_allclose(
        scores["minmax_consensus"], (scores["within_group_minmax"] + scores["across_groups_minmax"]) / 2
    )


@pytest.mark.parametrize("values", [[[0, 0]], [[2, 2]], [[0, 0], [0, 0]]])
def test_constant_and_singleton_axes_are_finite(values):
    for score in tf_importance_scores(pd.DataFrame(values)).values():
        assert np.isfinite(score.to_numpy()).all()


def test_reject_bad_inputs_and_keep_names_in_rankings():
    with pytest.raises(ValueError, match="finite"):
        gene_importance_scores(pd.DataFrame([[np.nan]]))
    with pytest.raises(ValueError, match="unique"):
        gene_importance_scores(pd.DataFrame([[1, 2]], columns=["same", "same"]))
    with pytest.raises(ValueError, match="nonnegative"):
        tf_importance_scores(pd.DataFrame([[-1]]))
    raw = pd.DataFrame([[3, 9], [7, 1]], index=["Topic_9", "Topic_2"], columns=["TF_B", "TF_A"])
    top = top_features(tf_importance_scores(raw), "minmax_consensus", 1)
    assert top["group"].tolist() == ["Topic_9", "Topic_2"]
    assert top["feature"].tolist() == ["TF_A", "TF_B"]
    assert top["raw"].tolist() == [9, 7]


@pytest.mark.parametrize("reference,display_topics", [([0, 2], [2, 0]), ([0, 1, 2], [0, 2]), ([2], [2])])
def test_downstream_notebook_final_grn_score_cells(tmp_path, reference, display_topics):
    """Exercise the shipped notebook cells with signed GRNs and noncontiguous IDs."""
    from types import SimpleNamespace

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    path = Path(__file__).resolve().parents[1] / "docs/notebooks/downstream.ipynb"
    notebook = json.loads(path.read_text())
    cells = {cell.get("id"): "".join(cell["source"]) for cell in notebook["cells"]}
    # Topic 1 is a broad, strong reference; TF absent has no supported edges.
    act = np.array([[10.0, 5, 0], [1000, 10, 0], [2, 8, 0]])[:, :, None]
    rep = -np.array([[4.0, 2, 0], [3, 1, 0], [0, 0, 0]])[:, :, None]
    act_before, rep_before = act.copy(), rep.copy()
    namespace = {
        "np": np,
        "pd": pd,
        "torch": torch,
        "plt": plt,
        "sns": sns,
        "model": SimpleNamespace(num_topics=3),
        "grn_act": act,
        "grn_rep": rep,
        "tf_names": ["broad", "specific", "absent"],
        "scaling_topic_ids": reference,
        "selected_topic_ids": display_topics,
        "display_topic_ids": display_topics,
        "display_topic_names": [f"Topic_{i}" for i in display_topics],
        "overview_topic_ids": display_topics,
        "GRN_OUTPUT": tmp_path,
        "tf_normalised": torch.tensor([[1, 1, 0]] * 3),
        "SCALE_SELECTED_TOPICS": len(reference) != 3,
        "TOPIC_ACTIVITY_THRESHOLD": None,
        "EXCLUDE_TOPICS": [1] if 1 not in reference else [],
        "df_topic_celltype": pd.DataFrame(
            {"A": [0.5, 0.1, 0.2], "B": [0.1, 0.4, 0.6]}, index=["Topic_0", "Topic_1", "Topic_2"]
        ),
    }
    exec(cells["tf-score-options"], namespace)
    namespace["TF_TOP_N"] = 1
    displayed = []
    namespace["display"] = displayed.append
    for cell_id in ("tf-score-compute", "tf-score-export", "tf-score-plot", "tf-rank-comparison"):
        exec(cells[cell_id], namespace)
    output = tmp_path / "tf_score_views"
    raw_rep = pd.read_csv(output / "topic_tf_repressor_raw.csv", index_col=0)
    np.testing.assert_array_equal(raw_rep.to_numpy(), -rep[reference].sum(axis=2))
    scores = namespace["topic_tf_scores"]["activator"]
    assert scores["raw"].index.tolist() == [f"Topic_{i}" for i in reference]
    if reference == [0, 2]:
        # Across-topic scaling excludes the large Topic_1 baseline.
        np.testing.assert_allclose(scores["across_groups_minmax"], [[1, 0, 0], [0, 1, 0]])
        np.testing.assert_allclose(scores["within_group_minmax"], [[1, 0.5, 0], [0.25, 1, 0]])
        np.testing.assert_allclose(scores["minmax_consensus"], [[1, 0.25, 0], [0.125, 1, 0]])
    if len(reference) == 1:
        np.testing.assert_array_equal(scores["across_groups_minmax"], 0)
        assert pd.read_csv(output / "top_topic_tf_repressor.csv").empty
    for mode in ("activator", "repressor"):
        comparison = pd.read_csv(output / f"topic_tf_{mode}_ranking_comparison.csv")
        pd.testing.assert_frame_equal(
            comparison,
            compare_tf_rankings(
                namespace["topic_tf_scores"][mode],
                top_n=namespace["TF_TOP_N"],
                topics=[f"Topic_{i}" for i in display_topics],
            ),
            check_dtype=False,
        )
        table = pd.read_csv(output / f"top_topic_tf_{mode}.csv")
        assert "absent" not in table.tf.tolist()
        assert (table.raw > 0).all()
        assert set(table.topic).issubset({f"Topic_{i}" for i in display_topics})
    if len(reference) == 1:
        # One reference topic has no across-topic contrast: every TF ties at 0 there, the tie goes
        # to the largest raw contribution, and the three top-1 lists agree. (Differences between
        # views are covered by test_tf_ranking_comparison_exposes_different_top_lists_without_rescaling.)
        assert not [table for table in displayed if "Top within topic" in table.columns]
    # Display-only filtering leaves reference scores and the GRNs untouched.
    before = scores["minmax_consensus"].copy()
    namespace["TF_DISPLAY_TOPICS"] = display_topics[:1]
    exec(cells["tf-score-compute"], namespace)
    pd.testing.assert_frame_equal(before, namespace["topic_tf_scores"]["activator"]["minmax_consensus"])
    np.testing.assert_array_equal(act, act_before)
    np.testing.assert_array_equal(rep, rep_before)
    metadata = json.loads((output / "score_manifest.json").read_text())
    assert metadata["reference_topic_ids"] == reference
    assert metadata["display_topic_ids"] == display_topics
    plt.close("all")


def test_tf_ranking_comparison_exposes_different_top_lists_without_rescaling():
    raw = pd.DataFrame(
        [[100, 80, 10, 0], [110, 0, 0, 0], [120, 100, 10, 0]],
        index=["Topic_9", "Topic_2", "Topic_5"],
        columns=["broad", "balanced", "specific", "absent"],
    )
    scores = tf_importance_scores(raw)
    before = scores["minmax_consensus"].copy()
    table = compare_tf_rankings(scores, top_n=1, topics=["Topic_9"])
    assert table.tf.tolist() == ["balanced", "specific", "broad"]
    by_tf = table.set_index("tf")
    assert by_tf.index[by_tf.within_top].tolist() == ["broad"]
    assert by_tf.index[by_tf.across_top].tolist() == ["specific"]
    assert by_tf.index[by_tf.mean_top].tolist() == ["balanced"]
    assert by_tf.loc["broad", "mean_rank"] == 3
    full = compare_tf_rankings(scores, top_n=1)
    pd.testing.assert_frame_equal(table, full.loc[full.topic == "Topic_9"].reset_index(drop=True))
    pd.testing.assert_frame_equal(before, scores["minmax_consensus"])


def test_tf_ranking_comparison_ties_single_topic_and_no_supported_links():
    scores = tf_importance_scores(pd.DataFrame([[3, 3, 0]], columns=["first", "second", "absent"]))
    assert compare_tf_rankings(scores, top_n=1).tf.tolist() == ["first"]
    table = compare_tf_rankings(scores, top_n=10)
    assert table.tf.tolist() == ["first", "second"]
    assert (table.across_score == 0).all()
    empty = compare_tf_rankings(tf_importance_scores(pd.DataFrame([[0, 0]])))
    assert empty.empty and {"topic", "tf", "mean_rank", "within_top", "across_top"}.issubset(empty.columns)
    assert compare_tf_rankings(scores, topics=[]).empty


def test_equal_scores_are_resolved_by_raw_contribution_not_column_order():
    # Both TFs peak in Topic_0, so both score exactly 1 across topics there; "tiny" comes first in
    # column order but contributes 100 times less.
    raw = pd.DataFrame([[0.05, 5.0, 1.0], [0.01, 1.0, 2.0]], index=["Topic_0", "Topic_1"],
                       columns=["tiny", "large", "other"])
    scores = tf_importance_scores(raw)
    assert scores["across_groups_minmax"].loc["Topic_0", ["tiny", "large"]].tolist() == [1.0, 1.0]
    ranks = compare_tf_rankings(scores, top_n=2, topics=["Topic_0"]).set_index("tf")
    assert ranks.loc["large", "across_rank"] == 1 and ranks.loc["tiny", "across_rank"] == 2
    top1 = compare_tf_rankings(scores, top_n=1, topics=["Topic_0"]).set_index("tf")
    assert top1.index[top1.across_top].tolist() == ["large"] and "tiny" not in top1.index
    ranked = top_features(scores, "across_groups_minmax", top_n=1)
    assert ranked.loc[ranked.group == "Topic_0", "feature"].tolist() == ["large"]


def test_tf_ranking_comparison_rejects_mismatched_views_and_invalid_requests():
    scores = tf_importance_scores(pd.DataFrame([[1, 2], [2, 1]]))
    for top_n in [0, -1, 1.5, True]:
        with pytest.raises(ValueError, match="positive integer"):
            compare_tf_rankings(scores, top_n=top_n)
    for topics in [[2], [0, 0]]:
        with pytest.raises(ValueError, match="Display topics"):
            compare_tf_rankings(scores, topics=topics)
    reversed_scores = dict(scores, across_groups_minmax=scores["across_groups_minmax"].iloc[::-1])
    with pytest.raises(ValueError, match="same topic and TF order"):
        compare_tf_rankings(reversed_scores)
    with pytest.raises(ValueError, match="nonnegative"):
        compare_tf_rankings(dict(scores, raw=-scores["raw"]))
