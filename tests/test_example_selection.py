"""An explicit worked example must not become the score-scaling reference."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scdori._core.exploration import choose_example_topics
from scdori._core.interpretation import tf_importance_scores
from scdori.configuration import load_notebook_config


def test_explicit_topic_overrides_strongest_and_keeps_reference():
    table = pd.DataFrame({"ery3": [0.9, 0.2, 0.1], "other": [0.1, 0.7, 0.6]}, index=["Topic_2", "Topic_20", "Topic_31"])
    before = table.copy()
    assert choose_example_topics(table, ["ery3"], [20]) == {"ery3": ["Topic_20"]}
    pd.testing.assert_frame_equal(table, before)
    with pytest.raises(ValueError, match="cell types"):
        choose_example_topics(table, ["missing"], [20])
    with pytest.raises(ValueError, match="retained"):
        choose_example_topics(table, ["ery3"], [21])


@pytest.mark.parametrize(
    "field,value",
    [
        ("example_topics", [True]),
        ("example_topics", [-1]),
        ("example_topics", [1, 1]),
        ("example_topics", ["20"]),
        ("example_tfs", [2]),
        ("example_celltypes", [""]),
    ],
)
def test_invalid_example_config_fails_early(tmp_path, field, value):
    path = tmp_path / "scdori_config.json"
    path.write_text(json.dumps({"downstream": {field: value}}))
    with pytest.raises(ValueError):
        load_notebook_config(path)


def test_shared_config_roundtrip(tmp_path):
    from scdori.configuration import save_notebook_config

    fields = {"example_celltypes": ["ery3"], "example_topics": [20], "example_tfs": ["tf"], "example_gene": "gene"}
    path = tmp_path / "scdori_config.json"
    path.write_text(json.dumps({"downstream": fields}))
    settings = load_notebook_config(path)
    save_notebook_config(settings, "downstream", tmp_path / "output")
    saved = json.loads((tmp_path / "output/notebook_config.json").read_text())
    for key, value in fields.items():
        assert getattr(settings.downstream, key) == value
        assert saved["downstream"][key] == value


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("top_k", [1, 3])
def test_plot_subset_does_not_change_tf_scores(sign, top_k, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scdori._core.downstream import get_top_activators_per_topic, get_top_repressor_per_topic

    # Non-contiguous IDs guard against interpreting the selected row position as topic ID.
    grn = np.random.default_rng(0).uniform(0.1, 2, (6, 3, 4)) * sign
    fn = get_top_activators_per_topic if sign == 1 else get_top_repressor_per_topic
    monkeypatch.setattr(plt, "show", lambda: None)
    all_scores, all_tfs = fn(grn, ["a", "b", "c"], None, selected_topics=[1, 3, 5], top_k=top_k)
    subset_scores, subset_tfs = fn(grn, ["a", "b", "c"], None, selected_topics=[1, 3, 5], display_topics=[3], top_k=top_k)
    pd.testing.assert_frame_equal(all_scores, subset_scores)
    assert subset_tfs == all_tfs
    with pytest.raises(ValueError, match="included"):
        fn(grn, ["a", "b", "c"], None, selected_topics=[1, 3, 5], display_topics=[0])
    plt.close("all")


@pytest.mark.parametrize("sign", [1, -1])
def test_tied_tf_plot_subset_keeps_scores_and_does_not_raise(sign, monkeypatch):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scdori._core.downstream import get_top_activators_per_topic, get_top_repressor_per_topic

    # Many equal TF strengths trigger different tie handling in sort_values/nlargest.
    weights = np.tile([0., 1.], 20)
    grn = np.tile(weights[None, :, None], (6, 1, 2)) * sign
    before = grn.copy()
    names = [f"tf{i}" for i in range(40)]
    plotted = []
    monkeypatch.setattr(sns, "clustermap", lambda frame, **kwargs: plotted.append(frame.copy()))
    monkeypatch.setattr(plt, "show", lambda: None)
    fn = get_top_activators_per_topic if sign == 1 else get_top_repressor_per_topic
    all_scores, all_tfs = fn(grn, names, None, selected_topics=[1, 3, 5], top_k=2, zscore=False)
    subset_scores, subset_tfs = fn(grn, names, None, selected_topics=[1, 3, 5],
                                 display_topics=[3], top_k=2, zscore=False)
    assert list(plotted[-1].index) == ["Topic_3"]
    assert set(plotted[-1].columns).issubset(all_tfs)
    assert len(plotted[-1].columns) == 2
    pd.testing.assert_frame_equal(all_scores, subset_scores)
    assert all_tfs == subset_tfs
    np.testing.assert_array_equal(grn, before)


def test_notebook_single_topic_scales_before_display():
    notebook = json.loads((Path(__file__).parents[1] / "docs/notebooks/downstream.ipynb").read_text())
    grn = np.random.default_rng(10).uniform(0, 1, (6, 3, 4))
    scope = {
        "np": np,
        "pd": pd,
        "scaling_topic_ids": [1, 3, 5],
        "TF_DISPLAY_TOPICS": [3],
        "TF_TOP_N": 2,
        "TF_MAX_PLOT_TFS": 3,
        "grn_act": grn,
        "grn_rep": -grn * 0.4,
        "model": type("Model", (), {"num_topics": 6})(),
        "tf_names": ["a", "b", "c"],
        "tf_importance_scores": tf_importance_scores,
    }
    score_cell = next(cell for cell in notebook["cells"] if cell.get("id") == "tf-score-compute")
    exec("".join(score_cell["source"]), scope)
    assert scope["tf_score_display"] == ["Topic_3"]
    for sign, network in [("activator", grn), ("repressor", grn * 0.4)]:
        expected = tf_importance_scores(
            pd.DataFrame(
                network.sum(axis=2)[[1, 3, 5]], index=["Topic_1", "Topic_3", "Topic_5"], columns=["a", "b", "c"]
            )
        )
        for view, frame in expected.items():
            pd.testing.assert_frame_equal(scope["topic_tf_scores"][sign][view], frame)
