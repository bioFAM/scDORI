"""Exercise the tutorial's two-population target comparison on small fitted networks."""
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns

from scdori._core.exploration import choose_example_topics


def source():
    n = json.loads((Path(__file__).parents[1] / "docs/notebooks/downstream.ipynb").read_text())
    return next("".join(c["source"]) for c in n["cells"] if c["id"] == "context-target-comparison")


@pytest.fixture
def scope(tmp_path):
    weights = np.zeros((5, 2, 4))
    weights[3, 1] = [3, 2, 0, 0]
    weights[1, 1] = [0, 1, 4, 0]
    # An excluded topic must not replace the retained leading topic, even with greater usage.
    usage = pd.DataFrame({"Population_A": [.9, .1, .8], "Population_B": [.9, .7, .2]},
                         index=["Topic_0", "Topic_1", "Topic_3"])
    return dict(np=np, pd=pd, plt=plt, sns=sns, json=json, display=lambda value: None,
                choose_example_topics=choose_example_topics, CONTEXT_TF="TF_B",
                CONTEXT_CELLTYPES=["Population_A", "Population_B"], CONTEXT_TARGETS=2,
                df_topic_celltype=usage, scaling_topic_ids=[1, 3], tf_names=["TF_A", "TF_B"],
                rna_metacell=SimpleNamespace(var_names=["Gene_A", "Shared", "Gene_B", "Absent"]),
                grn_act=weights, GRN_OUTPUT=tmp_path)


def test_comparison_preserves_topic_ids_and_raw_weights(scope):
    before = scope["grn_act"].copy()
    exec(source(), scope)
    expected = pd.DataFrame({"Topic_3": [3., 2., 0.], "Topic_1": [0., 1., 4.]},
                            index=["Gene_A", "Shared", "Gene_B"])
    pd.testing.assert_frame_equal(scope["context_target_table"], expected)
    saved = pd.read_csv(scope["GRN_OUTPUT"] / "tf_context_comparison/target_weights.csv", index_col=0)
    pd.testing.assert_frame_equal(saved.rename_axis(None), expected)
    np.testing.assert_array_equal(scope["grn_act"], before)
    assert scope["scaling_topic_ids"] == [1, 3]
    assert scope["context_target_ready"]


@pytest.mark.parametrize("case", ["disabled", "missing_tf", "missing_group", "same_topic", "no_links"])
def test_inapplicable_comparison_skips_without_inventing_targets(scope, case):
    if case == "disabled":
        scope["CONTEXT_TF"] = None
    elif case == "missing_tf":
        scope["CONTEXT_TF"] = "NotFitted"
    elif case == "missing_group":
        scope["CONTEXT_CELLTYPES"] = ["Population_A", "NotAnnotated"]
    elif case == "same_topic":
        scope["df_topic_celltype"].loc["Topic_3"] = .8
    else:
        scope["grn_act"][:] = 0
    exec(source(), scope)
    assert not scope["context_target_ready"]
    assert not (scope["GRN_OUTPUT"] / "tf_context_comparison").exists()


def test_zero_support_in_one_context_is_visible_not_silently_replaced(scope):
    scope["grn_act"][1] = 0
    exec(source(), scope)
    assert scope["context_target_ready"]
    assert (scope["context_target_table"]["Topic_1"] == 0).all()
    assert scope["context_summary"].iloc[1]["Leading targets"] == "No retained activating links"
