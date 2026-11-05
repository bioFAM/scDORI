"""Exercise the actual notebook access cells with nonconsecutive topic IDs."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scdori._core.exploration import linked_peak_table, choose_example_topics
from scdori._core.interpretation import tf_importance_scores


def cells():
    notebook = json.loads((Path(__file__).parents[1] / "docs/notebooks/downstream.ipynb").read_text())
    return {c["id"]: "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"}


@pytest.fixture
def scope():
    rng = np.random.default_rng(12)
    theta = np.array([[.03, .3, .02, .65], [.01, .5, .04, .45], [.1, .1, .1, .7]])
    topic_peaks = rng.uniform(0, 1, (4, 5))
    genes, peaks, tfs = ["g0", "g1", "g2"], [f"p{i}" for i in range(5)], ["TF0", "TF1"]
    names = [f"Topic_{i}" for i in range(4)]
    act = rng.uniform(0, 1, (4, 2, 3))
    rep = -rng.uniform(0, 1, (4, 2, 3))
    env = dict(np=np, pd=pd, model=SimpleNamespace(num_topics=4),
               rna_metacell=SimpleNamespace(var_names=genes, obs_names=["a", "b", "c"]),
               atac_metacell=SimpleNamespace(var_names=peaks), tf_names=tfs,
               scdori_latent=theta, topic_peak_embedding=topic_peaks.T,
               scaling_topic_ids=[1, 3], topic_gene_embedding=rng.uniform(0, 1, (2, 3)),
               gene_peak=rng.uniform(0, 1, (3, 5)), grn_act=act, grn_rep=rep,
               df_topic_celltype=pd.DataFrame({"A": theta[:2].mean(0), "B": theta[2]}, index=names).iloc[::-1],
               peak_mean_accessibility=pd.DataFrame({"A": [1, 2, 3, 0, 0], "B": [0, 0, 1, 4, 5]}, index=peaks),
               example_celltypes=["A", "B"], MIN_REGION_LINK=0., MIN_REGION_ACCESSIBILITY=0.,
               linked_peak_table=linked_peak_table, display=lambda x: None, display_topic_ids=[3], gene_name="g1")
    env["topic_tf_scores"] = {
        kind: tf_importance_scores(pd.DataFrame((sign * network).sum(axis=2)[[1, 3]],
                                              index=["Topic_1", "Topic_3"], columns=tfs))
        for kind, network, sign in [("activator", act, 1), ("repressor", rep, -1)]
    }
    exec(cells()["matrix-access"], env)
    return env


def test_labelled_views_and_examples_preserve_original_ids(scope):
    np.testing.assert_array_equal(scope["topic_peaks"].loc["Topic_3"], scope["topic_peak_embedding"][:, 3])
    np.testing.assert_array_equal(scope["peak_gene_links"]["g1"], scope["gene_peak"][1])
    assert scope["topic_genes"].index.tolist() == ["Topic_1", "Topic_3"]
    assert np.shares_memory(scope["peak_gene_links"].values, scope["gene_peak"])
    exec(cells()["matrix-topic-example"], scope)
    exec(cells()["matrix-regions-example"], scope)
    assert list(scope["regions_by_celltype"]) == ["A", "B"]


@pytest.mark.parametrize("kind, array", [("activator", "grn_act"), ("repressor", "grn_rep")])
def test_network_queries_preserve_sign_and_average_cells_without_thresholding(scope, kind, array):
    before = scope[array].copy()
    np.testing.assert_array_equal(scope["network_for_topic"](3, kind), before[3])
    actual = scope["network_for_celltype"]("A", kind)
    expected = np.einsum("ck,ktg->ctg", scope["scdori_latent"][:2], before).mean(0)
    np.testing.assert_allclose(actual, expected)
    rankings = scope["regulators_for_topic"](3, kind, n=1)
    assert len(rankings) == 1 and (rankings["raw"] > 0).all()
    np.testing.assert_array_equal(scope[array], before)


def test_invalid_queries_do_not_silently_switch_topics_or_signs(scope):
    for topic in [-1, 4, True, "3"]:
        with pytest.raises(ValueError, match="integer topic"):
            scope["network_for_topic"](topic)
    with pytest.raises(ValueError, match="outside the ranking reference"):
        scope["regulators_for_topic"](0)
    with pytest.raises(ValueError, match="kind"):
        scope["network_for_celltype"]("A", "both")


def test_multiple_cell_types_can_choose_different_topics_without_changing_reference():
    table = pd.DataFrame({"A": [.8, .2, 0.], "B": [.1, .7, .2], "C": [.1, .1, .8]},
                         index=["Topic_1", "Topic_3", "Topic_7"])
    original = table.copy()
    result = choose_example_topics(table, ["A", "B", "C"], n=1)
    assert result == {"A": ["Topic_1"], "B": ["Topic_3"], "C": ["Topic_7"]}
    pd.testing.assert_frame_equal(table, original)
