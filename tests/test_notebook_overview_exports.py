"""Overview display choices and named exports must preserve the fitted results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns


def sources():
    nb = json.loads((Path(__file__).parents[1] / "docs/notebooks/downstream.ipynb").read_text())
    return {c["id"]: "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"}


@pytest.mark.parametrize("labels", [["Cardiomyocytes", "Erythroid3", "Endothelium"], ["Population A", "Population B"]])
def test_overview_uses_retained_topics_and_keeps_detailed_example(labels):
    # Topic 1 is excluded even though its usage is largest.
    frame = pd.DataFrame(np.arange(4 * len(labels)).reshape(4, -1),
                         index=["Topic_1", "Topic_3", "Topic_8", "Topic_20"], columns=labels).astype(float)
    frame.loc["Topic_1"] = 100
    before = frame.copy()
    env = dict(pd=pd, plt=plt, sns=sns, df_topic_celltype=frame,
               scaling_topic_ids=[3, 8, 20], display_topic_ids=[20], CELLTYPES_TO_SHOW=["Erythroid3"],
               OVERVIEW_CELLTYPES=["Erythroid3", "Cardiomyocytes", "Endothelium"])
    source = sources()["overview-panel-choices"]
    exec(source, env)
    assert set(env["overview_celltypes"]) == set(labels)
    assert "Topic_1" not in env["overview_topics"]
    assert len(env["overview_celltypes"]) > 1
    assert env["display_topic_ids"] == [20]
    assert env["CELLTYPES_TO_SHOW"] == ["Erythroid3"]
    pd.testing.assert_frame_equal(frame, before)
    plt.close("all")


def test_complete_topic_overview_has_all_selected_rows_and_celltype_colors(tmp_path):
    frame = pd.DataFrame([[.3,.1,.01], [.04,.2,.08], [.01,.02,.4]],
                         index=["Topic_3","Topic_8","Topic_20"], columns=["Rare","Cardiac","Blood"])
    colors={"Rare":"#ff0000","Cardiac":"#00ff00","Blood":"#0000ff"}
    env=dict(pd=pd,sns=sns,plt=plt,df_topic_celltype=frame,select_topics=["Topic_3","Topic_20"],
             color_dict=colors,GRN_OUTPUT=tmp_path)
    source=next(s for s in sources().values() if "TOPIC_COLOUR_MIN =" in s)
    exec(source,env)
    assert set(env['g'].data2d.index)=={"Topic_3","Topic_20"}
    assert set(env['g'].data2d.columns)==set(frame.columns)
    assert env['g'].ax_heatmap.collections[0].get_clim()==(.05,.25)
    assert env['col_colors']['Cell type'].to_dict()==colors
    assert env['g'].ax_col_colors is not None
    assert (tmp_path/'topic_celltype_overview.png').is_file()
    plt.close('all')


@pytest.mark.parametrize("font_scale", [0.8, 1.5])
def test_celltype_activity_labels_do_not_depend_on_previous_plot_style(tmp_path, font_scale):
    groups=[f"Group_{i}" for i in range(8)]
    tfs=[f"TF{i}" for i in range(24)]
    frame=pd.DataFrame(np.random.default_rng(7).uniform(size=(8,24)),index=groups,columns=tfs)
    env=dict(pd=pd,sns=sns,plt=plt,celltype_tf_scores=frame,activity_celltypes=groups,
             color_dict={g:"#336699" for g in groups},tf_list_plot=tfs,GRN_OUTPUT=tmp_path)
    source=sources()["18dd3cae-45fe-4a1e-a58b-4fd38ce7c68f"]
    # Earlier peak maps use a large font; the heatmap must still label every TF.
    with sns.plotting_context("notebook",font_scale=font_scale):
        exec(source,env)
        assert sorted(t.get_text() for t in env['g'].ax_heatmap.get_yticklabels())==sorted(tfs)
        assert sorted(t.get_text() for t in env['g'].ax_heatmap.get_xticklabels())==sorted(groups)
    pd.testing.assert_frame_equal(env['activity_to_plot'],frame.T)
    plt.close('all')


def test_export_roundtrip_preserves_labels_and_negative_edges(tmp_path):
    frame = pd.DataFrame([[-.25, 0.], [0., -.75]], index=["TF_B", "TF_A"], columns=["Gene_Z", "Gene_Y"])
    source = sources()["matrix-exports"].replace(
        'EXPORTS = ["topic_genes", "activator_rankings", "repressor_rankings", "gene_regions"]',
        'EXPORTS = ["repressor_network"]')
    env = dict(Path=Path, json=json, GRN_OUTPUT=tmp_path, TOPIC=20, GENE="Gene_Z",
               example_celltypes=["Population A"], scaling_topic_ids=[3,20], display=lambda x:None,
               network_for_topic=lambda topic, kind: frame)
    exec(source, env)
    saved = pd.read_csv(tmp_path / "matrix_exports/repressor_network_topic_20.csv.gz", index_col=0)
    pd.testing.assert_frame_equal(saved, frame)
    selection=json.loads((tmp_path / "matrix_exports/selection.json").read_text())
    assert selection["score_reference_topics"] == [3,20]
    assert selection["topic"] == 20


def test_unknown_export_fails_before_writing_tables(tmp_path):
    source = sources()["matrix-exports"].replace(
        'EXPORTS = ["topic_genes", "activator_rankings", "repressor_rankings", "gene_regions"]',
        'EXPORTS = ["topic_genes", "typo"]')
    with pytest.raises(ValueError, match="Unknown export"):
        exec(source, dict(Path=Path, json=json, GRN_OUTPUT=tmp_path))
    assert not list(tmp_path.rglob("*.csv.gz"))
