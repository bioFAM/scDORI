import copy
import json
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scdori._core.interpretation import tf_importance_scores
from scdori._core.tf_display import filter_tf_display, topic_expression_support
from scdori.pl.tf_scores import plot_tf_score_views


def fixture():
    # The first gene is NOT a TF: depth normalization must still include it.
    rna = ad.AnnData(np.array([[98, 2, 0], [8, 0, 2], [0, 0, 0]], dtype=float))
    rna.var_names = ['other', 'TF1', 'TF2']; rna.obs_names = ['c1', 'c2', 'c3']
    theta = pd.DataFrame([[.9, .1], [.8, .2], [.1, .9]], index=rna.obs_names, columns=['Topic_20', 'Topic_7'])
    return rna, theta


@pytest.mark.parametrize('layout', [np.asarray, sparse.csr_matrix, sparse.csc_matrix, sparse.csr_array, sparse.csc_array])
def test_expression_means_detection_and_gene_panel_depth(layout):
    rna, theta = fixture(); rna.X = layout(rna.X)
    evidence = topic_expression_support(rna, theta, ['TF2', 'TF1'], 2)
    np.testing.assert_allclose(evidence['mean_counts_per_10k'], [[1000, 100], [1000, 0]])
    np.testing.assert_allclose(evidence['detection_fraction'], [[.5, .5], [.5, 0]])
    assert evidence['selected_cells'] == {'Topic_20': ['c1', 'c2'], 'Topic_7': ['c2', 'c3']}
    assert list(evidence['mean_counts_per_10k'].columns) == ['TF2', 'TF1']
    assert evidence['normalization_gene_count'] == 3


def test_backed_and_zero_depth_and_small_group(tmp_path):
    rna, theta = fixture(); rna.X = sparse.csr_matrix(rna.X)
    path = tmp_path/'rna.h5ad'; rna.write_h5ad(path)
    backed = ad.read_h5ad(path, backed='r')
    try:
        result = topic_expression_support(backed, theta, ['TF1'], 99)
        np.testing.assert_allclose(result['mean_counts_per_10k'], [[200/3], [200/3]])
        np.testing.assert_allclose(result['detection_fraction'], [[1/3], [1/3]])
    finally:
        backed.file.close()


def test_refuse_misaligned_cells_and_transformed_counts():
    rna, theta = fixture()
    with pytest.raises(ValueError, match='cell names'):
        topic_expression_support(rna, theta.iloc[::-1], ['TF1'])
    rna.X[0, 1] = .2
    with pytest.raises(ValueError, match='integer counts'):
        topic_expression_support(rna, theta, ['TF1'])


def test_contribution_filter_removes_tiny_specific_without_rescaling():
    raw = pd.DataFrame([[100, .01, 0], [50, 0, 0]], columns=['major', 'tiny', 'absent'], index=['Topic_20','Topic_7'])
    scores = tf_importance_scores(raw); original = copy.deepcopy(scores)
    eligible, audit = filter_tf_display(scores, min_share=.001)
    assert scores['minmax_consensus'].loc['Topic_20','tiny'] > .5
    assert not eligible.loc['Topic_20','tiny']
    assert eligible['major'].all()
    assert not eligible['absent'].any()
    for key in scores:
        pd.testing.assert_frame_equal(scores[key], original[key])
    assert audit.query('topic == "Topic_20" and tf == "tiny"').iloc[0].reason == 'small contribution'
    default, _ = filter_tf_display(scores)
    pd.testing.assert_frame_equal(default, raw > 0)


def test_expression_cutoffs_are_inclusive_and_signs_independent():
    raw = pd.DataFrame([[1., 9]], index=['Topic_2'], columns=['low','high'])
    evidence = {'mean_counts_per_10k':pd.DataFrame([[.5, 20]], index=raw.index, columns=raw.columns),
                'detection_fraction':pd.DataFrame([[.1, .9]], index=raw.index, columns=raw.columns)}
    eligible, _ = filter_tf_display(tf_importance_scores(raw), expression=evidence, min_mean=.5, min_detection=.1, min_share=.1)
    assert eligible.all().all()
    opposite, _ = filter_tf_display(tf_importance_scores(raw.iloc[:,::-1].set_axis(['low','high'],axis=1)), expression=evidence, min_share=.2)
    assert opposite.loc['Topic_2','low'] and not opposite.loc['Topic_2','high']
    excluded, _ = filter_tf_display(tf_importance_scores(raw), expression=evidence, min_detection=.2)
    assert not excluded.loc['Topic_2','low']
    with pytest.raises(ValueError, match='required'):
        filter_tf_display(tf_importance_scores(raw), min_mean=1)
    with pytest.raises(ValueError, match='cover every'):
        filter_tf_display(tf_importance_scores(raw), expression={k:v.drop(columns='low') for k,v in evidence.items()})


def test_masked_plots_keep_same_order_and_scores():
    scores = tf_importance_scores(pd.DataFrame([[4,1], [1,4]], index=['a','b'], columns=['x','y']))
    eligible, _ = filter_tf_display(scores,min_share=.3)
    fig, before = plot_tf_score_views(scores,['a','b'],['x','y'])
    plt.close(fig)
    fig, after = plot_tf_score_views(scores,['a','b'],['x','y'],eligible=eligible)
    assert before == after
    panels = [ax for ax in fig.axes if ax.get_xlabel() == 'TF']
    assert len(panels) == 3
    for ax in panels:
        assert np.ma.getmaskarray(ax.collections[0].get_array()).sum() == 2
    plt.close(fig)


def test_notebook_optional_section_runs_and_exports_without_changing_scores(tmp_path):
    nb=json.loads((Path(__file__).parents[1]/'docs/notebooks/downstream.ipynb').read_text())
    source=next(''.join(c['source']) for c in nb['cells'] if c.get('id')=='tf-filter-review')
    rna,theta=fixture()
    raw=pd.DataFrame([[100., .01],[1,10]], index=theta.columns, columns=['TF1','TF2'])
    scores={kind:tf_importance_scores(raw) for kind in ['activator','repressor']}
    original=copy.deepcopy(scores)
    from types import SimpleNamespace
    namespace=dict(REVIEW_WEAK_TFS=True, GRN_OUTPUT=tmp_path, pd=pd, np=np, json=json, plt=plt,
                   rna_metacell=rna, scdori_latent=theta.values, model=SimpleNamespace(num_topics=2),
                   tf_score_reference=['Topic_0','Topic_1'], tf_score_display=['Topic_0','Topic_1'], tf_names=['TF1','TF2'],
                   trainConfig=SimpleNamespace(cells_per_topic=2), topic_tf_scores={k:{s:f.set_axis(['Topic_0','Topic_1']) for s,f in v.items()} for k,v in scores.items()},
                   MIN_TF_CONTRIBUTION_SHARE=.001, MIN_TF_MEAN_COUNTS_PER_10K=0., MIN_TF_DETECTION_FRACTION=0.,
                   TF_TOP_N=10, TF_MAX_PLOT_TFS=30, display=lambda x:None, plot_tf_score_views=plot_tf_score_views)
    exec(source,namespace)
    assert (tmp_path/'tf_display_review/activator_all_tfs.csv').is_file()
    assert (tmp_path/'tf_display_review/repressor_filtered.png').is_file()
    for kind in scores:
        for key in scores[kind]:
            pd.testing.assert_frame_equal(scores[kind][key], original[kind][key])
    namespace['MIN_TF_MEAN_COUNTS_PER_10K']=1e9
    exec(source,namespace)  # All filtered: no crash, full audit still written.
    assert not pd.read_csv(tmp_path/'tf_display_review/activator_all_tfs.csv').included.any()
    assert not (tmp_path/'tf_display_review/activator_filtered.png').exists()
