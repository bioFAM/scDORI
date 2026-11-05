"""Run the actual tutorial branches for valid empty biological results."""
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scdori._core.exploration import (
    example_tf, top_topics_for_tf, top_targets, top_groups_for_topics, linked_peak_table,
)
from scdori._core.interpretation import tf_importance_scores
from scdori._core.downstream import compute_repressor_tf_activity_per_cell, save_regulons
from scdori._core.tf_preview import TFExpressionPreview


def cells():
    n = json.loads((Path(__file__).parents[1] / 'docs/notebooks/downstream.ipynb').read_text())
    return {c['id']: ''.join(c['source']) for c in n['cells'] if c['cell_type'] == 'code'}


@pytest.fixture
def scope(tmp_path):
    act = np.zeros((3, 2, 3)); rep = np.zeros_like(act)
    rna = ad.AnnData(np.ones((4, 3)), obs=pd.DataFrame({'celltype':['A','A','B','B']}, index=list('abcd')),
                    var=pd.DataFrame(index=['TFa','TFb','Gene']))
    atac = ad.AnnData(np.ones((4, 2)), obs=rna.obs.copy(), var=pd.DataFrame(index=['p0','p1']))
    theta = np.array([[0,1,0], [0,.6,.4], [0,.2,.8], [0,0,1.]])
    return dict(np=np, pd=pd, plt=plt, sns=sns, json=json, Path=Path,
                display=lambda v: None, GRN_OUTPUT=tmp_path, grn_act=act, grn_rep=rep,
                tf_names=['TFa','TFb'], rna_metacell=rna, atac_metacell=atac,
                model=SimpleNamespace(num_topics=3), scdori_latent=theta,
                topic_peak_embedding=np.ones((2,3))*.5, topic_gene_embedding=np.ones((2,3)),
                topic_tf_scores={k:tf_importance_scores(pd.DataFrame(np.ones((2,2)),
                    index=['Topic_1','Topic_2'], columns=['TFa','TFb'])) for k in ['activator','repressor']},
                gene_peak=np.ones((3,2)), scaling_topic_ids=[1,2], display_topic_ids=[1],
                df_topic_celltype=pd.DataFrame({'A':theta[:2].mean(0),'B':theta[2:].mean(0)},
                                              index=['Topic_0','Topic_1','Topic_2']),
                peak_mean_accessibility=pd.DataFrame({'A':[1.,2.],'B':[2.,1.]},index=['p0','p1']),
                peak_celltype_df=pd.DataFrame({'A':[-1.,1.],'B':[1.,-1.]},index=['p0','p1']),
                example_celltypes=['A'], example_topics={'A':['Topic_1']}, overview_celltypes=['A','B'],
                CELLTYPES_TO_SHOW=['A'], TFS_TO_EXPLORE=['TFb'], GENE_TO_EXPLORE=None,
                TARGETS_PER_TOPIC=2, TOPICS_PER_EXAMPLE=2, REGIONS_TO_SHOW=2,
                MIN_REGION_LINK=0., MIN_REGION_ACCESSIBILITY=0.,
                example_tf=example_tf, top_topics_for_tf=top_topics_for_tf, top_targets=top_targets,
                top_groups_for_topics=top_groups_for_topics, linked_peak_table=linked_peak_table)


TARGET_IDS=['1c60bfdc-65d5-4dc1-ac24-aeb700e4ba02','d8c87379-2d51-4fcf-87da-70d11cd24848',
            '6b7083de-4ddb-42c8-ad83-c6643bc489d3','26620136-73b4-45ad-8eb1-612bd08022de',
            'cc80f8d1-7f9f-4c9f-aa7b-052988183c6a','23e29d40-089c-44f6-8790-834b8b73f18f',
            '02e93229-d38a-450b-b3e3-a9fe7d629015']
REGION_IDS=['a9d612fd-0de9-469b-b064-8d71e4cf8ca0','dc8f8cbf-5b78-428f-8ff9-829fa35d276c',
            '77ab6c54-b211-46b3-8ab9-20bf358365cc','df14b650-c7c3-4144-b796-edfa9a5a3748',
            '28671ab3-e9a9-4c0d-865d-507cbb14dfd1','eb92ff16-5f5c-4312-9add-e0ef499848d9']


@pytest.mark.parametrize('case', ['repressor_only','missing_tf','no_links','automatic_repressor'])
def test_no_activating_example_reaches_matrix_exports(scope,case):
    source=cells()
    if case in ['repressor_only','automatic_repressor']:
        scope['grn_rep'][1,1,2]=-2
    if case=='missing_tf':scope['TFS_TO_EXPLORE']=['AbsentTF']
    if case=='automatic_repressor':scope['TFS_TO_EXPLORE']=[]
    before=scope['grn_rep'].copy()
    for id in TARGET_IDS+REGION_IDS+['matrix-access','matrix-topic-example','matrix-regions-example','matrix-exports']:
        exec(source[id],scope)
    assert scope['tf_plot'] is None and not scope['tf_targets']
    assert scope['target_table'].empty and scope['region_table'].empty
    if case in ['repressor_only','automatic_repressor']:
        assert any(row['Sign']=='repressor' and row['Leading targets']=='Gene' for row in scope['comparison_rows'])
    assert (scope['GRN_OUTPUT']/'matrix_exports/topic_genes.csv.gz').is_file()
    assert not (scope['GRN_OUTPUT']/'matrix_exports/gene_regions.csv.gz').exists()
    assert json.loads((scope['GRN_OUTPUT']/'matrix_exports/selection.json').read_text())['skipped_exports']==['gene_regions']
    np.testing.assert_array_equal(before,scope['grn_rep'])


@pytest.mark.parametrize('case',['no_links','no_accessibility','missing_gene'])
def test_gene_without_candidate_regions_skips_plots_but_exports_remain(scope,case):
    scope['tf_targets']={};scope['GENE_TO_EXPLORE']='Gene'
    if case=='no_links':scope['gene_peak'][:]=0
    if case=='no_accessibility':scope['peak_mean_accessibility'][:]=0
    if case=='missing_gene':scope['GENE_TO_EXPLORE']='NotFitted'
    source=cells()
    for id in REGION_IDS+['matrix-access','matrix-topic-example','matrix-regions-example','matrix-exports']:
        exec(source[id],scope)
    assert scope['region_table'].empty
    assert (scope['GRN_OUTPUT']/'matrix_exports/activator_rankings.csv.gz').exists()
    if case!='missing_gene':
        assert pd.read_csv(scope['GRN_OUTPUT']/'matrix_exports/gene_regions.csv.gz',index_col=0).empty


def test_supported_example_keeps_target_and_region_values(scope):
    scope['grn_act'][1,1]=[.2,0,.9]
    source=cells()
    for id in TARGET_IDS[:3]:exec(source[id],scope)
    assert scope['tf_plot']=='TFb'
    assert scope['target_table']['Target gene'].tolist()==['Gene','TFa']
    np.testing.assert_array_equal(scope['target_table']['Link weight'],[.9,.2])
    scope['GENE_TO_EXPLORE']='Gene'
    expected=linked_peak_table(scope['gene_peak'],scope['rna_metacell'].var_names,
        scope['atac_metacell'].var_names,'Gene',scope['peak_mean_accessibility'],['A'],n=2)
    exec(source[REGION_IDS[0]],scope)
    pd.testing.assert_frame_equal(scope['region_table'],expected)


def test_bulk_activity_assignment_aligns_rows_replaces_existing_and_can_repeat(scope):
    rna=scope['rna_metacell'];before=rna.X.copy()
    original_celltype=rna.obs['celltype'].copy()
    rna.obs['TFa_activity']=-1
    table=pd.DataFrame(np.arange(1200).reshape(4,300),index=list('dcba'),
                       columns=['TFa_activity']+[f'x{i}_activity' for i in range(299)])
    scope['df_cell_tf']=table
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for _ in range(2):exec(cells()['2c7a2cd6-906d-4410-9ae2-08aec64e7b29'],scope)
    assert not any(isinstance(w.message,pd.errors.PerformanceWarning) for w in caught)
    pd.testing.assert_frame_equal(rna.obs[table.columns],table.reindex(rna.obs_names))
    assert rna.obs.columns.is_unique
    pd.testing.assert_series_equal(rna.obs['celltype'],original_celltype)
    np.testing.assert_array_equal(rna.X,before)


def test_repressor_panel_requires_support_not_positive_zscore(scope):
    scope['df_celltype_tf_rep']=pd.DataFrame([[-2.,99.],[-1.,99.]],index=['A','B'],columns=['TFa','TFb'])
    scope.update(compute_repressor_tf_activity_per_cell=compute_repressor_tf_activity_per_cell,
                 tf_importance_scores=tf_importance_scores,activity_celltypes=['A','B'],
                 color_dict={'A':'#111111','B':'#eeeeee'})
    code=cells()['repressor-celltype-plot']
    exec(code,scope)
    assert scope['rep_tfs']==[]
    scope['grn_rep'][1,0,2]=-1
    exec(code,scope)
    assert scope['rep_tfs']==['TFa']  # needs repressive support; a high z-score alone does not count
    plt.close('all')


def test_regulon_summary_output_preserves_files(tmp_path,capsys):
    values=np.array([[[1,0],[0,2]],[[3,4],[5,0]]],dtype=float)
    for verbose,folder in [(True,'full'),(False,'summary')]:
        save_regulons(values,['A','B'],['g0','g1'],2,tmp_path/folder,verbose=verbose)
        output=capsys.readouterr().out
        assert len(output.splitlines())==(2 if verbose else 1)
    for file in (tmp_path/'full').rglob('*.tsv'):
        assert file.read_bytes()==(tmp_path/'summary'/file.relative_to(tmp_path/'full')).read_bytes()


def test_tf_summary_has_all_ten_names_and_escapes_html(monkeypatch):
    outputs=[]
    monkeypatch.setattr('IPython.display.display',outputs.append)
    names=[f'Long_TF_name_{i}' for i in range(9)]+['TF<script>']
    preview=TFExpressionPreview(pd.DataFrame([np.arange(1,11)],columns=names,index=['Topic_0']),[])
    before=pd.get_option('display.max_colwidth')
    preview.show()
    assert all(name in outputs[0].data for name in names[:-1])
    assert 'TF&lt;script&gt;' in outputs[0].data and '<script>' not in outputs[0].data
    assert '<details>' in outputs[1].data
    assert pd.get_option('display.max_colwidth')==before
