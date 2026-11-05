"""Run the optional comparison cells, including active and skipped examples."""
import json
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import seaborn as sns

from scdori._core.exploration import choose_example_topics
from scdori._core.interpretation import tf_importance_scores


def cells():
    n=json.loads((Path(__file__).parents[1]/'docs/notebooks/downstream.ipynb').read_text())
    return {c['id']: ''.join(c['source']) for c in n['cells'] if c['cell_type']=='code' and c['id'].startswith('lineage-')}


@pytest.fixture
def scope(tmp_path):
    rng=np.random.default_rng(1)
    topics=[f'Topic_{k}' for k in range(12)]
    groups=['NMP','Somitic_mesoderm','Spinal_cord']
    tfs=['Pax6','Sox2','Rfx4','Tbx6','Mesp2','Lef1','T','Hoxb9']
    genes=['Sox2','Pax6','Tbx6','Mesp1','Mesp2','Fgf4','Foxc2','Meox1']
    theta=rng.dirichlet(np.ones(12),18)
    cell_index=[f'c{i}' for i in range(18)]
    cell_groups=pd.Series(groups*6,index=cell_index)
    act=rng.uniform(size=(12,len(tfs),len(genes)));rep=-rng.uniform(size=act.shape)
    act[10:12,tfs.index('Sox2')]=0
    rep[10:12,tfs.index('Sox2')]=0
    act[8,tfs.index('T')]=0;rep[8,tfs.index('T')]=0
    scores={kind:tf_importance_scores(pd.DataFrame(sign*grn.sum(2),index=topics,columns=tfs)) for kind,grn,sign in [('activator',act,1),('repressor',rep,-1)]}
    obs=pd.DataFrame(theta,index=cell_index,columns=topics).assign(celltype=cell_groups)
    rna=ad.AnnData(np.ones((18,len(genes))),obs=obs,var=pd.DataFrame(index=genes),obsm={'X_umap':rng.normal(size=(18,2))})
    sc.settings.figdir=tmp_path
    env=dict(np=np,pd=pd,plt=plt,sns=sns,json=json,display=lambda x:None,
             choose_example_topics=choose_example_topics,df_topic_celltype=pd.DataFrame(theta,index=cell_index,columns=topics).groupby(cell_groups).mean().T,
             gene_scores_all=pd.DataFrame(index=topics),scaling_topic_ids=list(range(12)),sc=sc,rna_metacell=rna,
             topic_tf_scores=scores,GRN_OUTPUT=tmp_path,
             network_for_topic=lambda topic,kind='activator':pd.DataFrame((act if kind=='activator' else rep)[topic],index=tfs,columns=genes))
    yield env
    plt.close('all')


def test_active_example_plots_and_audits_preserve_inputs(scope):
    before=scope['df_topic_celltype'].copy()
    scores_before={kind:frames['minmax_consensus'].copy() for kind,frames in scope['topic_tf_scores'].items()}
    for source in cells().values():exec(source,scope)
    assert scope['comparison_ready']
    audit=scope['comparison_tf_audit']
    assert set(audit.loc[audit.TF=='Cdx2','status'])=={'not fitted'}
    assert audit.loc[(audit.TF=='Sox2')&(audit.topic=='Topic_10'),'rank'].isna().all()
    for name in ['topic_use.csv','tf_checks.csv','cell_topic_weights.csv']:
        assert (scope['comparison_dir']/name).exists()
    pd.testing.assert_frame_equal(scope['df_topic_celltype'],before)
    for kind,frame in scores_before.items():
        pd.testing.assert_frame_equal(scope['topic_tf_scores'][kind]['minmax_consensus'],frame)


def test_unrelated_dataset_skips_optional_example(scope):
    scope['df_topic_celltype'].columns=['A','B','C']
    for source in cells().values():exec(source,scope)
    assert not scope['comparison_ready']
    assert not (scope['GRN_OUTPUT']/'lineage_comparison').exists()


def test_auto_topic_suggestions_keep_reference_and_nonconsecutive_ids(scope):
    scope['gene_scores_all']=scope['gene_scores_all'].loc[['Topic_3','Topic_8','Topic_11']]
    source=cells()['lineage-choices'].replace('COMPARISON_TOPICS = [8, 10, 11]', 'COMPARISON_TOPICS = []')
    exec(source,scope)
    assert scope['comparison_ready']
    assert set(scope['comparison_topics']) <= {3,8,11}
    assert scope['scaling_topic_ids']==list(range(12))
