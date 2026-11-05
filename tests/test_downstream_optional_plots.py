"""Disabling duplicate figures must preserve every numerical result."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
import seaborn as sns
from scdori.pl.plotting import plot_topic_activation_heatmap
from scdori._core.downstream import get_top_activators_per_topic, get_top_repressor_per_topic


def test_group_table_without_plot_is_identical(monkeypatch):
    adata=SimpleNamespace(obsm={'X_scdori':np.array([[.8,.2],[.4,.6],[.1,.9]])},
                          obs=pd.DataFrame({'celltype':['A','A','B']}))
    calls=[]
    monkeypatch.setattr(sns,'clustermap',lambda *a,**k:calls.append(a))
    monkeypatch.setattr(plt,'show',lambda:None)
    plotted=plot_topic_activation_heatmap(adata,groupby_key=['celltype'],aggregation='mean')
    table=plot_topic_activation_heatmap(adata,groupby_key=['celltype'],aggregation='mean',plot=False)
    pd.testing.assert_frame_equal(plotted,table)
    np.testing.assert_allclose(table['A'],[.6,.4])
    assert len(calls)==1


@pytest.mark.parametrize('sign',[1,-1])
def test_regulator_scores_without_plot_are_identical(sign,monkeypatch):
    grn=np.random.default_rng(2).uniform(size=(5,8,10))*sign
    before=grn.copy()
    names=[f'TF{i}' for i in range(8)]
    calls=[]
    monkeypatch.setattr(sns,'clustermap',lambda *a,**k:calls.append(a))
    monkeypatch.setattr(plt,'show',lambda:None)
    fn=get_top_activators_per_topic if sign==1 else get_top_repressor_per_topic
    a,ta=fn(grn,names,None,selected_topics=[1,3,4],display_topics=[3],top_k=2)
    b,tb=fn(grn,names,None,selected_topics=[1,3,4],display_topics=[3],top_k=2,plot=False)
    pd.testing.assert_frame_equal(a,b)
    assert ta==tb and len(calls)==1
    np.testing.assert_array_equal(grn,before)
