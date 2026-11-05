import logging

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scdori.pp.gene_selection import compute_hvgs_and_tfs
from scdori.pp.peak_selection import keep_promoters_and_select_hv_peaks, retain_requested_peaks


@pytest.mark.parametrize('layout', [np.array, sparse.csr_matrix, sparse.csc_matrix])
@pytest.mark.parametrize('promoters', [True, False])
def test_requested_peak_survives_both_filters_without_count_or_order_changes(layout, promoters, caplog, tmp_path):
    counts = np.array([[0, 1, 3, 0], [0, 1, 0, 8]], dtype=float)
    a = ad.AnnData(layout(counts), obs=pd.DataFrame({'leiden': ['a', 'b']}, index=['c1', 'c2']),
                   var=pd.DataFrame({'promoter': [False, True, False, False]}, index=['outside', 'prom', 'variable', 'missing_window']))
    with caplog.at_level(logging.WARNING):
        windowed = retain_requested_peaks(a, ['prom', 'variable'], ['outside', 'outside', 'not_in_input'])
    assert 'not_in_input' in caplog.text and 'outside selected gene windows' in caplog.text
    result = keep_promoters_and_select_hv_peaks(windowed, 1, promoter_col='promoter',
                                              user_peaks=['outside'], keep_promoters=promoters, atac_count_mode='fragments')
    assert result.var_names.tolist() == (['outside', 'prom'] if promoters else ['outside'])
    dense = lambda x: x.toarray() if sparse.issparse(x) else x
    np.testing.assert_array_equal(dense(result.X), counts[:, :result.n_vars])
    np.testing.assert_array_equal(dense(a.X), counts)
    result.write_h5ad(tmp_path / 'atac.h5ad')
    restored = ad.read_h5ad(tmp_path / 'atac.h5ad')
    assert list(restored.uns['scdori_requested_peaks']['missing']) == ['not_in_input']


def test_requested_peaks_without_promoter_annotations_and_empty_request():
    a = ad.AnnData(np.array([[0., 2., 0.], [0., 0., 6.]]),
                   obs=pd.DataFrame({'leiden': ['a', 'b']}, index=['a', 'b']),
                   var=pd.DataFrame(index=['requested', 'p1', 'p2']))
    result = keep_promoters_and_select_hv_peaks(a, 2, user_peaks=['requested'], atac_count_mode='fragments')
    assert result.var_names.tolist() == ['requested', 'p2']
    windowed = retain_requested_peaks(a, ['p2', 'p1'])
    assert windowed.var_names.tolist() == ['p1', 'p2']
    assert keep_promoters_and_select_hv_peaks(a, 0, keep_promoters=False).n_vars == 0


def test_gene_tf_requests_are_retained_and_missing_names_reported(caplog, tmp_path):
    a = ad.AnnData(np.array([[1., 0., 4.], [0., 0., 4.]]), var=pd.DataFrame(index=['Gene', 'TF', 'NoMotif']))
    with caplog.at_level(logging.WARNING):
        out, genes, tfs = compute_hvgs_and_tfs(a, ['TF'], user_genes=['Gene', 'absent'],
                                              user_tfs=['TF', 'NoMotif', 'absent'], num_genes=2, num_tfs=1)
    assert genes == ['Gene'] and tfs == ['TF']  # requested TF retained even at zero expression
    assert 'absent after RNA filtering' in caplog.text and 'without matching motif' in caplog.text
    assert out.uns['scdori_requested_features']['tfs_without_motifs'] == ['NoMotif']
    out.write_h5ad(tmp_path / 'rna.h5ad')
    np.testing.assert_array_equal(out.X, a.X[:, :2])


def test_notebook_keeps_requested_peaks_when_gene_window_intersection_is_empty(tmp_path):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    notebook = json.loads((Path(__file__).parents[1] / 'docs/notebooks/preprocessing.ipynb').read_text())
    code = ''.join(notebook['cells'][21]['source'])
    a = ad.AnnData(np.ones((2, 2)), var=pd.DataFrame(index=['outside', 'other']))
    scope = {
        'out_dir': tmp_path, 'all_peaks_bed': tmp_path / 'all.bed', 'gene_bed_file': tmp_path / 'genes.bed',
        'run_bedtools_intersect': lambda **kw: kw['out_bed'].write_text(''), 'pd': pd, 'data_atac': a,
        'ppConfig': SimpleNamespace(peaks_user=['outside']), 'retain_requested_peaks': retain_requested_peaks,
        'logger': logging.getLogger(__name__),
    }
    exec(code, scope)
    assert scope['data_atac'].var_names.tolist() == ['outside']
