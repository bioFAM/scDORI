import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from scdori._core.exploration import apply_label_colors  # noqa: E402


def test_region_heatmap_keeps_all_37_labels_and_input_values():
    doc = json.loads((Path(__file__).parents[1] / 'docs/notebooks/downstream.ipynb').read_text())
    labels = [f'Population_{i}' for i in range(37)]
    table = pd.DataFrame(np.random.default_rng(0).normal(size=(9, 37)), columns=labels)
    before = table.copy()
    rna = ad.AnnData(np.ones((37, 2)), obs=pd.DataFrame({'celltype': labels}, index=labels))
    scope = {'sns': sns, 'pd': pd, 'plt': plt, 'apply_label_colors': apply_label_colors,
             'rna_metacell': rna, 'CELLTYPE_PALETTE': dict.fromkeys(labels, '#224466'),
             'peak_gene_celltype': table, 'enhancers': list(table.index), 'region_groups': labels}
    cell = next(c for c in doc['cells'] if c['id'] == '77ab6c54-b211-46b3-8ab9-20bf358365cc')
    exec(''.join(cell['source']), scope)
    grid = scope['region_grid']
    grid.fig.canvas.draw()
    assert [t.get_text() for t in grid.ax_heatmap.get_xticklabels()] == [x.replace('_', ' ') for x in labels]
    assert all(t.get_visible() for t in grid.ax_heatmap.get_xticklabels())
    pd.testing.assert_frame_equal(table, before)
    plt.close('all')
