import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scdori.pp.filtering import remove_mitochondrial_genes

MOUSE = {"mito": ["mt-Nd1", "mt-Co1", "mt-Atp6"], "other": ["Mtor", "Mtf1", "Gata1", "Nd1"]}
HUMAN = {"mito": ["MT-ND1", "MT-CO1", "MT-ATP6"], "other": ["MTOR", "MTHFR", "GATA1", "ND1"]}


def _rna(genes):
    X = np.arange(4 * len(genes), dtype=np.float32).reshape(4, len(genes))
    return ad.AnnData(X, var=pd.DataFrame(index=genes))


@pytest.mark.parametrize("prefix", ["MT-", "mt-", "Mt-"])
@pytest.mark.parametrize("species", [MOUSE, HUMAN], ids=["mouse", "human"])
def test_mitochondrial_prefix_ignores_case_on_both_sides(prefix, species):
    genes = species["other"][:2] + species["mito"] + species["other"][2:]
    rna = _rna(genes)
    kept = remove_mitochondrial_genes(rna, mito_prefix=prefix)
    assert list(kept.var_names) == species["other"]
    assert rna.var["mt"].tolist() == [gene in species["mito"] for gene in genes]
    np.testing.assert_array_equal(kept.X, rna[:, species["other"]].X)
    assert rna.n_vars == len(genes)  # the input keeps all genes


def test_default_prefix_still_removes_mouse_and_human_genes():
    for species in (MOUSE, HUMAN):
        kept = remove_mitochondrial_genes(_rna(species["mito"] + species["other"]))
        assert list(kept.var_names) == species["other"]
