"""Library calls that replace hand-written v2.0.0 code must give bit-identical results."""

import numpy as np
import pandas as pd
import pytest
import torch

from scdori._core.interpretation import gene_importance_scores


def _v200_downstream_softmax(z):
    """The v2.0.0 expression in ``gene_importance_scores``, kept unchanged for comparison."""
    exp = np.exp(z - z.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _raw_scores(seed, topics, genes):
    rng = np.random.default_rng(seed)
    raw = rng.gamma(0.5, 1.0, (topics, genes)) * (rng.random((topics, genes)) < 0.5)  # many exact zeros
    raw[:, :3] = 0.0  # genes without signal in any topic (constant columns)
    raw[:, 3] = 7.0  # constant non-zero gene
    raw[:, 4:7] = raw[:, [4]]  # tied genes
    raw[:, 7] = 1e6 * raw[:, 8]  # very large values
    return pd.DataFrame(raw)


@pytest.mark.parametrize("seed,topics,genes", [(0, 1, 50), (1, 2, 20), (2, 8, 3000), (3, 40, 500), (4, 100, 200)])
def test_downstream_softmax_is_bitwise_the_v200_expression(seed, topics, genes):
    scores = gene_importance_scores(_raw_scores(seed, topics, genes))
    z = scores["across_topics_zscore"].to_numpy()
    new = scores["downstream_softmax"].to_numpy()
    old = _v200_downstream_softmax(z)
    assert new.dtype == old.dtype
    assert np.array_equal(new, old)


def test_downstream_softmax_is_bitwise_the_v200_expression_on_model_output():
    from scdori._core.models import scDoRI

    torch.manual_seed(71)
    model = scDoRI(torch.device("cpu"), 40, 120, 6, 12, 2, 16, 8)
    with torch.no_grad():
        raw = pd.DataFrame(model.topic_gene_matrix().numpy())
    scores = gene_importance_scores(raw)
    old = _v200_downstream_softmax(scores["across_topics_zscore"].to_numpy())
    assert np.array_equal(scores["downstream_softmax"].to_numpy(), old)
