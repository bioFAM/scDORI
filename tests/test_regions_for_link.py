import numpy as np
import pytest
import torch

from scdori._core.downstream import compute_atac_grn_activator_with_significance
from scdori._core.exploration import regions_for_link
from scdori._core.models import scDoRI

TFS, GENES, PEAKS = ["T0", "T1", "T2"], [f"g{i}" for i in range(7)], [f"chr1:{i}-{i + 1}" for i in range(11)]


@pytest.fixture
def model():
    torch.manual_seed(0)
    m = scDoRI(torch.device("cpu"), 7, 11, 3, 4, 2, 9, 5)
    with torch.no_grad():
        m.gene_peak_factor_fixed.copy_((torch.rand(7, 11) > 0.4).float())
        act = torch.rand(11, 3)
        act[act < 0.3] = 0
        m.tf_binding_matrix_activator.copy_(act)
        m.tf_binding_matrix_repressor.copy_(-torch.rand(11, 3))
    return m


def full_scores(m, sign):
    """Vectorised ATAC foreground, written as in compute_atac_grn_*_with_significance."""
    acc = torch.softmax(m.topic_peak_decoder, dim=1)
    gp = m.gene_peak_factor_learnt * m.gene_peak_factor_fixed
    binding = m.tf_binding_matrix_activator if sign == "activator" else m.tf_binding_matrix_repressor
    out = []
    for i in range(m.num_topics):
        term = acc[i] if sign == "activator" else 1 / (acc[i] + 1e-20)
        out.append((((term[:, None].T * gp) @ binding) / (gp.sum(dim=1, keepdim=True) + 1e-8)).T)
    return torch.stack(out).detach().numpy()  # topics × TFs × genes


@pytest.mark.parametrize("sign", ["activator", "repressor"])
def test_contributions_sum_to_the_atac_score(model, sign):
    expected = full_scores(model, sign)
    for topic in range(4):
        for t, tf in enumerate(TFS):
            for g, gene in enumerate(GENES):
                regions, summary = regions_for_link(model, tf, gene, topic, TFS, GENES, PEAKS, sign=sign)
                assert summary.atac_score == pytest.approx(expected[topic, t, g], rel=1e-5, abs=1e-9)
                assert regions.contribution.sum() == pytest.approx(summary.atac_score, rel=1e-6, abs=1e-12)


def test_rows_are_linked_regions_ranked_by_contribution(model):
    regions, summary = regions_for_link(model, "T0", "g3", 1, TFS, GENES, PEAKS)
    linked = {PEAKS[p] for p in np.flatnonzero(model.gene_peak_factor_fixed[3].detach().numpy() > 0)}
    assert set(regions.index) == linked and summary.linked_regions == len(linked)
    assert (np.diff(regions.contribution.abs().to_numpy()) <= 1e-12).all()
    assert regions.share.sum() == pytest.approx(1.0)
    assert regions.accessibility_percentile_in_topic.between(0, 100).all()
    assert len(regions_for_link(model, "T0", "g3", 1, TFS, GENES, PEAKS, n=2)[0]) == min(2, len(linked))


def test_no_binding_gives_no_support(model):
    with torch.no_grad():
        model.tf_binding_matrix_activator[:, 2] = 0
    regions, summary = regions_for_link(model, "T2", "g0", 0, TFS, GENES, PEAKS)
    assert summary.atac_score == 0 and summary.regions_with_binding == 0
    assert (regions.contribution == 0).all() and (regions.share == 0).all()


def test_repressor_support_is_negative_and_uses_inverse_accessibility(model):
    regions, summary = regions_for_link(model, "T1", "g2", 2, TFS, GENES, PEAKS, sign="repressor")
    assert summary.atac_score < 0 and (regions.contribution <= 0).all()
    acc = torch.softmax(model.topic_peak_decoder[2], dim=0).detach().numpy()
    closed = PEAKS[int(np.argmin(np.where(model.gene_peak_factor_fixed[2].detach().numpy() > 0, acc, np.inf)))]
    assert regions.loc[closed, "topic_accessibility"] == regions.topic_accessibility.min()


def test_summary_reads_saved_networks_without_changing_them(model, tmp_path):
    atac = compute_atac_grn_activator_with_significance(
        model, torch.device("cpu"), cutoff_val=0.5, outdir=tmp_path, num_permutations=20, seed=7)
    before = atac.copy()
    gate = np.full((4, 3), 0.25)
    final = np.full((4, 3, 7), -1.0)
    scores = full_scores(model, "activator")
    for topic in range(4):
        passed = atac[topic] != 0
        if passed.any():
            # The saved network is the passing scores divided by the topic maximum.
            ratio = atac[topic][passed] / scores[topic][passed]
            assert np.allclose(ratio, ratio[0], rtol=1e-4)
    _, summary = regions_for_link(model, "T1", "g4", 3, TFS, GENES, PEAKS, atac_grn=atac, gate=gate, final_grn=final)
    assert summary.passed_background_filter == (atac[3, 1, 4] != 0)
    assert summary.tf_expression_gate == 0.25 and summary.final_network_value == -1.0
    assert summary.phase2_weight == pytest.approx(torch.relu(model.tf_gene_topic_activator_grn[3, 1, 4]).item())
    np.testing.assert_array_equal(atac, before)


@pytest.mark.parametrize("kwargs", [{"tf": "missing"}, {"gene": "missing"}, {"topic": 4}, {"topic": 1.0},
                                    {"sign": "both"}, {"n": 0}])
def test_rejects_invalid_requests(model, kwargs):
    call = {"tf": "T0", "gene": "g0", "topic": 0, "sign": "activator", "n": None, **kwargs}
    with pytest.raises(ValueError):
        regions_for_link(model, call["tf"], call["gene"], call["topic"], TFS, GENES, PEAKS,
                         sign=call["sign"], n=call["n"])


def test_gate_can_be_the_tensor_held_by_the_downstream_notebook(model):
    gate = np.full((4, 3), 0.5, dtype=np.float32)
    _, from_array = regions_for_link(model, "T0", "g0", 0, TFS, GENES, PEAKS, gate=gate)
    _, from_tensor = regions_for_link(model, "T0", "g0", 0, TFS, GENES, PEAKS, gate=torch.as_tensor(gate))
    assert from_array["tf_expression_gate"] == from_tensor["tf_expression_gate"] == 0.5
