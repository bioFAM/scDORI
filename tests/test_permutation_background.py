"""Permutation background of compute_atac_grn_*_with_significance, computed per chromosome.

The per-chromosome computation draws the same permutations as v2.0.0 and must give the same links, with
values equal up to floating-point rounding. When blocks are not valid, or the background has NaN, the
v2.0.0 computation runs unchanged.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scdori._core import downstream as ds
from scdori._core.downstream import (
    compute_atac_grn_activator_with_significance,
    compute_atac_grn_repressor_with_significance,
)
from scdori._core.models import scDoRI

CUTOFFS = (0.01, 0.05, 0.1, 0.2)
FUNCTIONS = {"activator": compute_atac_grn_activator_with_significance,
             "repressor": compute_atac_grn_repressor_with_significance}

# Ten peaks per label; labels include contigs, no-"chr" names and a mitochondrial label.
LABELS = ["chr1", "2", "chrX", "MT", "chr1_KI270706v1_random", "GL000194.1"]
CHROMOSOMES = np.repeat(LABELS, 10)


def make_model(seed=0, cross_link=False):
    """12 genes x 60 peaks; each linked gene links peaks of one label; gene 11 has no link."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = scDoRI(torch.device("cpu"), 12, 60, 5, 3, 1, 9, 5)
    fixed = np.zeros((12, 60), dtype=np.float32)
    for g in range(11):
        c = g % len(LABELS)
        fixed[g, rng.choice(np.arange(10 * c, 10 * c + 10), size=4, replace=False)] = rng.uniform(0.2, 1, 4)
    if cross_link:
        fixed[0, 15] = 0.5  # gene 0 links "chr1" and "2"
    activator = rng.uniform(0, 1, (60, 5)) * (rng.uniform(size=(60, 5)) < 0.6)
    repressor = -rng.uniform(0, 1, (60, 5)) * (rng.uniform(size=(60, 5)) < 0.6)
    with torch.no_grad():
        model.topic_peak_decoder.copy_(torch.from_numpy(rng.normal(size=(3, 60)).astype(np.float32)))
        model.gene_peak_factor_fixed.copy_(torch.from_numpy(fixed))
        model.gene_peak_factor_learnt.copy_(torch.from_numpy(rng.uniform(0.5, 1.5, (12, 60)).astype(np.float32)))
        model.tf_binding_matrix_activator.copy_(torch.from_numpy(activator.astype(np.float32)))
        model.tf_binding_matrix_repressor.copy_(torch.from_numpy(repressor.astype(np.float32)))
    return model


def released_v200(model, kind, cutoff, num_permutations, seed):
    """compute_atac_grn_*_with_significance of v2.0.0 (the loop copied unchanged); returns networks and cutoffs."""
    topic_peaks = torch.nn.Softmax(dim=1)(model.topic_peak_decoder)
    effective_gene_peak_factor = model.gene_peak_factor_fixed * model.gene_peak_factor_learnt
    chip = model.tf_binding_matrix_activator if kind == "activator" else model.tf_binding_matrix_repressor
    networks, cutoffs = [], []
    with torch.no_grad():
        for i in range(model.num_topics):
            generator = ds._permutation_generator(seed, i, chip.device)
            if kind == "activator":
                topic_gene_peak = (topic_peaks[i][:, None].T.clone()) * effective_gene_peak_factor
            else:
                topic_gene_peak = (1 / (topic_peaks[i].clone() + 1e-20))[:, None].T * effective_gene_peak_factor
            topic_gene_tf = torch.matmul(topic_gene_peak, chip)
            grn_fg = (topic_gene_tf / (effective_gene_peak_factor.sum(dim=1, keepdim=True) + 1e-8)).T
            grn_bg_topic = []
            for _ in range(num_permutations):
                chip_random = chip[torch.randperm(chip.size(0), device=chip.device, **generator)]
                topic_gene_tf_bg = torch.matmul(topic_gene_peak, chip_random)
                grn_bg = topic_gene_tf_bg / (effective_gene_peak_factor.sum(dim=1, keepdim=True) + 1e-8)
                grn_bg_topic.append(grn_bg.T.detach().cpu())
            grn_bg_topic = torch.stack(grn_bg_topic).cpu().numpy()
            q = 100 * (1 - cutoff) if kind == "activator" else 100 * (cutoff)
            cutoff1 = np.percentile(grn_bg_topic, q, axis=0)
            grn_fg1 = grn_fg.cpu().numpy()
            if kind == "activator":
                significant = np.where(grn_fg1 > cutoff1, grn_fg1, 0)
                significant = significant / (significant.max() + 1e-15)
            else:
                significant = np.where(grn_fg1 < cutoff1, grn_fg1, 0)
                significant = significant / (significant.min() + 1e-15)
            networks.append(significant)
            cutoffs.append(cutoff1)
    return np.stack(networks), np.stack(cutoffs)


def run(kind, model, path, seed, cutoff=0.1, n=70, **kw):
    torch.manual_seed(123)  # the global stream, used when seed is None
    out = FUNCTIONS[kind](model, "cpu", cutoff, path, n, seed=seed, **kw)
    return out, torch.get_rng_state()


def reference(kind, model, seed, cutoff=0.1, n=70):
    torch.manual_seed(123)
    networks, cutoffs = released_v200(model, kind, cutoff, n, seed)
    return networks, cutoffs, torch.get_rng_state()


@pytest.mark.parametrize("kind", ["activator", "repressor"])
@pytest.mark.parametrize("seed", [None, 7])
@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_chromosome_background_matches_v200(kind, seed, cutoff, tmp_path):
    model = make_model(1)
    expected, _, expected_state = reference(kind, model, seed, cutoff)
    if cutoff >= 0.05:
        assert expected.any()  # at 1% and 70 permutations some cases have no link at all
    actual, state = run(kind, model, tmp_path, seed, cutoff, peak_chromosomes=CHROMOSOMES)
    np.testing.assert_array_equal(actual != 0, expected != 0)  # the same links
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert torch.equal(state, expected_state)  # the random stream advanced exactly as in v2.0.0
    assert not actual[:, :, 11].any()  # the gene without links


@pytest.mark.parametrize("kind", ["activator", "repressor"])
@pytest.mark.parametrize("seed", [None, 7])
@pytest.mark.parametrize("max_elements", [2**24, 1000, 50])
def test_chromosome_cutoffs_match_numpy_percentile(kind, seed, max_elements, monkeypatch):
    monkeypatch.setattr(ds, "_QUANTILE_MAX_ELEMENTS", max_elements)  # 1000 // 70 = 14 rows per call; 50 -> 1 row
    model = make_model(2)
    _, expected, _ = reference(kind, model, seed, 0.05)
    tp = torch.softmax(model.topic_peak_decoder, dim=1)
    gpf = model.gene_peak_factor_fixed * model.gene_peak_factor_learnt
    chip = model.tf_binding_matrix_activator if kind == "activator" else model.tf_binding_matrix_repressor
    blocks, reason = ds._chromosome_blocks(gpf, CHROMOSOMES)
    assert reason is None
    torch.manual_seed(123)
    with torch.no_grad():
        for i in range(3):
            tgp = tp[i][:, None].T * gpf if kind == "activator" else (1 / (tp[i] + 1e-20))[:, None].T * gpf
            q = 100 * (1 - 0.05) if kind == "activator" else 100 * 0.05
            generator = ds._permutation_generator(seed, i, chip.device)
            actual = ds._chromosome_background_percentile(tgp, chip, gpf, 70, generator, q, blocks)
            assert actual.dtype == np.float64 and actual.shape == expected[i].shape
            np.testing.assert_allclose(actual, expected[i], rtol=1e-6, atol=1e-9)


def test_torch_quantile_float64_against_numpy_percentile():
    """The library runs torch.quantile in float64 on float32 backgrounds; numpy interpolates in float64 too."""
    rng = np.random.default_rng(0)
    x = (rng.exponential(size=(1000, 3000)) * (rng.uniform(size=(1000, 3000)) < 0.3)).astype(np.float32)
    for q in (0.80, 0.90, 0.95, 0.99, 0.01, 0.05, 0.10, 0.20):
        t = torch.quantile(torch.from_numpy(x).double(), torch.tensor([q], dtype=torch.float64), dim=0,
                           interpolation="linear")
        np.testing.assert_allclose(t[0].numpy(), np.percentile(x, 100 * q, axis=0), rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize(
    "names,expected",
    [
        (["chr1:10-20", "chr1:30-40", "chrM:1-5"], ["chr1", "chr1", "chrM"]),
        (["chr1-10-20", "chr2-30-40", "chrX-1-5"], ["chr1", "chr2", "chrX"]),
        (["1:10-20", "MT-1-5", "X:3-9"], ["1", "MT", "X"]),
        (["chr1_KI270706v1_random:5-9", "chrUn_GL000220v1-1-2", "GL000194.1-100-200", "KI270728.1:7-8"],
         ["chr1_KI270706v1_random", "chrUn_GL000220v1", "GL000194.1", "KI270728.1"]),
        (["HLA-A*01:01:01:01:10-20", "scaffold_12-3-4"], ["HLA-A*01:01:01:01", "scaffold_12"]),
        (["peak1", "chr1:10-20"], None),
        (["chr1:abc-def"], None),
        ([], None),
    ],
)
def test_peak_chromosome_labels_from_names(names, expected):
    labels = ds._peak_chromosomes(names)
    if expected is None:
        assert labels is None
    else:
        np.testing.assert_array_equal(labels, expected)


def test_peak_chromosome_labels_from_var():
    names = [f"peak{i}" for i in range(4)]  # names alone could not be parsed
    adata = SimpleNamespace(var=pd.DataFrame({"chr": pd.Categorical(["1", "1", "MT", "chr1_KI270706v1_random"])},
                                             index=names), var_names=pd.Index(names))
    np.testing.assert_array_equal(ds._peak_chromosomes(adata), ["1", "1", "MT", "chr1_KI270706v1_random"])
    adata.var.loc["peak2", "chr"] = np.nan
    assert ds._peak_chromosomes(adata) is None
    names = ["chr2:1-5", "2-7-9"]
    no_column = SimpleNamespace(var=pd.DataFrame(index=names), var_names=pd.Index(names))
    np.testing.assert_array_equal(ds._peak_chromosomes(no_column), ["chr2", "2"])


def test_blocks_cover_every_peak_and_every_linked_gene_once():
    model = make_model(3)
    gpf = model.gene_peak_factor_fixed * model.gene_peak_factor_learnt
    blocks, reason = ds._chromosome_blocks(gpf, CHROMOSOMES)
    assert reason is None and len(blocks) == len(LABELS)
    genes = np.concatenate([g.numpy() for g, _ in blocks])
    peaks = np.concatenate([p.numpy() for _, p in blocks])
    assert sorted(genes) == list(range(11))  # every linked gene exactly once; gene 11 has no link
    assert sorted(peaks) == list(range(60))  # every peak exactly once
    for g, p in blocks:
        assert len(set(CHROMOSOMES[p.numpy()])) == 1
        outside = np.setdiff1d(np.arange(60), p.numpy())
        assert not gpf[g][:, outside].any()


@pytest.mark.parametrize("kind", ["activator", "repressor"])
@pytest.mark.parametrize(
    "case,message",
    [
        ("cross_link", "1 genes link peaks on more than one chromosome label"),
        ("no_labels", "Peak chromosomes are not known"),
        ("wrong_length", "5 peak chromosome labels for 60 peaks"),
    ],
)
def test_invalid_blocks_use_the_v200_computation(kind, case, message, tmp_path, caplog):
    model = make_model(4, cross_link=case == "cross_link")
    kw = {"no_labels": {}, "wrong_length": {"peak_chromosomes": CHROMOSOMES[:5]}}.get(
        case, {"peak_chromosomes": CHROMOSOMES})
    expected, _, expected_state = reference(kind, model, None)
    with caplog.at_level(logging.WARNING, logger=ds.logger.name):
        actual, state = run(kind, model, tmp_path, None, **kw)
    assert any(message in r.getMessage() for r in caplog.records)
    assert actual.tobytes() == expected.tobytes()  # the unchanged v2.0.0 loop
    assert torch.equal(state, expected_state)


def test_model_attribute_supplies_the_labels(tmp_path, caplog):
    model = make_model(5)
    given, _ = run("activator", model, tmp_path / "a", 7, peak_chromosomes=CHROMOSOMES)
    model.peak_chromosomes = CHROMOSOMES  # what load_downstream_model sets
    with caplog.at_level(logging.WARNING, logger=ds.logger.name):
        attribute, _ = run("activator", model, tmp_path / "b", 7)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    np.testing.assert_array_equal(attribute, given)


@pytest.mark.parametrize("kind", ["activator", "repressor"])
@pytest.mark.parametrize("seed", [None, 7])
def test_nan_background_uses_the_v200_computation(kind, seed, tmp_path, caplog):
    model = make_model(6)
    with torch.no_grad():
        chip = model.tf_binding_matrix_activator if kind == "activator" else model.tf_binding_matrix_repressor
        chip[3, 2] = float("nan")
    expected, expected_cutoffs, expected_state = reference(kind, model, seed)
    assert np.isnan(expected_cutoffs).any()
    with caplog.at_level(logging.WARNING, logger=ds.logger.name):
        actual, state = run(kind, model, tmp_path, seed, peak_chromosomes=CHROMOSOMES)
    assert any("NaN in the permutation background" in r.getMessage() for r in caplog.records)
    assert actual.tobytes() == expected.tobytes()
    assert torch.equal(state, expected_state)
