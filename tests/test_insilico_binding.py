"""Motif-weighted in-silico ChIP-seq (used since v2.1.0): motif p-values, weights, background, sign, range, notebook."""

import json
import math

import numpy as np
import pytest

from scdori import compute_in_silico_chipseq_weighted, compute_motif_pvalues, load_motif_database
from scdori.pp.correlation import peak_tf_correlation
from scdori.pp.motif_scanning import motif_pvalue_floor


def _reference_binding(r, L, hit_p=1e-3, weight_top_pct=5.0, bg_percentile=99.0):
    """The benchmarked implementation, written out per TF."""
    m = -math.log10(hit_p)
    H = L >= m
    W = np.zeros_like(L)
    for j in range(L.shape[1]):
        h = H[:, j]
        if h.any():
            q = np.percentile(L[h, j], 100 - weight_top_pct)
            W[h, j] = np.clip((L[h, j] - m) / max(q - m, 1e-9), 0, 1)
    ra, rr = np.clip(r, 0, None), np.clip(-r, 0, None)
    act, rep = np.zeros_like(r), np.zeros_like(r)
    for j in range(r.shape[1]):
        bg = ~H[:, j]
        if bg.sum() < 100:
            bg = np.zeros_like(bg)
            bg[np.argsort(L[:, j], kind="stable")[:5000]] = True
        ka = H[:, j] & (ra[:, j] > np.percentile(ra[bg, j], bg_percentile))
        kr = H[:, j] & (rr[:, j] > np.percentile(rr[bg, j], bg_percentile))
        act[ka, j] = W[ka, j] * ra[ka, j]
        rep[kr, j] = -W[kr, j] * rr[kr, j]
    return act, rep


def _random_inputs(seed=0, n_peaks=3000, n_tfs=6, hit_share=0.3):
    rng = np.random.default_rng(seed)
    r = rng.uniform(-0.9, 0.9, size=(n_peaks, n_tfs))
    hit = rng.random((n_peaks, n_tfs)) < hit_share
    L = np.where(hit, 3 + rng.exponential(2.0, size=(n_peaks, n_tfs)), 0.0)
    return r, L


def test_binding_equals_the_benchmarked_per_tf_computation():
    r, L = _random_inputs()
    L[:, 5] = np.where(np.arange(L.shape[0]) < 2950, 3 + np.arange(L.shape[0]) / 1000, 0)  # 50 no-hit peaks: fallback
    act, rep, _ = compute_in_silico_chipseq_weighted(r, L)
    ref_act, ref_rep = _reference_binding(r, L)
    np.testing.assert_array_equal(act, ref_act)
    np.testing.assert_array_equal(rep, ref_rep)


def test_weight_is_zero_at_the_cutoff_one_at_the_top_and_linear_between():
    n = 1000
    L = np.zeros((n, 1))
    L[:200, 0] = np.linspace(3, 13, 200)  # hits from the cutoff (p = 1e-3) up to 1e-13
    r = np.full((n, 1), 0.5)
    r[200:, 0] = 0.0  # background without any correlation, so every hit passes
    act, rep, info = compute_in_silico_chipseq_weighted(r, L)
    top = np.percentile(L[:200, 0], 95)
    assert info["weight_top_logp"][0] == pytest.approx(top)
    weight = act[:200, 0] / 0.5
    assert weight[0] == 0  # at the cutoff
    np.testing.assert_allclose(weight[L[:200, 0] >= top], 1.0)
    between = (L[:200, 0] > 3) & (L[:200, 0] < top)
    np.testing.assert_allclose(weight[between], (L[:200, 0][between] - 3) / (top - 3))
    assert not rep.any()


def test_background_is_the_peaks_without_a_motif_hit():
    r, L = _random_inputs(seed=1)
    _, _, info = compute_in_silico_chipseq_weighted(r, L)
    for j in range(r.shape[1]):
        background = L[:, j] < 3
        assert info["cutoff_activator"][j] == np.percentile(np.clip(r[background, j], 0, None), 99)
        assert info["cutoff_repressor"][j] == np.percentile(np.clip(-r[background, j], 0, None), 99)


def test_background_falls_back_to_the_weakest_hits_when_few_peaks_lack_a_hit():
    rng = np.random.default_rng(2)
    n = 400
    L = np.full((n, 1), 0.0)
    L[50:, 0] = 3 + rng.exponential(2.0, size=n - 50)  # 50 peaks without a hit, fewer than 100
    r = rng.uniform(-1, 1, size=(n, 1))
    _, _, info = compute_in_silico_chipseq_weighted(r, L, n_bg_fallback=120)
    weakest = np.argsort(L[:, 0], kind="stable")[:120]
    assert info["cutoff_activator"][0] == np.percentile(np.clip(r[weakest, 0], 0, None), 99)
    _, _, enough = compute_in_silico_chipseq_weighted(r, L, n_bg_fallback=120, min_background=50)
    assert enough["cutoff_activator"][0] == np.percentile(np.clip(r[:50, 0], 0, None), 99)


def test_sign_and_unscaled_range():
    r, L = _random_inputs(seed=3)
    act, rep, _ = compute_in_silico_chipseq_weighted(r, L)
    assert (act >= 0).all() and (rep <= 0).all()
    assert (act[act > 0] <= r[act > 0]).all()  # weight x r with weight <= 1
    assert (-rep[rep < 0] <= -r[rep < 0]).all()
    assert (r[act > 0] > 0).all() and (r[rep < 0] < 0).all()
    assert not ((act != 0) & (L < 3)).any() and not ((rep != 0) & (L < 3)).any()  # only motif hits
    assert act.max() < r.max()  # not rescaled per TF to reach 1
    assert act.any() and rep.any()


def test_bad_inputs_fail():
    r, L = _random_inputs()
    with pytest.raises(ValueError, match="same shape"):
        compute_in_silico_chipseq_weighted(r[:, :2], L)
    with pytest.raises(ValueError):
        compute_in_silico_chipseq_weighted(r, L, weight_top_pct=0)


def test_peak_tf_correlation_keeps_the_released_arithmetic():
    rng = np.random.default_rng(4)
    atac, rna = rng.random((30, 8)), rng.random((30, 5))
    atac[:, 2] = 1.0  # a constant peak correlates 0
    X = (atac - atac.mean(axis=0)) / (atac.std(axis=0) + 1e-8)
    Y = (rna - rna.mean(axis=0)) / (rna.std(axis=0) + 1e-8)
    np.testing.assert_array_equal(peak_tf_correlation(atac, rna), np.nan_to_num((X.T @ Y) / 30))
    assert not peak_tf_correlation(atac, rna)[2].any()


def _private_fimo_floor(pwm):
    """The last finite entry of tangermeme's own score table (private; used here only to check the public rule)."""
    import inspect

    from tangermeme.tools.fimo import _pwm_to_mapping, fimo

    defaults = inspect.signature(fimo).parameters
    lo = np.log2(np.asarray(pwm, dtype=np.float64) + defaults["eps"].default) - math.log2(0.25)
    _, table = _pwm_to_mapping(lo, defaults["bin_size"].default)
    return 2.0 ** table[np.isfinite(table)][-1]


@pytest.mark.parametrize("seed", range(5))
def test_pvalue_floor_is_fimos_smallest_reported_p(seed):
    rng = np.random.default_rng(seed)
    width = int(rng.integers(5, 15))
    pwm = rng.dirichlet(np.full(4, 0.3), size=width).T  # alphabet x positions
    if seed == 0:
        pwm[:, 0] = 0.25  # a position with four tied bases
    assert math.log10(motif_pvalue_floor(pwm)) == pytest.approx(math.log10(_private_fimo_floor(pwm)), abs=1e-12)


def _write_meme(path, pwm):
    path.write_text(
        "MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"
        "Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n"
        f"MOTIF motif1 TestTF_1\nletter-probability matrix: alength= 4 w= {len(pwm)} nsites= 20 E= 0\n"
        + "\n".join(" ".join(map(str, row)) for row in pwm)
        + "\n\n"
    )


def test_motif_pvalues_take_fimos_minimum_p_and_floor_perfect_matches(tmp_path):
    from tangermeme.io import extract_loci
    from tangermeme.tools.fimo import fimo
    import pandas as pd

    rng = np.random.default_rng(0)
    genome = rng.choice(list("ACGT"), size=2000)
    genome[118:124] = list("ACGTAC")  # the consensus: a perfect match
    genome[1118:1124] = list("ACGTAA")  # one mismatch
    (tmp_path / "genome.fa").write_text(">chr1\n" + "".join(genome) + "\n")
    bed = tmp_path / "peaks.bed"
    bed.write_text("chr1\t100\t140\tperfect\nchr1\t1100\t1140\tclose\nchr1\t600\t640\tother\n")
    pwm = np.full((6, 4), 0.01)
    pwm[np.arange(6), [0, 1, 2, 3, 0, 1]] = 0.97
    _write_meme(tmp_path / "motifs.meme", pwm)
    motifs, names = load_motif_database(tmp_path / "motifs.meme", ["TestTF"])
    logp = compute_motif_pvalues(bed, tmp_path / "genome.fa", motifs, names, n_peaks=3, window=40, threshold=1e-2)
    assert list(logp.index) == ["perfect", "close", "other"] and list(logp.columns) == ["TestTF"]
    floor = motif_pvalue_floor(next(iter(motifs.values())))
    assert logp.loc["perfect", "TestTF"] == pytest.approx(-math.log10(floor))  # FIMO's p here equals the floor
    loci = pd.read_csv(bed, sep="\t", header=None).iloc[:, :4]
    loci.columns = ["chr", "start", "end", "peak_name"]
    X = extract_loci(loci, str(tmp_path / "genome.fa"), in_window=40).float()
    hits = fimo(motifs, X, threshold=1e-2)[0]
    p_close = hits.loc[hits.sequence_name == 1, "p-value"].min()
    assert 1e-3 < p_close < 1e-2  # one mismatch in a 6-bp motif
    assert logp.loc["close", "TestTF"] == pytest.approx(-math.log10(p_close))
    other = hits.loc[hits.sequence_name == 2, "p-value"]
    expected_other = -math.log10(other.min()) if len(other) else 0.0
    assert logp.loc["other", "TestTF"] == pytest.approx(expected_other)


def test_preprocessing_notebook_uses_the_weighted_binding():
    from pathlib import Path

    notebook = json.loads((Path(__file__).parents[1] / "docs/notebooks/preprocessing.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "compute_motif_pvalues(" in code and "compute_in_silico_chipseq_weighted(" in code
    assert "compute_motif_scores(" not in code and "compute_in_silico_chipseq(" not in code


def test_averaged_correlation_is_the_mean_over_clusterings_and_leaves_inputs_unchanged():
    import anndata as ad

    from scdori import averaged_metacell_correlation

    rng = np.random.default_rng(5)
    n = 240
    groups = np.repeat(np.arange(6), n // 6)
    rna = ad.AnnData(rng.poisson(2 + groups[:, None] % 3, size=(n, 30)).astype(np.float32))
    rna.var_names = [f"g{i}" for i in range(30)]
    rna.var["gene_type"] = ["TF" if i < 4 else "gene" for i in range(30)]
    rna.obs["sample"] = np.where(np.arange(n) % 2 == 0, "s1", "s2")
    atac = ad.AnnData(rng.poisson(1 + groups[:, None] % 2, size=(n, 25)).astype(np.float32))
    atac.var_names = [f"chr1:{i}-{i + 1}" for i in range(25)]
    atac.obs_names = rna.obs_names
    rna_before, atac_before = rna.X.copy(), atac.X.copy()
    kwargs = dict(resolution=2, batch_key="sample")
    both = averaged_metacell_correlation(rna, atac, atac.var_names, n_clusterings=2, first_seed=101, **kwargs)
    first = averaged_metacell_correlation(rna, atac, atac.var_names, n_clusterings=1, first_seed=101, **kwargs)
    second = averaged_metacell_correlation(rna, atac, atac.var_names, n_clusterings=1, first_seed=102, **kwargs)
    assert both.dtype == np.float32 and both.shape == (25, 4)
    np.testing.assert_allclose(both, (first.astype(np.float64) + second) / 2, atol=1e-6)
    np.testing.assert_array_equal(rna.X, rna_before)
    np.testing.assert_array_equal(atac.X, atac_before)
    assert "scdori_averaged_leiden" not in rna.obs


def test_windows_of_n_bases_are_no_hit_but_real_negative_score_hits_count(tmp_path):
    """FIMO skips N bases, so an all-N window scores 0 and passes the cutoff of a long low-information motif.

    All-N windows are dropped; real hits with negative scores are kept, also in a peak that has all-N windows
    (the benchmarked implementation set such peaks to no hit).
    """
    from tangermeme.io import extract_loci
    from tangermeme.tools.fimo import fimo
    import pandas as pd

    consensus = "ACGTACGTACGTACGTACGT"
    near = list(consensus)
    for i in (1, 4, 7, 10, 13, 16, 19):  # 7 mismatches: a real hit with a negative score
        near[i] = "T" if near[i] != "T" else "A"
    genome = ["T"] * 1000
    genome[100:200] = "N" * 100
    genome[520:540] = near
    genome[590:620] = "N" * 30
    genome[620:640] = near
    (tmp_path / "genome.fa").write_text(">chr1\n" + "".join(genome) + "\n")
    bed = tmp_path / "peaks.bed"
    bed.write_text("chr1\t130\t170\tn_run\nchr1\t510\t550\tnegative\nchr1\t610\t650\tmixed\nchr1\t800\t840\tnothing\n")
    pwm = np.full((20, 4), 0.01)
    pwm[np.arange(20), ["ACGT".index(b) for b in consensus]] = 0.97
    _write_meme(tmp_path / "motifs.meme", pwm)
    motifs, names = load_motif_database(tmp_path / "motifs.meme", ["TestTF"])
    loci = pd.read_csv(bed, sep="\t", header=None)
    loci.columns = ["chr", "start", "end", "peak_name"]
    X = extract_loci(loci, str(tmp_path / "genome.fa"), in_window=60).float()  # fimo scans starts 0..39
    hits = fimo(motifs, X, threshold=1e-3)[0]
    is_n = (X.sum(dim=1) == 0).numpy()
    hits["all_n"] = [is_n[s, a:b].all() for s, a, b in zip(hits.sequence_name, hits.start, hits.end)]
    assert hits.loc[hits.sequence_name == 0, "all_n"].all() and (hits.sequence_name == 0).any()
    assert (hits.loc[hits.all_n, "score"] == 0).all()
    negative = hits.loc[hits.sequence_name == 1]
    assert len(negative) and (negative.score < 0).all()
    mixed = hits.loc[hits.sequence_name == 2]
    assert mixed.all_n.any() and (~mixed.all_n).any()

    logp = compute_motif_pvalues(bed, tmp_path / "genome.fa", motifs, names, n_peaks=4, window=60, threshold=1e-3)
    assert logp.loc["n_run", "TestTF"] == 0
    assert logp.loc["negative", "TestTF"] == pytest.approx(-math.log10(negative["p-value"].min()))
    assert logp.loc["mixed", "TestTF"] == pytest.approx(-math.log10(mixed.loc[~mixed.all_n, "p-value"].min()))
    assert logp.loc["nothing", "TestTF"] == 0
    assert logp.attrs["all_n_hits"]["TestTF"] == int(hits.all_n.sum())
    assert logp.attrs["all_n_entries"]["TestTF"] == 1  # only n_run is left without a hit


def test_fimo_p_of_zero_gets_the_motif_floor(tmp_path, monkeypatch):
    """FIMO reports p = 0 for some perfect matches; check the substitution and its record with a patched scan."""
    import scdori.pp.motif_scanning as motif_scanning

    genome = ["T"] * 400
    genome[118:124] = list("ACGTAC")
    (tmp_path / "genome.fa").write_text(">chr1\n" + "".join(genome) + "\n")
    bed = tmp_path / "peaks.bed"
    bed.write_text("chr1\t100\t140\tperfect\n")
    pwm = np.full((6, 4), 0.01)
    pwm[np.arange(6), [0, 1, 2, 3, 0, 1]] = 0.97
    _write_meme(tmp_path / "motifs.meme", pwm)
    motifs, names = load_motif_database(tmp_path / "motifs.meme", ["TestTF"])
    import functools

    real_fimo = motif_scanning.fimo

    @functools.wraps(real_fimo)  # keeps fimo's signature, from which the floor reads eps and bin_size
    def fimo_with_zero_p(*args, **kwargs):
        hits = real_fimo(*args, **kwargs)
        for h in hits:
            h.loc[h.score.idxmax(), "p-value"] = 0.0
        return hits

    monkeypatch.setattr(motif_scanning, "fimo", fimo_with_zero_p)
    logp = compute_motif_pvalues(bed, tmp_path / "genome.fa", motifs, names, n_peaks=1, window=40, threshold=1e-3)
    floor = motif_pvalue_floor(next(iter(motifs.values())))
    assert logp.loc["perfect", "TestTF"] == pytest.approx(-math.log10(floor))
    assert logp.attrs["perfect_match_hits"]["TestTF"] == 1
    assert logp.attrs["pvalue_floor"]["TestTF"] == floor
