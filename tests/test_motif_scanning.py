import numpy as np

from scdori import compute_motif_scores, load_motif_database


def test_real_motif_scan_from_meme_and_fasta(tmp_path):
    """Exercise motif loading, FASTA extraction and real FIMO without external data."""
    genome = list("A" * 1000)
    genome[118:124] = "ACGTAC"
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\n" + "".join(genome) + "\n")
    bed = tmp_path / "peaks.bed"
    bed.write_text("chr1\t100\t140\thas_motif\nchr1\t300\t340\tno_motif\n")
    meme = tmp_path / "motifs.meme"
    pwm = np.full((6, 4), 0.01)
    pwm[np.arange(6), [0, 1, 2, 3, 0, 1]] = 0.97
    meme.write_text(
        "MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"
        "Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n"
        "MOTIF motif1 TestTF_1\nletter-probability matrix: alength= 4 w= 6 nsites= 20 E= 0\n"
        + "\n".join(" ".join(map(str, row)) for row in pwm)
        + "\n\n"
    )
    motifs, names = load_motif_database(meme, ["TestTF"])
    assert len(motifs) == 1
    scores = compute_motif_scores(bed, fasta, motifs, names, n_peaks=2, window=40, threshold=1e-3)
    assert list(scores.index) == ["has_motif", "no_motif"]
    assert list(scores.columns) == ["TestTF"]
    np.testing.assert_allclose(scores["TestTF"].to_numpy(), [1, 0])
