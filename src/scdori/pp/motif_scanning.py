import inspect
import logging
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tangermeme.io import extract_loci, read_meme
from tangermeme.tools.fimo import fimo
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_motif_scores(
    bed_file: Path,
    fasta_file: Path,
    pwms_sub: dict,
    key_to_tf: dict,
    n_peaks: int,
    window: int = 500,
    threshold: float = 1e-3,
) -> pd.DataFrame:
    """
    Compute a motif score matrix (n_peaks x n_TFs) by scanning peak sequences with FIMO (from tangermeme).

    Steps
    -----
    1. Read peak coordinates from `bed_file`.
    2. Extract sequences from `fasta_file`.
    3. Run FIMO on these sequences using the PWMs in `pwms_sub`.
    4. Aggregate scores per (peak, motif) combination; rename motifs by their TF name.
    5. Combine into a DataFrame of shape (n_peaks, n_TFs) and scale values to [0, 1].

    Parameters
    ----------
    bed_file : pathlib.Path
        The BED file containing peak coordinates. Column 4 must be the peak name.
    fasta_file : pathlib.Path
        The path to the reference genome FASTA file.
    pwms_sub : dict
        A dictionary of position weight matrices (PWMs), e.g. from a .meme file,
        keyed by some motif identifier.
    key_to_tf : dict
        A mapping from motif identifiers to transcription factor (TF) names.
    n_peaks : int
        The total number of peaks expected (should match the number of lines in `bed_file`).
    window : int, optional
        Window size around the peak to extract sequences for motif scanning. Default is 500.
    threshold : float, optional
        Minimum p-value (or e-value) threshold for motif hits in FIMO. Default is 1e-3.

    Returns
    -------
    pd.DataFrame
        A DataFrame of shape (n_peaks, n_TFs). Rows are indexed by peak names from
        `bed_file`, columns are TF names. Values are scaled motif scores in [0, 1].

    Notes
    -----
    - If a TF does not appear in any motif hits, its column will be filled with zeros.
    - For each motif name in `pwms_sub`, we rename it to the corresponding TF from `key_to_tf`.
    - The final DataFrame is MinMax-scaled by each column (TF).
    """
    logger.info(f"Computing motif scores for {bed_file} (n_peaks={n_peaks}) with window={window}")
    loci = pd.read_csv(bed_file, sep="\t", header=None)
    loci.columns = ["chr", "start", "end", "peak_name"]

    # Extract sequences
    X = extract_loci(loci, str(fasta_file), in_window=window).float()

    # Run FIMO
    hits_list = fimo(pwms_sub, X, threshold=threshold)

    peak_motif_scores = []

    for k in tqdm(range(len(hits_list))):
        # Group by motif_name and sequence_name, taking the max score per group
        motif_df = (
            hits_list[k][["motif_name", "sequence_name", "score"]]
            .groupby(["motif_name", "sequence_name"])
            .max()
            .reset_index()
        )

        if motif_df.shape[0] > 0:
            all_sequences = pd.DataFrame({"sequence_name": range(n_peaks)})
            motif_name = motif_df.motif_name.values[0]
            tf_name = key_to_tf[motif_name]

            # Merge with a full list of sequences, fill missing scores with 0
            complete_df = all_sequences.merge(motif_df, on="sequence_name", how="left")
            complete_df["score"] = complete_df["score"].fillna(0)

            # Keep only the "score" column, rename it to the TF name
            complete_df = complete_df[["sequence_name", "score"]].set_index("sequence_name")
            complete_df.columns = [tf_name]

            peak_motif_scores.append(complete_df)

    # Concatenate all motif DataFrames
    if len(peak_motif_scores) > 0:
        peak_motif_scores = pd.concat(peak_motif_scores, axis=1)
    else:
        logger.warning("No motif scores were computed. Returning an empty DataFrame.")
        peak_motif_scores = pd.DataFrame(index=range(n_peaks))

    # Ensure all TFs appear in columns, filling missing ones with 0
    remaining_tfs = set(key_to_tf.values()) - set(peak_motif_scores.columns)
    for tf in remaining_tfs:
        peak_motif_scores[tf] = 0

    # Reorder columns
    final_tf_list = sorted(set(key_to_tf.values()))
    peak_motif_scores = peak_motif_scores[final_tf_list]

    # Scale motif scores to [0, 1]
    scaler = MinMaxScaler()
    motif_scores = scaler.fit_transform(peak_motif_scores.values)

    bed_file_peak = pd.read_csv(bed_file, sep="\t", header=None)
    df_motif = pd.DataFrame(motif_scores, columns=peak_motif_scores.columns, index=bed_file_peak[3].values)

    logger.info(f"Finished computing motif scores: {df_motif.shape}")
    return df_motif


def simulate_pwm_scoring(pwm_matrix, bed_file, fasta_file, window):
    """
    Simulate PWM scoring for a given motif using random data.

    Parameters
    ----------
    pwm_matrix : np.ndarray
        Position Weight Matrix for the motif (rows = positions, columns = nucleotides).
    bed_file : pathlib.Path or str
        The BED file containing peak locations.
    fasta_file : pathlib.Path or str
        Path to the genome FASTA file.
    window : int
        The window size around peaks to consider for motif scanning.

    Returns
    -------
    pd.DataFrame
        A simulated DataFrame of motif scores with columns
        ['motif_name', 'sequence_name', 'score'].

    Notes
    -----
    - In a real scenario, this would scan each sequence (peak ± window) with the PWM and return actual FIMO-like hits.
    - Here, random scores are generated for testing.
    """
    n_peaks = 60000  # Example number of peaks
    scores = np.random.rand(n_peaks)  # Randomly generated scores
    sequence_names = range(n_peaks)

    df = pd.DataFrame({"motif_name": ["motif1"] * n_peaks, "sequence_name": sequence_names, "score": scores})
    return df


def motif_pvalue_floor(pwm) -> float:
    """Best possible p-value of a motif in tangermeme FIMO's score table: the table's smallest non-zero entry.

    tangermeme ``fimo`` turns scores into p-values with a table of its rounded score distribution
    (``log2(pwm + eps) - log2(0.25)``, rounded in steps of ``bin_size``; both are ``fimo``'s own defaults, read
    from its signature here). A perfect match can fall beyond the table's last finite entry, and ``fimo`` then
    reports p = 0. The last finite entry is the probability, under a uniform background, of a sequence reaching
    the highest rounded score: the product over positions of (number of bases tied at that position's highest
    rounded score) / 4. This is the floor the benchmarked implementation used for perfect matches (it read the same
    table entry); it is never below the motif's best possible p, so a cutoff stricter than that p is not passed.
    """
    defaults = inspect.signature(fimo).parameters
    eps, bin_size = defaults["eps"].default, defaults["bin_size"].default
    weights = np.asarray(pwm, dtype=np.float64)  # alphabet x positions, as tangermeme reads MEME files
    rounded = np.round((np.log2(weights + eps) - math.log2(0.25)) / bin_size)
    ties = (rounded == rounded.max(axis=0, keepdims=True)).sum(axis=0)
    return float(np.prod(ties / 4.0))


def compute_motif_pvalues(
    bed_file: Path,
    fasta_file: Path,
    pwms_sub: dict,
    key_to_tf: dict,
    n_peaks: int,
    window: int = 500,
    threshold: float = 1e-3,
    chunk_size: int = 20,
) -> pd.DataFrame:
    """Motif evidence for the motif-weighted in-silico ChIP-seq: -log10 p of the best FIMO hit per (peak, TF).

    Peaks are scanned with tangermeme ``fimo`` at p < ``threshold``, as in :func:`compute_motif_scores`, in
    groups of ``chunk_size`` motifs to bound memory. For each (peak, TF) the smallest ``p-value`` that ``fimo``
    reports is kept and returned as -log10 p; 0 means no hit. Hits with a negative score are hits.

    ``fimo`` reports p = 0 for some perfect matches; those get the motif's best possible p from FIMO's score
    table (:func:`motif_pvalue_floor`), as in the benchmarked implementation. The number of such hits and the floor
    used are stored per TF in ``result.attrs["perfect_match_hits"]`` and ``result.attrs["pvalue_floor"]``.

    ``fimo`` skips N bases when it scores a window, so a window made only of N scores exactly 0, and motifs whose
    p-value cutoff lies below a score of 0 (long, low-information motifs) report such windows as hits. Hits whose
    window consists only of N are dropped; all other hits, including real hits with a score of 0 or below, are
    kept. This departs from the benchmarked implementation, which treated a (peak, TF) whose best score was 0 as no
    hit and so also dropped real negative-score hits in those peaks: on one benchmark dataset the two rules
    differ in 301 (peak, TF) entries (all hits here, none in the benchmarked implementation), in 12 peaks next to
    N runs. Per TF, ``result.attrs["all_n_hits"]`` counts
    the dropped hits and ``result.attrs["all_n_entries"]`` the (peak, TF) pairs left without any hit by it.
    Windows that are only partly N are scored by ``fimo`` without their N positions and are kept.

    Returns
    -------
    pd.DataFrame
        Shape (n_peaks, n_TFs), rows named by the peak names in ``bed_file`` and columns by TF (sorted).
    """
    logger.info(f"Computing motif p-values for {bed_file} (n_peaks={n_peaks}) with window={window}")
    loci = pd.read_csv(bed_file, sep="\t", header=None).iloc[:, :4]
    loci.columns = ["chr", "start", "end", "peak_name"]
    if len(loci) != n_peaks:
        raise ValueError(f"{bed_file} has {len(loci)} peaks, expected {n_peaks}")
    X = extract_loci(loci, str(fasta_file), in_window=window).float()
    # N positions are all-zero columns of the one-hot sequence; cumulative counts locate all-N windows.
    is_n = (X.sum(dim=1) == 0).numpy()
    peaks_with_n = np.flatnonzero(is_n.any(axis=1))
    n_row = np.full(n_peaks, -1)
    n_row[peaks_with_n] = np.arange(len(peaks_with_n))
    n_cumulative = np.concatenate(
        [np.zeros((len(peaks_with_n), 1), dtype=np.int64), np.cumsum(is_n[peaks_with_n], axis=1)], axis=1
    )
    tfs = sorted(set(key_to_tf.values()))
    column = {tf: i for i, tf in enumerate(tfs)}
    logp = np.zeros((n_peaks, len(tfs)))
    floors, floored, all_n_hits, all_n_entries = {}, {}, {}, {}
    keys = list(pwms_sub)
    for start in tqdm(range(0, len(keys), chunk_size), desc="Scanning motifs"):
        subset = {key: pwms_sub[key] for key in keys[start : start + chunk_size]}
        for hits in fimo(subset, X, threshold=threshold):
            if not len(hits):
                continue
            key = hits.motif_name.values[0]
            tf = key_to_tf[key]
            row = n_row[hits.sequence_name.to_numpy()]
            all_n = np.zeros(len(hits), dtype=bool)
            if (row >= 0).any():
                where = np.flatnonzero(row >= 0)
                starts = hits.start.to_numpy()[where].astype(np.int64)
                ends = hits.end.to_numpy()[where].astype(np.int64)
                n_in_window = n_cumulative[row[where], ends] - n_cumulative[row[where], starts]
                all_n[where] = n_in_window == ends - starts
            had_hit = set(hits.sequence_name[all_n])
            hits = hits[~all_n]  # windows of N only (see above): not hits
            all_n_hits[tf] = int(all_n.sum())
            all_n_entries[tf] = len(had_hit - set(hits.sequence_name))
            if not len(hits):
                continue
            best = hits.groupby("sequence_name")["p-value"].min().reset_index()
            p = best["p-value"].to_numpy(dtype=np.float64)
            floor = motif_pvalue_floor(pwms_sub[key])
            floors[tf], floored[tf] = floor, int((p == 0).sum())
            logp[best.sequence_name.to_numpy(), column[tf]] = -np.log10(np.where(p > 0, p, floor))
    result = pd.DataFrame(logp, index=loci["peak_name"].astype(str).values, columns=tfs)
    result.attrs["pvalue_floor"] = pd.Series(floors, dtype=float).reindex(tfs)
    result.attrs["perfect_match_hits"] = pd.Series(floored, dtype="Int64").reindex(tfs).fillna(0).astype(int)
    result.attrs["all_n_hits"] = pd.Series(all_n_hits, dtype="Int64").reindex(tfs).fillna(0).astype(int)
    result.attrs["all_n_entries"] = pd.Series(all_n_entries, dtype="Int64").reindex(tfs).fillna(0).astype(int)
    logger.info(
        "Finished computing motif p-values: %s; %d hits with p = 0 received the motif's best possible p; "
        "%d hits in windows of only N bases were dropped, leaving %d (peak, TF) pairs without a hit",
        result.shape,
        int(result.attrs["perfect_match_hits"].sum()),
        int(result.attrs["all_n_hits"].sum()),
        int(result.attrs["all_n_entries"].sum()),
    )
    return result


def load_motif_database(motif_path: Path, final_tfs: list[str]) -> tuple[dict, dict]:
    """
    Read a .meme motif file and select only the motifs corresponding to TFs of interest.

    Parameters
    ----------
    motif_path : pathlib.Path
        The path to the .meme file.
    final_tfs : list of str
        A list of TF names for which we want motifs.

    Returns
    -------
    (pwms_sub, key_to_tf) : tuple of (dict, dict)
        pwms_sub : dict
            A dictionary of PWM matrices (motif_name -> PWM) only for the desired TFs.
        key_to_tf : dict
            A mapping from motif_name to TF name (one motif per TF).

    Notes
    -----
    - This function expects that each motif key in the meme file can be parsed to extract a TF name. We assume a format like "MOTIF something Tbx5_...".
    """
    logger.info(f"Reading motif file: {motif_path}")
    pwms = read_meme(motif_path)

    selected_keys = []
    selected_tfs = []
    for key in pwms.keys():
        # Example parse: "MOTIF  something Tbx5_..."
        tf_name = key.split(" ")[1].split("_")[0].strip("()").strip()
        if tf_name in final_tfs:
            selected_keys.append(key)
            selected_tfs.append(tf_name)

    df_map = pd.DataFrame({"key": selected_keys, "TF": selected_tfs}).drop_duplicates("TF")
    pwms_sub = {row.key: pwms[row.key] for _, row in df_map.iterrows()}
    key_to_tf = dict(zip(df_map["key"], df_map["TF"], strict=False))

    logger.info(f"Subselected {len(pwms_sub)} motifs for {len(final_tfs)} TFs.")
    return pwms_sub, key_to_tf


def run_bedtools_intersect(a_bed: Path, b_bed: Path, out_bed: Path) -> None:
    """Run a 'bedtools intersect' command to keep intervals in file A that overlap intervals in file B.

    Parameters
    ----------
    a_bed : pathlib.Path
        Path to the BED file "A".
    b_bed : pathlib.Path
        Path to the BED file "B".
    out_bed : pathlib.Path
        Output path for the intersected results.

    Returns
    -------
    None
        Writes a new BED file to `out_bed` containing only intervals from A
        that intersect with B at least once (option -u -wa).
    """
    cmd = ["bedtools", "intersect", "-u", "-wa", "-a", str(a_bed), "-b", str(b_bed)]
    logger.info(f"Running: {cmd}")
    with open(out_bed, "w") as output:
        subprocess.run(cmd, stdout=output, check=True)
