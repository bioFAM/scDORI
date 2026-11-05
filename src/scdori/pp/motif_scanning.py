import logging
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
