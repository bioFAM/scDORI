"""Execute the three original notebooks on local synthetic multiome/reference data.

This exercises real preprocessing (BEDTools, Harmony/Leiden, FIMO), training,
permutation filtering, regulon export and local-resource ORA. It validates software
integration, not gastrulation biology or remote genome/resource availability.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import anndata as ad
import nbformat
import numpy as np
import pandas as pd
import scipy.sparse as sp
from nbclient import NotebookClient

from scdori._core.config import TrainConfig


def fixture(root, count_layout="dense"):
    """Write an aligned small raw dataset, reference genome/GTF and MEME motifs."""
    rng = np.random.default_rng(41)
    data, genome, motifs = root / "data", root / "genome", root / "motifs"
    for path in (data, genome, motifs):
        path.mkdir(parents=True)
    genes = [f"Gene{i}" for i in range(55)] + ["Gata6", "Tal1", "Sox2", "Tbx5", "Hnf4a"]
    n, p = 96, 90
    state = np.arange(n) % 3
    obs = pd.DataFrame(
        {
            "sample": pd.Categorical(np.where(np.arange(n) % 2, "sample_a", "sample_b")),
            "celltype": pd.Categorical([f"lineage_{i}" for i in state]),
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    rna = rng.poisson(2 + 5 * (state[:, None] == np.arange(len(genes))[None, :] % 3)).astype(np.float32)
    atac = rng.poisson(1 + 4 * (state[:, None] == np.arange(p)[None, :] % 3)).astype(np.float32)
    if count_layout == "csc":
        rna, atac = sp.csc_matrix(rna), sp.csc_matrix(atac)
    peaks = [f"chr1:{1000 + i * 500}-{1200 + i * 500}" for i in range(p)]
    ad.AnnData(rna, obs=obs.copy(), var=pd.DataFrame(index=genes)).write_h5ad(data / "rna.h5ad")
    a = ad.AnnData(atac, obs=obs.copy(), var=pd.DataFrame({"is_promoter": np.arange(p) % 10 == 0}, index=peaks))
    a.uns["scdori_atac_count_mode"] = "fragments"
    a.write_h5ad(data / "atac.h5ad")
    length = 60000
    sequence = rng.choice(list("ACGT"), length)
    patterns = ["ACGTAC", "GATTAC", "CCGTTA", "TACGGA", "GGCATA"]
    for i in range(p):
        start = 1080 + i * 500
        sequence[start : start + 6] = list(patterns[i % 5])
    (genome / "mm10.fa").write_text(">chr1\n" + "".join(sequence) + "\n")
    (genome / "mm10.chrom.sizes").write_text(f"chr1\t{length}\n")
    (genome / "annotation.gtf").write_text(
        "".join(
            f'chr1\tfixture\tgene\t{1001 + i * 700}\t{1300 + i * 700}\t.\t+\t.\tgene_id "id{i}"; gene_name "{g}"; gene_type "protein_coding";\n'
            for i, g in enumerate(genes)
        )
    )
    meme = "MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\nBackground letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n"
    for i, (tf, pattern) in enumerate(zip(genes[-5:], patterns, strict=True)):
        pwm = np.full((6, 4), 0.01)
        pwm[np.arange(6), ["ACGT".index(x) for x in pattern]] = 0.97
        meme += f"MOTIF motif{i} {tf}_1\nletter-probability matrix: alength= 4 w= 6 nsites= 20 E= 0\n"
        meme += "\n".join(" ".join(map(str, row)) for row in pwm) + "\n\n"
    (motifs / "fixture_mouse.meme").write_text(meme)
    sets = pd.DataFrame(
        [(f"set_{i}", g) for i in range(4) for g in genes[i * 10 : i * 10 + 15]], columns=["geneset", "genesymbol"]
    )
    sets.to_csv(root / "genesets.tsv", sep="\t", index=False)
    return data, genome, motifs


def execute(repo, root, name, label):
    """Execute an unchanged notebook using the shared project config on disk."""
    notebook = nbformat.read(repo / "docs/notebooks" / f"{name}.ipynb", as_version=4)
    folder = root / label
    folder.mkdir()
    client = NotebookClient(
        notebook, timeout=600, kernel_name="scdori-validation", resources={"metadata": {"path": str(folder)}}
    )
    client.on_cell_start = lambda cell, cell_index: print(f"{label}: cell {cell_index}", flush=True)
    started = time.perf_counter()
    try:
        client.execute()
    finally:
        nbformat.write(notebook, folder / "executed.ipynb")
    return {"notebook": name, "label": label, "cells": len(notebook.cells), "seconds": time.perf_counter() - started}


def main():
    """Run local references and both legacy/frozen training downstream workflows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bedtools", type=Path)
    parser.add_argument("--count-layout", choices=["dense", "csc"], default="dense")
    parser.add_argument(
        "--tf-expression-scaling", choices=["within_topic", "mean_within_across"], default=TrainConfig.tf_expression_scaling
    )
    parser.add_argument("--tf-expression-clamp", type=float, default=TrainConfig.tf_expression_clamp)
    parser.add_argument(
        "--topic-usage-weight", type=float, default=0.0, help="Weight for each optional phase-1 topic penalty"
    )
    parser.add_argument("--permutation-seed", type=int, default=None)
    parser.add_argument(
        "--inspection-checks", action="store_true", help="Also test phase-1-only and checkpoint continuation"
    )
    args = parser.parse_args()
    repo, root = Path(__file__).resolve().parents[1], args.out.resolve()
    if args.bedtools:
        os.environ["PATH"] = str(args.bedtools.resolve().parent) + os.pathsep + os.environ["PATH"]
    if shutil.which("bedtools") is None:
        raise RuntimeError("Real BEDTools is required; install it or specify --bedtools")
    root.mkdir(parents=True, exist_ok=False)
    kernels = root / "kernels/scdori-validation"
    kernels.mkdir(parents=True)
    (kernels / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "scDORI workflow validation",
                "language": "python",
            }
        )
    )
    os.environ.update(
        JUPYTER_PATH=str(root),
        PYTHONPATH=str(repo / "src"),
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        NUMBA_NUM_THREADS="2",
    )
    fixture(root, args.count_layout)
    # Exercise the public configuration path: no injected notebook code or
    # assignments to module globals. Each kernel discovers this ancestor file.
    os.environ.pop("SCDORI_CONFIG_FILE", None)
    project = {
        "preprocessing": {
            "data_dir": "data",
            "genome_dir": "genome",
            "motif_directory": "motifs",
            "rna_adata_file_name": "rna.h5ad",
            "atac_adata_file_name": "atac.h5ad",
            "motif_database": "fixture",
            "genes_user": [],
            "tfs_user": ["Gata6", "Tal1", "Sox2", "Tbx5", "Hnf4a"],
            "num_genes": 50,
            "num_tfs": 5,
            "num_peaks": 70,
            "peaks_user": ["chr1:1000-1200"],
            "window_size": 5000,
            "leiden_resolution": 2,
            "promoter_col": "is_promoter",
            "correlation_percentile": 80,
            "n_bg_peaks_for_corr": 10,
        },
        "training": {
            "num_topics": 4,
            "dim_encoder1": 16,
            "dim_encoder2": 8,
            "batch_size_cell": 16,
            "batch_size_cell_prediction": 32,
            "max_scdori_epochs": 3,
            "epoch_warmup_1": 1,
            "max_grn_epochs": 2,
            "cells_per_topic": 10,
            "num_permutations": 20,
            "tf_expression_scaling": args.tf_expression_scaling,
            "tf_expression_clamp": args.tf_expression_clamp,
            "theta_entropy_weight": args.topic_usage_weight,
            "usage_balance_weight": args.topic_usage_weight,
        },
        "downstream": {
            "geneset_file": "genesets.tsv",
            "ora_min_targets": 2,
            "permutation_seed": args.permutation_seed,
            "exclude_topics": [3],
            "topic_activity_threshold": 0.0,
            "run_gastrulation_examples": False,
        },
    }
    config_path = root / "scdori_config.json"
    config_path.write_text(json.dumps(project, indent=2))
    results = [execute(repo, root, "preprocessing", "preprocessing")]
    processed_layouts = {}
    for modality in ("rna", "atac"):
        processed = ad.read_h5ad(root / "data/generated" / f"{modality}_processed.h5ad")
        processed_layouts[modality] = processed.X.format if sp.issparse(processed.X) else "dense"
        if args.count_layout == "csc" and processed_layouts[modality] != "csr":
            raise AssertionError(f"Expected CSR processed {modality} after CSC preprocessing")
    for mode in ("legacy", "frozen"):
        project["training"].update(
            weights_folder_scdori=f"{mode}/phase1",
            weights_folder_grn=f"{mode}/phase2",
            freeze_phase1_in_grn=mode == "frozen",
            grn_topic_selection="all" if mode == "legacy" else "live",
        )
        project["downstream"]["grn_output"] = f"{mode}/downstream"
        config_path.write_text(json.dumps(project, indent=2))
        results.append(execute(repo, root, "training", f"training-{mode}"))
        results.append(execute(repo, root, "downstream", f"downstream-{mode}"))
        (root / "progress.json").write_text(json.dumps(results, indent=2))
    if args.inspection_checks:
        import torch

        project["training"].update(
            notebook_run_phase2=False,
            weights_folder_scdori="inspection/phase1",
            weights_folder_grn="inspection/phase2",
            run_phase1_diagnostics=True,
        )
        config_path.write_text(json.dumps(project, indent=2))
        results.append(execute(repo, root, "training", "phase1-only"))
        phase1 = root / "inspection/phase1/best_scdori_best_eval.pth"
        if not phase1.is_file() or (root / "inspection/phase2").exists():
            raise AssertionError("Phase-1-only workflow must save phase 1 without entering phase 2")
        before = hashlib.sha256(phase1.read_bytes()).hexdigest()
        # Deliberately plant a dead topic to exercise pruning; this is a software fixture.
        state = torch.load(phase1, weights_only=True, map_location="cpu")
        state["mu_theta.bias"][-1] = -40
        planted = phase1.parent / "planted_dead.pth"
        torch.save(state, planted)
        project["training"].update(
            notebook_run_phase2=True,
            run_phase1_diagnostics=False,
            phase1_checkpoint=str(planted),
            grn_topic_selection="live",
        )
        project["downstream"]["grn_output"] = "inspection/downstream"
        config_path.write_text(json.dumps(project, indent=2))
        results.append(execute(repo, root, "training", "continue-from-phase1"))
        if hashlib.sha256(phase1.read_bytes()).hexdigest() != before:
            raise AssertionError("Continuing phase 2 overwrote the phase-1 checkpoint")
        record = json.loads((root / "inspection/phase2/grn_topics.json").read_text())
        if 3 not in record["excluded_topics"]:
            raise AssertionError("Planted collapsed topic was not excluded")
        if np.load(root / "inspection/phase2/topic_tf_expression.npy")[3].any():
            raise AssertionError("Excluded topic has a nonzero TF-expression multiplier")
        results.append(execute(repo, root, "downstream", "downstream-pruned"))
    (root / "completion.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "fixture": "synthetic raw data and local references",
                "raw_count_layout": args.count_layout,
                "processed_count_layouts": processed_layouts,
                "inspection_checks": args.inspection_checks,
                "example_plots": "enabled; topics, TFs and targets selected from the fitted model",
                "runs": results,
            },
            indent=2,
        )
    )
    print("Completed original notebook integration:", root, flush=True)


if __name__ == "__main__":
    main()
