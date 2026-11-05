"""Compare cached and full frozen GRN training on aligned, processed real data.

Run with PYTHONPATH=src. TF columns are supplied explicitly in binding-matrix
order. Subsampling is deterministic and recorded; no input files are modified.
"""

import argparse
import copy
import json
import time
from contextlib import nullcontext
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from scdori._core.config import TrainConfig
from scdori._core.dataloader import create_minibatch
from scdori._core.frozen_grn import (
    _regularization,
    build_cell_cache,
    build_grn_factors,
    estimate_frozen_cache_bytes,
    forward_from_topics,
    freeze_phase1,
)
from scdori._core.models import initialize_scdori_parameters, scDoRI
from scdori._core.train_grn import get_tf_expression
from scdori._core.train_scdori import train_scdori_phases
from scdori._core.utils import log_nb_positive, set_seed


def main():
    """Run a reproducible benchmark and write timings and numerical comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tf-indices", type=Path, required=True, help="Numpy array in binding-matrix column order")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3141)
    parser.add_argument("--cells", type=int, default=256)
    parser.add_argument("--genes", type=int, default=256)
    parser.add_argument("--peaks", type=int, default=2048)
    parser.add_argument("--tfs", type=int, default=32)
    parser.add_argument("--topics", type=int, default=8)
    parser.add_argument("--phase1-epochs", type=int, default=5)
    parser.add_argument("--grn-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute equivalence tolerance")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative equivalence tolerance")
    parser.add_argument(
        "--profile", action="store_true", help="Profile extra steps after timing; writes local Chrome traces"
    )
    parser.add_argument(
        "--atac-count-mode",
        choices=["insertions", "fragments"],
        help="Override metadata only when input units are known",
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; check the PyTorch build and driver")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    rna_full = ad.read_h5ad(args.data / "rna_metacell.h5ad", backed="r")
    atac_full = ad.read_h5ad(args.data / "atac_metacell.h5ad", backed="r")
    if not rna_full.obs_names.equals(atac_full.obs_names):
        raise ValueError("Unaligned RNA/ATAC cell IDs")
    tf_ids = np.load(args.tf_indices)
    act_full = np.load(args.data / "insilico_chipseq_embeddings.npy", mmap_mode="r")
    rep_full = np.load(args.data / "insilico_chipseq_embeddings_repressor.npy", mmap_mode="r")
    if len(tf_ids) != act_full.shape[1]:
        raise ValueError("TF indices must match all binding matrix columns")
    selected_tfs = np.sort(rng.choice(len(tf_ids), args.tfs, replace=False))
    other_genes = np.setdiff1d(np.arange(rna_full.n_vars), tf_ids)
    genes = np.sort(
        np.concatenate((rng.choice(other_genes, args.genes - args.tfs, replace=False), tf_ids[selected_tfs]))
    )
    peaks = np.sort(rng.choice(atac_full.n_vars, args.peaks, replace=False))
    cell_ids = np.sort(rng.choice(rna_full.n_obs, args.cells, replace=False))
    rna = rna_full[cell_ids, :].to_memory()[:, genes].copy()
    atac = atac_full[cell_ids, :].to_memory()[:, peaks].copy()
    if args.atac_count_mode:
        atac.uns["scdori_atac_count_mode"] = args.atac_count_mode
    rna_full.file.close()
    atac_full.file.close()
    tfs = np.searchsorted(genes, tf_ids[selected_tfs])
    num_cells = rna.obs["num_cells"].to_numpy(dtype=np.float32).reshape(-1, 1)
    # Local metacell identifiers end in the original sample label.
    samples = np.array([name.rsplit("|", 1)[-1] for name in rna.obs_names])
    _, codes = np.unique(samples, return_inverse=True)
    batches = np.eye(codes.max() + 1, dtype=np.float32)[codes]
    distance = np.load(args.data / "gene_peak_distance_test.npy", mmap_mode="r")[np.ix_(genes, peaks)]
    device = torch.device(args.device)

    def tensor(x):
        if sp.issparse(x):
            x = x.toarray()
        return torch.as_tensor(np.array(x), device=device, dtype=torch.float32)

    model = scDoRI(device, len(genes), len(peaks), args.tfs, args.topics, batches.shape[1], 64, 32).to(device)
    initialize_scdori_parameters(
        model,
        tensor(distance),
        tensor(distance > 0),
        tensor(act_full[np.ix_(peaks, selected_tfs)]),
        tensor(rep_full[np.ix_(peaks, selected_tfs)]),
    )
    split = rng.permutation(args.cells)
    train_ids, eval_ids = split[: int(args.cells * 0.8)], split[int(args.cells * 0.8) :]

    def loader(ids):
        return DataLoader(TensorDataset(torch.from_numpy(ids)), batch_size=args.batch_size)

    train, evaluation = loader(train_ids), loader(eval_ids)
    config = TrainConfig(
        max_scdori_epochs=args.phase1_epochs,
        epoch_warmup_1=1,
        cells_per_topic=20,
        batch_size_cell_prediction=args.batch_size,
        weights_folder_scdori=str(args.out / "phase1"),
    )

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    sync()
    phase1_start = time.perf_counter()
    model = train_scdori_phases(model, device, train, evaluation, rna, atac, num_cells, tfs, batches, config)
    sync()
    phase1_seconds = time.perf_counter() - phase1_start
    freeze_phase1(model)
    start = time.perf_counter()
    topic_tf = get_tf_expression("True", model, device, train, rna, atac, num_cells, tfs, batches, config)
    sync()
    topic_tf_seconds = time.perf_counter() - start
    start = time.perf_counter()
    factors = build_grn_factors(model)
    sync()
    factor_seconds = time.perf_counter() - start
    start = time.perf_counter()
    cache = build_cell_cache(model, device, rna, atac, num_cells, tfs, batches, args.batch_size)
    sync()
    cell_seconds = time.perf_counter() - start
    preparation = factor_seconds + cell_seconds
    initial = copy.deepcopy(model.state_dict())
    results, predictions, weights = {}, {}, {}

    def step(ids, cached, optimizer, profiling=False):
        scope = torch.profiler.record_function if profiling else lambda _: nullcontext()
        if cached:
            with scope("scdori/load_and_transfer"):
                x = tensor(rna.X[ids.numpy()])
                batch = tensor(batches[ids.numpy()])
            with scope("scdori/forward"):
                mu = forward_from_topics(
                    model, cache.theta[ids].to(device), cache.library[ids].to(device), batch, topic_tf, factors
                )
        else:
            with scope("scdori/load_and_transfer"):
                inp, tf, lib, nc, batch = create_minibatch(device, ids, rna, atac, num_cells, tfs, batches)
            x = inp[:, : model.num_genes]
            with scope("scdori/forward"):
                mu = model(x, inp[:, model.num_genes :], tf, topic_tf, lib[:, :1], lib[:, 1:], nc, batch, phase="grn")[
                    "mu_nb_rna_grn"
                ]
        # All other reconstruction terms and penalties are constant in this mode.
        loss = -config.weight_rna_from_grn * log_nb_positive(x, mu, F.softplus(model.rna_alpha_nb)).sum(1).mean()
        loss = loss + _regularization(model, config, grn=True)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite benchmark loss")
        optimizer.zero_grad(set_to_none=True)
        with scope("scdori/backward_and_step"):
            loss.backward()
            optimizer.step()

    for cached in (False, True):
        model.load_state_dict(initial)
        model.eval()
        model.rna_grn_batch_norm.train()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate_grn)
        epoch_seconds = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.grn_epochs):
            sync()
            start = time.perf_counter()
            for (ids,) in train:
                step(ids, cached, optimizer)
            sync()
            epoch_seconds.append(time.perf_counter() - start)
        model.eval()
        heldout = []
        with torch.no_grad():
            for (ids,) in evaluation:
                heldout.append(
                    forward_from_topics(
                        model,
                        cache.theta[ids].to(device),
                        cache.library[ids].to(device),
                        tensor(batches[ids.numpy()]),
                        topic_tf,
                        factors,
                    ).cpu()
                )
        prediction = torch.cat(heldout)
        label = "cached" if cached else "full"
        predictions[label] = prediction
        weights[label] = torch.cat(
            [
                model.tf_gene_topic_activator_grn.detach().flatten().cpu(),
                model.tf_gene_topic_repressor_grn.detach().flatten().cpu(),
            ]
        )
        results[label] = {
            "epoch_seconds": epoch_seconds,
            "median_epoch_seconds": float(np.median(epoch_seconds)),
            "heldout_nb_loss": -log_nb_positive(
                tensor(rna.X[eval_ids]), prediction.to(device), F.softplus(model.rna_alpha_nb)
            )
            .sum(1)
            .mean()
            .item(),
            "cuda_peak_bytes_including_shared_cache": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        }
        if args.profile:
            model.rna_grn_batch_norm.train()
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
                for number, (ids,) in enumerate(train):
                    if number == 3:
                        break
                    step(ids, cached, optimizer, profiling=True)
                sync()
            prof.export_chrome_trace(str(args.out / f"{label}-trace.json"))
            (args.out / f"{label}-profile.txt").write_text(
                prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30)
            )
    report = {
        "data": str(args.data.resolve()),
        "seed": args.seed,
        "device": str(device),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "threads": args.threads,
        "shape": {
            "cells": args.cells,
            "genes": args.genes,
            "peaks": args.peaks,
            "tfs": args.tfs,
            "topics": args.topics,
        },
        "phase1_epochs": args.phase1_epochs,
        "phase1_seconds": phase1_seconds,
        "topic_tf_seconds": topic_tf_seconds,
        "factor_build_seconds": factor_seconds,
        "cell_cache_build_seconds": cell_seconds,
        "cache_storage_estimate": estimate_frozen_cache_bytes(model, args.cells),
        "atac_count_mode": atac.uns.get("scdori_atac_count_mode", "insertions"),
        "grn_epochs": args.grn_epochs,
        "cache_build_seconds": preparation,
        "cell_cache_bytes": sum(t.numel() * t.element_size() for t in (cache.theta, cache.library, cache.auxiliary)),
        "factor_cache_bytes": sum(t.numel() * t.element_size() for t in factors),
        "results": results,
        "steady_state_speedup": results["full"]["median_epoch_seconds"] / results["cached"]["median_epoch_seconds"],
        "prediction_max_abs_difference": (predictions["full"] - predictions["cached"]).abs().max().item(),
        "grn_weight_max_abs_difference": (weights["full"] - weights["cached"]).abs().max().item(),
        "grn_weight_correlation": float(np.corrcoef(weights["full"].numpy(), weights["cached"].numpy())[0, 1]),
        "limitation": "Short subset experiment from a newly trained phase 1 model; not full-data biological validation.",
    }
    report["phase2_seconds_including_preparation"] = {
        "full": topic_tf_seconds + sum(results["full"]["epoch_seconds"]),
        "cached": topic_tf_seconds + preparation + sum(results["cached"]["epoch_seconds"]),
    }
    report["equivalence_check"] = {
        "atol": args.atol,
        "rtol": args.rtol,
        "predictions_pass": bool(
            torch.allclose(predictions["full"], predictions["cached"], atol=args.atol, rtol=args.rtol)
        ),
        "grn_weights_pass": bool(torch.allclose(weights["full"], weights["cached"], atol=args.atol, rtol=args.rtol)),
    }
    # This deliberately excludes held-out evaluation and checkpoint I/O in both paths.
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez(
        args.out / "selection.npz",
        cells=cell_ids,
        genes=genes,
        peaks=peaks,
        tf_binding_columns=selected_tfs,
        train=train_ids,
        evaluation=eval_ids,
    )
    print(json.dumps(report, indent=2))
    if not (report["equivalence_check"]["predictions_pass"] and report["equivalence_check"]["grn_weights_pass"]):
        raise AssertionError(
            "Full/cached equivalence failed; see report.json and tolerances before accepting the speedup"
        )


if __name__ == "__main__":
    main()
