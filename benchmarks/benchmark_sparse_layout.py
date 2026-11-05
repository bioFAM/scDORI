"""Measure CSC/CSR row reads and host memory in separate Linux processes."""

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy
import scipy.sparse as sp

from scdori import prepare_count_matrix


def rss_bytes():
    """Read current resident memory from Linux procfs."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS unavailable")


def worker(args):
    """Run one layout with deterministic input and batch order."""
    rng = np.random.default_rng(20260922)
    if args.input:
        data = ad.read_h5ad(args.input)
        if not sp.issparse(data.X):
            raise ValueError("Benchmark requires sparse counts")
        source_layout = data.X.format
        data.X = data.X.tocsc()
    else:
        # Build sparse counts directly; never allocate the dense shape.
        rows, cols, per_col = 20000, 10000, 400
        indices = rng.integers(0, rows, size=cols * per_col, dtype=np.int32)
        values = rng.integers(1, 8, size=len(indices)).astype(np.float32)
        counts = sp.csc_matrix((values, indices, np.arange(cols + 1, dtype=np.int32) * per_col), shape=(rows, cols))
        counts.sum_duplicates()
        data = ad.AnnData(counts)
        del counts, values, indices
        source_layout = "synthetic CSC"
    original = data.X  # Conversion already holds this buffer; keep for equality check.
    baseline = rss_bytes()
    samples = [baseline]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.002):
            samples.append(rss_bytes())

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    start = time.perf_counter()
    data = prepare_count_matrix(data, use_csr=args.worker == "csr")
    elapsed = time.perf_counter() - start
    samples.append(rss_bytes())
    stop.set()
    monitor.join()
    lifetime_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    # Complete equality checked outside the conversion/row-read measurements.
    equal = (original != data.X).nnz == 0
    del original
    if not equal:
        raise AssertionError("Sparse conversion changed counts")
    timings, digest = [], hashlib.sha256()
    batch_rng = np.random.default_rng(3141)
    for iteration in range(args.batches + 2):
        ids = batch_rng.integers(0, data.n_obs, size=args.batch_size)
        start = time.perf_counter()
        batch = data.X[ids]
        elapsed_batch = time.perf_counter() - start
        # Dense conversion/hash deliberately excluded from sparse-read timing.
        digest.update(batch.toarray().tobytes())
        if iteration >= 2:
            timings.append(elapsed_batch)
    return {
        "layout": data.X.format,
        "source_layout": source_layout,
        "shape": list(data.shape),
        "nnz": data.X.nnz,
        "dtype": str(data.X.dtype),
        "index_dtype": str(data.X.indices.dtype),
        "values_identical": equal,
        "conversion_seconds": elapsed,
        "rss_before_conversion_bytes": baseline,
        "conversion_sampled_peak_rss_bytes": max(samples),
        "conversion_sampled_rss_increase_bytes": max(samples) - baseline,
        "process_peak_rss_through_conversion_bytes": lifetime_peak,
        "sparse_buffer_bytes": data.X.data.nbytes + data.X.indices.nbytes + data.X.indptr.nbytes,
        "row_read_seconds": timings,
        "row_read_median_seconds": float(np.median(timings)),
        "batch_values_sha256": digest.hexdigest(),
    }


def main():
    """Run isolated workers and write an auditable JSON result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("sparse-layout.json"))
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--worker", choices=["csc", "csr"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.batches < 1 or args.batch_size < 1:
        parser.error("batches and batch-size must be positive")
    if args.worker:
        print(json.dumps(worker(args)))
        return
    results = []
    for layout in ("csc", "csr"):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            layout,
            "--batches",
            str(args.batches),
            "--batch-size",
            str(args.batch_size),
        ]
        if args.input:
            command += ["--input", str(args.input.resolve())]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        results.append(json.loads(result.stdout))
    if results[0]["batch_values_sha256"] != results[1]["batch_values_sha256"]:
        raise AssertionError("CSC and CSR batches differ")
    report = {
        "scope": "CPU sparse row reads; not full training; two warm-up reads excluded",
        "input": str(args.input.resolve()) if args.input else "synthetic",
        "python": platform.python_version(),
        "scipy": scipy.__version__,
        "anndata": ad.__version__,
        "platform": platform.platform(),
        "batches": args.batches,
        "batch_size": args.batch_size,
        "memory_note": "Conversion RSS sampled every 2 ms may miss brief peaks; lifetime high-water includes loading. "
        "Holding the original CSC buffers matches conversion peak lifetime, but not post-conversion steady state.",
        "results": results,
        "row_read_speedup": results[0]["row_read_median_seconds"] / results[1]["row_read_median_seconds"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
