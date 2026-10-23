"""Reproducible scalar-request experiment; no GPU or extra packages required.

Run from the repository root:
    python course/projects/project1/experiment.py --sizes 3 8 17
Use --solution to review the reference without altering student files.
Every invocation is validated outside the timed region. CSV is written only
after every case succeeds. The reported CPU time includes instrumentation.
"""

import argparse
import csv
import hashlib
import importlib
import math
from pathlib import Path
import platform
import random
import statistics
import sys
import time


def make_inputs(n, seed):
    """The same (n, seed) always gives the same data for every tile."""
    rng = random.Random(seed + 1_000_003 * n)
    return ([[rng.uniform(-1, 1) for _ in range(n)] for _ in range(n)],
            [[rng.uniform(-1, 1) for _ in range(n)] for _ in range(n)])


def validate_result(out, counts, expected, n, tile):
    """Check against an independent numerical oracle AND request oracle.

    Explicit exceptions keep checks active under ``python -O``.
    Floating-point bound follows math.isclose: abs(error) <= max(
    1e-10, 1e-9*max(abs(actual), abs(expected))).
    """
    if len(out) != n or any(len(row) != n for row in out):
        raise ValueError("result shape mismatch")
    errors = []
    for i in range(n):
        for j in range(n):
            value = out[i][j]
            if not math.isfinite(value) or not math.isclose(value, expected[i][j], rel_tol=1e-9, abs_tol=1e-10):
                raise ValueError(f"numerical mismatch at ({i}, {j})")
            errors.append(abs(value - expected[i][j]))
    blocks = (n + tile - 1) // tile
    exact = {"a_reads": n*n*blocks, "b_reads": n*n*blocks,
             "c_writes": n*n, "requested_bytes": 4*(2*n*n*blocks + n*n)}
    if counts != exact:
        raise ValueError(f"request counts mismatch: expected {exact}, got {counts}")
    return max(errors)


def run_case(kernels, n, tile, *, seed, repeats, warmups, bandwidth_gbps, peak_tflops):
    a, b = make_inputs(n, seed)
    originals = ([row[:] for row in a], [row[:] for row in b])
    expected = [[math.fsum(a[i][k]*b[k][j] for k in range(n))
                 for j in range(n)] for i in range(n)]
    samples, max_error = [], 0.0
    for run in range(warmups + repeats):
        start = time.perf_counter()
        out, counts = kernels.matmul_tiled_profile(a, b, tile)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if a != originals[0] or b != originals[1]:
            raise ValueError("kernel mutated its inputs")
        max_error = max(max_error, validate_result(out, counts, expected, n, tile))
        if run >= warmups:
            samples.append(elapsed_ms)
    naive, model = kernels.matmul_traffic(n, tile)
    expected_naive = (2*n**3 + n*n) * 4
    if naive != expected_naive or model != counts["requested_bytes"]:
        raise ValueError("traffic formula disagrees with independent request counts")
    flops = 2*n**3  # conventional GEMM count: one multiply-add = two FLOPs
    lower, bound = kernels.estimate_time(model, flops, bandwidth_gbps, peak_tflops)
    expected_lower = max(model/(bandwidth_gbps*1e9), flops/(peak_tflops*1e12))
    expected_bound = "memory" if model/(bandwidth_gbps*1e9) >= flops/(peak_tflops*1e12) else "compute"
    if not math.isclose(lower, expected_lower, rel_tol=1e-12, abs_tol=0.0) or bound != expected_bound:
        raise ValueError("roofline formula disagrees with independent model")
    return {
        "n": n, "tile": tile, **counts, "naive_model_bytes": naive,
        "model_bytes": model, "request_reduction_ratio": naive/model,
        "gemm_flops": flops, "model_ai_flop_per_byte": flops/model,
        "max_abs_error": max_error, "cpu_instrumented_median_ms": statistics.median(samples),
        "cpu_instrumented_p90_ms": sorted(samples)[math.ceil(0.9*len(samples))-1],
        "model_lower_bound_s": lower, "model_bound": bound,
        "seed": seed, "warmups": warmups, "repeats": repeats,
        "bandwidth_gbps": bandwidth_gbps, "peak_tflops": peak_tflops,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", action="store_true")
    parser.add_argument("--sizes", nargs="+", type=int, default=[3, 8, 17])
    parser.add_argument("--tiles", nargs="+", type=int, default=[1, 2, 8, 16])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--bandwidth-gbps", type=float, default=1000.0,
                        help="fictional accelerator default; this is not host CPU bandwidth")
    parser.add_argument("--peak-tflops", type=float, default=100.0,
                        help="fictional accelerator default; match dtype and compute unit in report")
    parser.add_argument("--output", type=Path, help="write a UTF-8 CSV (default: stdout)")
    args = parser.parse_args(argv)
    if min(args.sizes + args.tiles + [args.repeats]) <= 0 or args.warmups < 0:
        parser.error("sizes, tiles and repeats must be positive; warmups must be nonnegative")
    if any(not math.isfinite(v) or v <= 0 for v in (args.bandwidth_gbps, args.peak_tflops)):
        parser.error("bandwidth and peak must be finite and positive")
    kernels = importlib.import_module("project1_solution" if args.solution else "project1")
    metadata = {
        "implementation": kernels.__name__,
        "source_sha256": hashlib.sha256(Path(kernels.__file__).read_bytes()).hexdigest(),
        "memory_sha256": hashlib.sha256(Path(__file__).with_name("memory.py").read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": platform.python_version(), "platform": platform.platform(),
    }
    rows = []
    for n in args.sizes:
        for tile in args.tiles:
            row = run_case(kernels, n, tile, seed=args.seed, repeats=args.repeats,
                           warmups=args.warmups, bandwidth_gbps=args.bandwidth_gbps,
                           peak_tflops=args.peak_tflops)
            rows.append({**row, **metadata})
    stream = args.output.open("w", newline="", encoding="utf-8") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
