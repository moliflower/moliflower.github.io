"""Reference solution for Project 1. Try the project yourself first."""

from __future__ import annotations

import math
from memory import MatrixMemory


def vector_add(a, b):
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    return [float(x) + float(y) for x, y in zip(a, b)]


def sum_reduce(x):
    values = [float(v) for v in x]
    if not values:
        return 0.0
    while len(values) > 1:
        nxt = []
        for i in range(0, len(values) - 1, 2):
            nxt.append(values[i] + values[i + 1])
        if len(values) % 2:
            nxt.append(values[-1])
        values = nxt
    return values[0]


def softmax(x):
    if not x:
        return []
    m = max(float(v) for v in x)
    exps = [math.exp(float(v) - m) for v in x]
    s = sum_reduce(exps)
    return [v / s for v in exps]


def rmsnorm(x, weight, eps=1e-6):
    if len(x) != len(weight):
        raise ValueError("x and weight must have the same length")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if not x:
        return []
    mean_sq = sum_reduce([float(v) * float(v) for v in x]) / len(x)
    scale = 1.0 / math.sqrt(mean_sq + eps)
    return [float(v) * scale * float(w) for v, w in zip(x, weight)]


def _check_matrix(a, name="a"):
    if not isinstance(a, list) or any(not isinstance(row, list) for row in a):
        raise ValueError(f"{name} must be a list of lists")
    if a and len({len(row) for row in a}) != 1:
        raise ValueError(f"{name} must be rectangular")
    if not a or not a[0]:
        raise ValueError(f"{name} must have at least one row and one column")


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def matmul_naive(a, b):
    _check_matrix(a, "a")
    _check_matrix(b, "b")
    m, k, n = len(a), len(b), len(b[0])
    if a and len(a[0]) != k:
        raise ValueError("inner dimensions do not match")
    c = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            total = 0.0
            for kk in range(k):
                total += float(a[i][kk]) * float(b[kk][j])
            c[i][j] = total
    return c


def matmul_tiled(a, b, tile=16):
    return matmul_tiled_profile(a, b, tile)[0]


def matmul_tiled_profile(a, b, tile=16):
    _check_matrix(a, "a")
    _check_matrix(b, "b")
    _positive_int(tile, "tile")
    m, k, n = len(a), len(b), len(b[0])
    if a and len(a[0]) != k:
        raise ValueError("inner dimensions do not match")
    memory = MatrixMemory(a, b)
    for i0 in range(0, m, tile):
        for j0 in range(0, n, tile):
            rows, cols = min(tile, m-i0), min(tile, n-j0)
            accum = [[0.0] * cols for _ in range(rows)]
            for k0 in range(0, k, tile):
                depth = min(tile, k-k0)
                at = [[memory.load_a(i0+i, k0+p) for p in range(depth)] for i in range(rows)]
                bt = [[memory.load_b(k0+p, j0+j) for j in range(cols)] for p in range(depth)]
                for i in range(rows):
                    for j in range(cols):
                        for p in range(depth):
                            accum[i][j] += at[i][p] * bt[p][j]
            for i in range(rows):
                for j in range(cols):
                    memory.store_c(i0+i, j0+j, accum[i][j])
    return memory.output, memory.stats()


def matmul_traffic(n, tile=16, dtype_bytes=4):
    if type(n) is not int or n < 0:
        raise ValueError("n must be a nonnegative integer")
    _positive_int(tile, "tile")
    _positive_int(dtype_bytes, "dtype_bytes")
    naive = (2 * n**3 + n**2) * dtype_bytes
    q = (n + tile - 1) // tile
    tiled = (2 * n**2 * q + n**2) * dtype_bytes
    return naive, tiled


def estimate_time(bytes_moved, flops, bandwidth_gbps=1000.0, peak_tflops=100.0):
    if any(not math.isfinite(v) or v < 0 for v in (bytes_moved, flops)):
        raise ValueError("bytes_moved and flops must be finite and nonnegative")
    if any(not math.isfinite(v) or v <= 0 for v in (bandwidth_gbps, peak_tflops)):
        raise ValueError("bandwidth and peak must be finite and positive")
    memory_s = bytes_moved / (bandwidth_gbps * 1e9)
    compute_s = flops / (peak_tflops * 1e12)
    if memory_s >= compute_s:
        return memory_s, "memory"
    return compute_s, "compute"
