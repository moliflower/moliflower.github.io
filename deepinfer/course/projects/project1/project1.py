"""DeepInfer 101 - Project 1: Kernel Zoo.

You will build a small library of reference kernels, then a tiled matrix
multiply, then a roofline-based benchmark report. Read ``README.md`` first.

Rules:
  - Standard library only for Phase 1 and Phase 2.
  - Keep the function signatures unchanged.
  - Do not use NumPy/PyTorch; the point is to implement the loops yourself.
"""

from __future__ import annotations

import math
from memory import MatrixMemory


# ---------------------------------------------------------------------------
# Phase 1: reference kernels
# ---------------------------------------------------------------------------
def vector_add(a: list[float], b: list[float]) -> list[float]:
    """Q1: Return a new list of ``a[i] + b[i]``; [] + [] returns [].

    Inputs contain finite real numbers. Preserve both inputs. Raise
    ValueError on length mismatch. No external numerical libraries.
    """
    raise NotImplementedError("Project 1 Phase 1: vector_add")


def sum_reduce(x: list[float]) -> float:
    """Q2: Pair adjacent values each round, carrying an odd tail forward.

    Preserve x; return one float. Empty list returns 0.0. Do not replace
    the tree with sum(), math.fsum(), or a left-to-right accumulator.
    """
    raise NotImplementedError("Project 1 Phase 1: sum_reduce")


def softmax(x: list[float]) -> list[float]:
    """Q3: Stable softmax of finite values; empty list returns [].

    Subtract the maximum before exponentiation; preserve x. You may
    reuse your Q2 implementation. Output has the same length as x.
    """
    raise NotImplementedError("Project 1 Phase 1: softmax")


def rmsnorm(x: list[float], weight: list[float], eps: float = 1e-6) -> list[float]:
    """Q4: x[i]*weight[i]/sqrt(mean(x**2)+eps), preserving both inputs.

    Raise ValueError on length mismatch or nonfinite/nonpositive eps.
    Two empty lists return [] after validating eps. Finite input values
    and intermediate squares/sums that fit a Python float are assumed.
    """
    raise NotImplementedError("Project 1 Phase 1: rmsnorm")


def matmul_naive(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Q5: Naive matrix multiply, using one dot product per output.

    ``a`` is ``m x k``, ``b`` is ``k x n``; return ``m x n``.
    Matrices must be nonempty lists of nonempty equal-length rows.
    Raise ValueError on empty, ragged, non-list or mismatched shapes.
    Preserve inputs; output rows must be independent lists.
    """
    raise NotImplementedError("Project 1 Phase 1: matmul_naive")


# ---------------------------------------------------------------------------
# Phase 2: tiling and traffic
# ---------------------------------------------------------------------------
def matmul_tiled(a: list[list[float]], b: list[list[float]], tile: int = 16) -> list[list[float]]:
    """Q6: Matrix product with the same shape contract as Q5.

    Q6: return matmul_tiled_profile(a, b, tile)[0]. Implement the profiled
    engine in Q9 first; one engine serves numerical and traffic experiments.
    tile must be a positive integer (bool is not accepted); otherwise
    raise ValueError. Numerical agreement uses floating-point tolerance.
    """
    raise NotImplementedError("Project 1 Phase 2: matmul_tiled")


def matmul_traffic(n: int, tile: int = 16, dtype_bytes: int = 4) -> tuple[int, int]:
    """Q7: Return ``(naive_bytes, tiled_bytes)`` for an ``n x n`` matmul.

    Assume each element is ``dtype_bytes`` and count **global** traffic only.

    Naive:
      - for each of the ``n^2`` outputs, read ``n`` elements of A and
        ``n`` elements of B: ``2 * n^3`` reads
      - write ``n^2`` outputs: ``n^2`` writes
      - total elements: ``2*n^3 + n^2``

    Tiled:
      - each ``tile x tile`` output block reads ``tile*n`` elements of A
        and ``n*tile`` elements of B
      - q = ceil(n/tile), including partial edge tiles
      - reads: ``2 * n^2 * q`` (valid elements only)
      - writes: ``n^2``
      - total elements: ``2*n^2*q + n^2``

    Both multiplied by ``dtype_bytes``. n is a nonnegative integer;
    tile/dtype_bytes are positive integers; bool is not accepted.
    Raise ValueError otherwise. n=0 returns (0, 0). This is a request
    model, not a hardware DRAM counter and not Python object memory.
    """
    raise NotImplementedError("Project 1 Phase 2: matmul_traffic")


# ---------------------------------------------------------------------------
# Phase 3: roofline report
# ---------------------------------------------------------------------------
def estimate_time(
    bytes_moved: float,
    flops: float,
    bandwidth_gbps: float = 1000.0,
    peak_tflops: float = 100.0,
) -> tuple[float, str]:
    """Q8: Return ``(seconds, bound)`` using the roofline lower bound.

    Defaults are a fictional 1TB/s, 100TFLOP/s accelerator; replace them with
    your GPU's numbers in the report.
    bytes_moved/flops must be finite and >= 0; rates finite and > 0.
    Raise ValueError otherwise. GB/s=1e9 bytes/s; TFLOP/s=1e12 FLOP/s.
    bound is "memory" on a tie (including zero work), else the slower
    resource: "memory" or "compute". This is not a runtime prediction.
    """
    raise NotImplementedError("Project 1 Phase 3: estimate_time")


def matmul_tiled_profile(a, b, tile=16):
    """Q9: return (C, stats) using explicit local tile buffers.

    Same nonempty matrix and positive-integer tile contract as Q5/Q6;
    invalid shapes/tile raise ValueError. Preserve inputs. stats comes
    from MatrixMemory.stats(), with a 4-byte-per-scalar traffic model;
    arithmetic remains Python float, not simulated float32 rounding.
    Do not read a/b directly after constructing MatrixMemory, or fabricate
    counts from formulas. See README for a loop scaffold and worked example.
    """
    # 1. Validate shapes; construct memory = MatrixMemory(a, b).
    # 2. For each output tile, create an independent accumulator tile.
    # 3. For each k tile, load bounded A/B tiles through memory.
    # 4. Multiply using local tiles; preserve accumulator between k tiles.
    # 5. Store each valid C once; return output and memory.stats().
    raise NotImplementedError("Project 1 Q9: explicit tile reuse")
