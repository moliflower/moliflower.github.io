"""DeepInfer 101 - Lab 1: Reduction, Softmax, and RMSNorm.

Fill in the functions below. Read ``README.md`` first.

Rules:
  - Standard library only (``math`` is allowed).
  - Do not use ``sum`` for Q1; implement the tree reduction yourself.
  - Keep the function signatures unchanged.

Numeric vector inputs are finite Python numbers with intermediate results
representable as Python floats; NaN/Inf input validation is out of scope.
All functions leave their inputs unchanged. Integer counts exclude bool.
"""

from __future__ import annotations

import math


def sum_reduce(x: list[float]) -> float:
    """Q1: pairwise (tree) reduction.

    Return the sum of ``x`` using a binary tree reduction:

        [a, b, c, d, e] -> [a+b, c+d, e] -> [(a+b)+(c+d), e] -> total

    Do **not** use ``sum``, ``math.fsum`` or a serial fold. Empty returns 0.0.

    Why not a simple loop? A left-to-right loop has a long dependency chain;
    a tree has depth log2(n) and is what GPUs do with warp shuffles.
    """
    raise NotImplementedError("Lab 1 Q1: implement sum_reduce")


def online_softmax_state(x: list[float]) -> tuple[float, float]:
    """Q2: return ``(max_value, sum_exp)`` in one pass over ``x``.

    Use the online update:

        m_new = max(m, v)
        s = s * exp(m - m_new) + exp(v - m_new)
        m = m_new

    Start with ``m = -inf`` and ``s = 0.0``. An empty list returns
    ``(-inf, 0.0)``.

    This is the 1D version of the online softmax used in FlashAttention.
    """
    raise NotImplementedError("Lab 1 Q2: implement online_softmax_state")


def softmax(x: list[float]) -> list[float]:
    """Q3: numerically stable softmax.

    Return a new list ``y`` with ``y[i] = exp(x[i] - m) / s`` where
    ``(m, s) = online_softmax_state(x)``.

    The largest value must map to a finite number even if ``x`` contains
    1000.0; subtracting the max is what prevents overflow.
    Empty input returns a new empty list.
    """
    raise NotImplementedError("Lab 1 Q3: implement softmax")


def rmsnorm(x: list[float], weight: list[float], eps: float = 1e-6) -> list[float]:
    """Q4: RMSNorm.

    Use Q1 for the reduction. Unequal lengths or nonfinite/nonpositive eps
    raise ValueError (validate before the empty-input return). Empty equal
    vectors return [].

        mean_sq = mean(x[i] * x[i])
        scale   = 1 / sqrt(mean_sq + eps)
        y[i]    = x[i] * scale * weight[i]
    """
    raise NotImplementedError("Lab 1 Q4: implement rmsnorm")


def layernorm(
    x: list[float],
    weight: list[float],
    bias: list[float],
    eps: float = 1e-5,
) -> list[float]:
    """Q5: LayerNorm.

    Use Q1 for mean and population variance (divide by n, not n-1).
    Lengths must match and eps must be finite and positive, else ValueError.
    Validate before returning [] for three empty vectors.

        mean = mean(x)
        var  = mean((x[i] - mean) ** 2)
        y[i] = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]
    """
    raise NotImplementedError("Lab 1 Q5: implement layernorm")


def softmax_traffic(n: int, passes: int = 3) -> int:
    """Q6: bytes moved by a softmax over ``n`` float32 values.

    - ``passes=3``: max pass + exp/sum pass + normalize pass
      (read x three times, write y once) -> 16*n bytes
    - ``passes=2``: fused max+sum pass + normalize pass
      (read x twice, write y once) -> 12*n bytes
    - ``passes=1``: fully fused, x stays on-chip
      (read x once, write y once) -> 8*n bytes

    n must be a nonnegative int and passes an int in {1,2,3}; otherwise
    raise ValueError. Boolean values are not integer counts in this Lab.
    """
    raise NotImplementedError("Lab 1 Q6: implement softmax_traffic")


def rmsnorm_traffic(n: int, on_chip: bool = True) -> int:
    """Q7: activation-only requested bytes over n float32 values.

    Weight reads are excluded; add 4*n bytes if weights are fetched once.
    n must be a nonnegative int (not bool), else ValueError.

    - ``on_chip=True``: the row stays in registers/shared memory, so
      read x once + write y once -> 8*n bytes
    - ``on_chip=False``: read x for the sum of squares, then read x again
      to normalize + write y -> 12*n bytes
    """
    raise NotImplementedError("Lab 1 Q7: implement rmsnorm_traffic")
