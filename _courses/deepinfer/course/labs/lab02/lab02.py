"""DeepInfer 101 - Lab 2: Attention from Scratch.

Fill in the functions below. Read ``README.md`` first.

Rules:
  - Standard library only (``math`` is allowed).
  - Do not use NumPy/PyTorch; the point is to implement the loops yourself.
  - Keep the function signatures unchanged.

Matrices are rectangular nested lists. Finite inputs and representable
intermediates are assumed, except softmax_rows also accepts masked -inf.
All results are fresh lists; inputs must remain unchanged. Counts exclude bool.
"""

from __future__ import annotations

import math


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Q1: matrix multiply. ``a`` is m x k, ``b`` is k x n; return m x n.

    Empty a or b returns [] (this Lab does not encode general zero-axis
    shapes). Otherwise raise ValueError for mismatched inner dimensions.
    """
    raise NotImplementedError("Lab 2 Q1: implement matmul")


def transpose(a: list[list[float]]) -> list[list[float]]:
    """Q2: return a fresh transpose of a rectangular matrix; [] -> []."""
    raise NotImplementedError("Lab 2 Q2: implement transpose")


def softmax_rows(x: list[list[float]]) -> list[list[float]]:
    """Q3: row-wise numerically stable softmax.

    Subtract the row max, exp, normalize. [] -> []. Raise ValueError for
    an empty row, all -inf row, NaN or +inf. Each valid row contains at least
    one finite value; a masked -inf entry has probability exactly zero.
    """
    raise NotImplementedError("Lab 2 Q3: implement softmax_rows")


def causal_mask(scores: list[list[float]]) -> list[list[float]]:
    """Q4: set ``scores[i][j]`` to ``-inf`` for ``j > i``.

    This is the decoder mask: position i can only see positions <= i.
    Return a new square matrix; [] -> []. Nonsquare scores raise ValueError.
    """
    raise NotImplementedError("Lab 2 Q4: implement causal_mask")


def attention(
    q: list[list[float]],
    k: list[list[float]],
    v: list[list[float]],
    causal: bool = True,
) -> list[list[float]]:
    """Q5: scaled dot-product attention.

    ``q`` is sq x d, ``k``/``v`` are sk x d. Return sq x d.
    Empty q or k returns []. Otherwise require d>0, equal head dimensions,
    nonempty v with len(v)==len(k), and sq==sk when causal, else ValueError.
    Rectangular noncausal attention is valid; offset decode masking is outside
    this Lab. Compose Q1-Q4 rather than reimplementing each stage.

        scores = q @ k^T / sqrt(d)
        if causal: scores = causal_mask(scores)
        probs  = softmax_rows(scores)
        out    = probs @ v
    """
    raise NotImplementedError("Lab 2 Q5: implement attention")


def multi_head_attention(
    q: list[list[float]],
    k: list[list[float]],
    v: list[list[float]],
    num_heads: int,
) -> list[list[float]]:
    """Q6: split the last dimension into ``num_heads`` heads, run attention
    per head, then concatenate the outputs.

    num_heads must be a positive int (not bool). Input sequence lengths
    must match; after these checks, three empty inputs return []. Otherwise
    require equal positive d_model divisible by num_heads, else ValueError.
    Each head uses contiguous columns and causal attention; no projections.
    ``q``/``k``/``v`` are seq x d_model. Return seq x d_model.
    """
    raise NotImplementedError("Lab 2 Q6: implement multi_head_attention")


def attention_flops(
    batch: int,
    heads: int,
    seq_q: int,
    seq_k: int,
    head_dim: int,
    causal: bool = True,
) -> int:
    """Q7: FLOPs of QK^T and AV.

    Full: ``4 * batch * heads * seq_q * seq_k * head_dim``.
    Causal (only when ``seq_q == seq_k``): use seq_q*(seq_q+1)//2
    valid pairs, each costing 4*head_dim FLOPs per batch/head.
    Raise ``ValueError`` if ``causal`` and ``seq_q != seq_k``.
    Dimensions must be nonnegative ints (not bool), else ValueError.
    """
    raise NotImplementedError("Lab 2 Q7: implement attention_flops")


def attention_bytes(
    batch: int,
    heads: int,
    seq_q: int,
    seq_k: int,
    head_dim: int,
    bytes_per_elem: int = 2,
    materialized: bool = True,
) -> int:
    """Q8: idealized read/write model, not tensor capacity or measured HBM IO.

    Always count Q, K, V, O:
        ``batch * heads * (2*seq_q + 2*seq_k) * head_dim`` elements.

    If materialized, count S write/read and P write/read:
        ``4 * batch * heads * seq_q * seq_k`` elements.

    False gives a Q/K/V read-once + O write-once lower bound. Real tiled
    attention may reread K/V. All arrays use bytes_per_elem in this model.

    Multiply by ``bytes_per_elem``.
    Dimensions and bytes_per_elem must be nonnegative ints (not bool),
    else ValueError. Zero describes a zero-work model, not a hardware dtype.
    """
    raise NotImplementedError("Lab 2 Q8: implement attention_bytes")
