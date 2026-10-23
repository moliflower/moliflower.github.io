"""Reference solution for Lab 2. Try the lab yourself before reading this."""

from __future__ import annotations

import math


def matmul(a, b):
    if not a or not b:
        return []
    m, k, n = len(a), len(b), len(b[0])
    if len(a[0]) != k:
        raise ValueError("inner dimensions do not match")
    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            total = 0.0
            for kk in range(k):
                total += float(a[i][kk]) * float(b[kk][j])
            out[i][j] = total
    return out


def transpose(a):
    if not a:
        return []
    return [[float(a[i][j]) for i in range(len(a))] for j in range(len(a[0]))]


def softmax_rows(x):
    out = []
    for row in x:
        if not row or any(math.isnan(v) or v == math.inf for v in row):
            raise ValueError("each row needs finite values or -inf")
        m = max(float(v) for v in row)
        if m == -math.inf:
            raise ValueError("each row needs at least one finite value")
        exps = [math.exp(float(v) - m) for v in row]
        s = sum(exps)
        out.append([v / s for v in exps])
    return out


def causal_mask(scores):
    n = len(scores)
    if any(len(row) != n for row in scores):
        raise ValueError("scores must be square")
    return [
        [float(scores[i][j]) if j <= i else -math.inf for j in range(n)]
        for i in range(n)
    ]


def attention(q, k, v, causal=True):
    if not q or not k:
        return []
    d = len(q[0])
    if not v or len(k) != len(v):
        raise ValueError("k and v must have the same nonzero sequence length")
    if d == 0:
        raise ValueError("head dimension must be positive")
    if len(k[0]) != d or len(v[0]) != d:
        raise ValueError("q, k, v must have the same head dimension")
    if causal and len(q) != len(k):
        raise ValueError("causal attention assumes seq_q == seq_k")
    scores = matmul(q, transpose(k))
    scale = 1.0 / math.sqrt(d)
    scores = [[value * scale for value in row] for row in scores]
    if causal:
        scores = causal_mask(scores)
    probs = softmax_rows(scores)
    return matmul(probs, v)


def _split_heads(x, heads, head_dim):
    return [[row[h * head_dim : (h + 1) * head_dim] for row in x] for h in range(heads)]


def multi_head_attention(q, k, v, num_heads):
    if type(num_heads) is not int or num_heads <= 0:
        raise ValueError("num_heads must be a positive integer")
    if not (len(q) == len(k) == len(v)):
        raise ValueError("q, k, v must have the same sequence length")
    if not q:
        return []
    d_model = len(q[0])
    if d_model == 0 or d_model % num_heads != 0:
        raise ValueError("positive d_model must be divisible by num_heads")
    if len(k[0]) != d_model or len(v[0]) != d_model:
        raise ValueError("q, k, v must have the same model dimension")
    head_dim = d_model // num_heads
    q_heads = _split_heads(q, num_heads, head_dim)
    k_heads = _split_heads(k, num_heads, head_dim)
    v_heads = _split_heads(v, num_heads, head_dim)
    head_outputs = [
        attention(q_heads[h], k_heads[h], v_heads[h], causal=True)
        for h in range(num_heads)
    ]
    seq = len(q)
    out = [[0.0] * d_model for _ in range(seq)]
    for h in range(num_heads):
        for i in range(seq):
            out[i][h * head_dim : (h + 1) * head_dim] = head_outputs[h][i]
    return out


def attention_flops(batch, heads, seq_q, seq_k, head_dim, causal=True):
    if any(type(n) is not int or n < 0 for n in (batch, heads, seq_q, seq_k, head_dim)):
        raise ValueError("dimensions must be non-negative integers")
    if causal and seq_q != seq_k:
        raise ValueError("causal attention assumes seq_q == seq_k")
    pairs = seq_q * (seq_q + 1) // 2 if causal else seq_q * seq_k
    return 4 * batch * heads * pairs * head_dim


def attention_bytes(
    batch,
    heads,
    seq_q,
    seq_k,
    head_dim,
    bytes_per_elem=2,
    materialized=True,
):
    if any(type(n) is not int or n < 0 for n in (batch, heads, seq_q, seq_k, head_dim, bytes_per_elem)):
        raise ValueError("dimensions and bytes_per_elem must be non-negative integers")
    elements = batch * heads * (2 * seq_q + 2 * seq_k) * head_dim
    if materialized:
        # S write/read, P write/read in a three-stage materialized pipeline.
        elements += 4 * batch * heads * seq_q * seq_k
    return elements * bytes_per_elem
