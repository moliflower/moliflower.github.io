"""Reference solution for Homework 2. Try the homework yourself first."""

from __future__ import annotations

def attention_flops(batch, heads, seq_q, seq_k, head_dim, causal=True):
    if min(batch, heads, seq_q, seq_k, head_dim) < 0:
        raise ValueError("all arguments must be non-negative")
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
    if min(batch, heads, seq_q, seq_k, head_dim, bytes_per_elem) < 0:
        raise ValueError("all arguments must be non-negative")
    elements = batch * heads * (2 * seq_q + 2 * seq_k) * head_dim
    if materialized:
        elements += 4 * batch * heads * seq_q * seq_k
    return elements * bytes_per_elem


def attention_ai(
    batch,
    heads,
    seq_q,
    seq_k,
    head_dim,
    bytes_per_elem=2,
    causal=True,
    materialized=True,
):
    flops = attention_flops(batch, heads, seq_q, seq_k, head_dim, causal)
    moved = attention_bytes(
        batch, heads, seq_q, seq_k, head_dim, bytes_per_elem, materialized
    )
    if moved == 0:
        raise ValueError("bytes moved must be positive")
    return flops / moved


def bound(ai, peak_tflops, bandwidth_gbps):
    if peak_tflops <= 0 or bandwidth_gbps <= 0 or ai < 0:
        raise ValueError("hardware numbers must be positive; ai must be non-negative")
    ridge = (peak_tflops * 1e12) / (bandwidth_gbps * 1e9)
    return "compute" if ai >= ridge else "memory"


def _check_non_negative(**kwargs):
    for name, value in kwargs.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")


def kv_cache_bytes(
    num_layers,
    num_kv_heads,
    head_dim,
    seq_len,
    batch=1,
    bytes_per_elem=2,
):
    _check_non_negative(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        batch=batch,
        bytes_per_elem=bytes_per_elem,
    )
    return 2 * num_layers * num_kv_heads * head_dim * seq_len * batch * bytes_per_elem


def kv_cache_growth(
    num_layers,
    num_kv_heads,
    head_dim,
    new_tokens,
    batch=1,
    bytes_per_elem=2,
):
    return kv_cache_bytes(
        num_layers, num_kv_heads, head_dim, new_tokens, batch, bytes_per_elem
    )


def num_pages(seq_len, page_size):
    if seq_len < 0 or page_size <= 0:
        raise ValueError("seq_len >= 0 and page_size > 0")
    return (seq_len + page_size - 1) // page_size


def page_waste(seq_len, page_size):
    return num_pages(seq_len, page_size) * page_size - seq_len


def paged_kv_bytes(
    num_layers,
    num_kv_heads,
    head_dim,
    seq_len,
    page_size,
    batch=1,
    bytes_per_elem=2,
):
    allocated = num_pages(seq_len, page_size) * page_size
    return kv_cache_bytes(
        num_layers, num_kv_heads, head_dim, allocated, batch, bytes_per_elem
    )


def prefix_cache_savings(
    num_layers,
    num_kv_heads,
    head_dim,
    prefix_len,
    batch=1,
    bytes_per_elem=2,
):
    return kv_cache_bytes(
        num_layers, num_kv_heads, head_dim, prefix_len, batch, bytes_per_elem
    )


def paged_batch_bytes(seq_lens, page_size, num_layers, num_kv_heads, head_dim,
                      bytes_per_elem=2):
    # Validate common parameters even when there are no sequences to visit.
    paged_kv_bytes(num_layers, num_kv_heads, head_dim, 0, page_size,
                   bytes_per_elem=bytes_per_elem)
    return sum(paged_kv_bytes(num_layers, num_kv_heads, head_dim, length,
                              page_size, bytes_per_elem=bytes_per_elem)
               for length in seq_lens)
