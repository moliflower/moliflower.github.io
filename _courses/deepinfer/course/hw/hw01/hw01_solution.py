"""Reference solution for Homework 1. Try the homework yourself first."""

from __future__ import annotations


def arithmetic_intensity(flops, bytes_moved):
    if bytes_moved <= 0 or flops < 0:
        raise ValueError("bytes_moved must be positive; flops must be non-negative")
    return flops / bytes_moved


def roofline_time(bytes_moved, flops, bandwidth_gbps, peak_tflops):
    if bandwidth_gbps <= 0 or peak_tflops <= 0 or min(bytes_moved, flops) < 0:
        raise ValueError("hardware must be positive; workload must be non-negative")
    memory_s = bytes_moved / (bandwidth_gbps * 1e9)
    compute_s = flops / (peak_tflops * 1e12)
    if memory_s >= compute_s:
        return memory_s, "memory"
    return compute_s, "compute"


def kv_cache_bytes(num_layers, num_kv_heads, head_dim, seq_len, batch=1, bytes_per_elem=2):
    if min(num_layers, num_kv_heads, head_dim, seq_len, batch, bytes_per_elem) < 0:
        raise ValueError("all arguments must be non-negative")
    return 2 * num_layers * num_kv_heads * head_dim * seq_len * batch * bytes_per_elem


def mla_kv_cache_bytes(num_layers, kv_lora_rank, qk_rope_head_dim, seq_len, batch=1, bytes_per_elem=1):
    if min(num_layers, kv_lora_rank, qk_rope_head_dim, seq_len, batch, bytes_per_elem) < 0:
        raise ValueError("all arguments must be non-negative")
    return num_layers * (kv_lora_rank + qk_rope_head_dim) * seq_len * batch * bytes_per_elem


def attention_flops(batch, seq_len, num_heads, head_dim, causal=True):
    if min(batch, seq_len, num_heads, head_dim) < 0:
        raise ValueError("all arguments must be non-negative")
    pairs = seq_len * (seq_len + 1) // 2 if causal else seq_len * seq_len
    return 4 * batch * num_heads * pairs * head_dim


def quantize_per_tensor(x, bits=8):
    if type(bits) is not int or not 2 <= bits <= 16:
        raise ValueError("bits must be an integer in [2, 16]")
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax
    amax = max((abs(float(v)) for v in x), default=0.0)
    scale = 1.0 if amax == 0.0 else amax / qmax
    q = []
    for v in x:
        qv = round(float(v) / scale)
        q.append(max(qmin, min(qmax, qv)))
    return q, scale


def dequantize_per_tensor(q, scale):
    return [float(v) * scale for v in q]


def quant_error(x, bits=8):
    q, scale = quantize_per_tensor(x, bits)
    if not x:
        return 0.0
    deq = dequantize_per_tensor(q, scale)
    return max(abs(float(a) - b) for a, b in zip(x, deq))


def max_kv_tokens(budget_bytes, num_layers, num_kv_heads, head_dim, bytes_per_elem=2):
    if budget_bytes < 0 or min(num_layers, num_kv_heads, head_dim, bytes_per_elem) <= 0:
        raise ValueError("budget must be non-negative; per-token dimensions must be positive")
    per_token = kv_cache_bytes(num_layers, num_kv_heads, head_dim, 1,
                               bytes_per_elem=bytes_per_elem)
    return budget_bytes // per_token
