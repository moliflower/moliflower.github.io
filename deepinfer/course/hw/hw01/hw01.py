"""DeepInfer 101 - Homework 1: Roofline, KV Cache, and Quantization.

Build cost models, then use them to solve a memory-budget problem.
Counts are non-negative integers; numerical inputs are finite. Negative
counts/workloads raise ValueError. Type checking is not required, except bits.

Rules:
  - Standard library only.
  - Keep the function signatures unchanged.
  - Copy ``REPORT_TEMPLATE.md`` to your own notes for the written checkoff.
"""

from __future__ import annotations


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """Q1: FLOPs per byte.

    Raise ``ValueError`` if ``bytes_moved <= 0`` or ``flops < 0``.
    """
    raise NotImplementedError("HW 1 Q1: implement arithmetic_intensity")


def roofline_time(
    bytes_moved: float,
    flops: float,
    bandwidth_gbps: float,
    peak_tflops: float,
) -> tuple[float, str]:
    """Q2: return ``(predicted_seconds, bound)``.

    Workload must be non-negative; hardware values must be positive.
    Invalid values raise ValueError. On a tie return "memory".
    """
    raise NotImplementedError("HW 1 Q2: implement roofline_time")


def kv_cache_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch: int = 1,
    bytes_per_elem: int = 2,
) -> int:
    """Q3: KV cache size for a standard MHA/GQA model.

        bytes = 2 * num_layers * num_kv_heads * head_dim
                * seq_len * batch * bytes_per_elem

    The leading 2 is for K and V. ``bytes_per_elem=2`` means bf16/fp16.
    """
    raise NotImplementedError("HW 1 Q3: implement kv_cache_bytes")


def mla_kv_cache_bytes(
    num_layers: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    seq_len: int,
    batch: int = 1,
    bytes_per_elem: int = 1,
) -> int:
    """Q4: KV cache size for an MLA model.

    MLA stores one compressed latent per token per layer, plus the RoPE part:

        bytes = num_layers * (kv_lora_rank + qk_rope_head_dim)
                * seq_len * batch * bytes_per_elem

    There is no factor 2: K and V share the same latent.
    """
    raise NotImplementedError("HW 1 Q4: implement mla_kv_cache_bytes")


def attention_flops(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    causal: bool = True,
) -> int:
    """Q5: FLOPs of the two attention matmuls (QK^T and AV).

    Full attention:

        QK^T: 2 * batch * num_heads * seq_len^2 * head_dim
        AV:   2 * batch * num_heads * seq_len^2 * head_dim
        total: 4 * batch * num_heads * seq_len^2 * head_dim

    Causal uses seq_len*(seq_len+1)//2 valid pairs including the diagonal.
    Multiply that by 4*batch*num_heads*head_dim. This is useful work,
    not the work of a dense implementation that computes then masks.
    """
    raise NotImplementedError("HW 1 Q5: implement attention_flops")


def quantize_per_tensor(x: list[float], bits: int = 8) -> tuple[list[int], float]:
    """Q6: symmetric per-tensor quantization.

    - ``qmax = 2**(bits-1) - 1``, ``qmin = -qmax``
    - ``amax = max(abs(v) for v in x)``
    - if ``amax == 0``: ``scale = 1.0``, all q are 0
    - otherwise ``scale = amax / qmax``
    - ``q[i] = clamp(round(x[i] / scale), qmin, qmax)``

    Return a new integer list and positive float scale; do not mutate x.
    Empty input returns ([], 1.0). Use Python round (ties to even).
    Raise ValueError if bits is not an integer in [2, 16] (bool is not a bit count).
    """
    raise NotImplementedError("HW 1 Q6: implement quantize_per_tensor")


def dequantize_per_tensor(q: list[int], scale: float) -> list[float]:
    """Q7: ``dequant[i] = q[i] * scale``."""
    raise NotImplementedError("HW 1 Q7: implement dequantize_per_tensor")


def quant_error(x: list[float], bits: int = 8) -> float:
    """Q8: max absolute quantization error.

    Quantize, dequantize, and return ``max(abs(x[i] - dequant[i]))``.
    An empty list returns ``0.0``. Validate bits even for empty input.
    """
    raise NotImplementedError("HW 1 Q8: implement quant_error")


def max_kv_tokens(
    budget_bytes: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_elem: int = 2,
) -> int:
    """Q9: largest total token count whose effective KV payload fits a budget.

    The budget already excludes model weights and workspace. Return an int.
    Tokens may be distributed across sequences; no paging or prefix sharing.
    Reuse Q3. Raise ValueError for a negative budget or non-positive model
    dimensions/bytes_per_elem. A budget smaller than one token returns 0.
    """
    raise NotImplementedError("HW 1 Q9: implement max_kv_tokens")
