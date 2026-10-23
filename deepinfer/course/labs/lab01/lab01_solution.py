"""Reference solution for Lab 1. Try the lab yourself before reading this."""

from __future__ import annotations

import math


def sum_reduce(x):
    values = [float(v) for v in x]
    if not values:
        return 0.0
    while len(values) > 1:
        next_values = []
        for i in range(0, len(values) - 1, 2):
            next_values.append(values[i] + values[i + 1])
        if len(values) % 2:
            next_values.append(values[-1])
        values = next_values
    return values[0]


def online_softmax_state(x):
    m = -math.inf
    s = 0.0
    for v in x:
        v = float(v)
        m_new = max(m, v)
        s = s * math.exp(m - m_new) + math.exp(v - m_new)
        m = m_new
    return m, s


def softmax(x):
    if not x:
        return []
    m, s = online_softmax_state(x)
    return [math.exp(float(v) - m) / s for v in x]


def rmsnorm(x, weight, eps=1e-6):
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if len(x) != len(weight):
        raise ValueError("x and weight must have the same length")
    if not x:
        return []
    mean_sq = sum_reduce([float(v) * float(v) for v in x]) / len(x)
    scale = 1.0 / math.sqrt(mean_sq + eps)
    return [float(v) * scale * float(w) for v, w in zip(x, weight)]


def layernorm(x, weight, bias, eps=1e-5):
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if not (len(x) == len(weight) == len(bias)):
        raise ValueError("x, weight and bias must have the same length")
    if not x:
        return []
    mean = sum_reduce(x) / len(x)
    var = sum_reduce([(float(v) - mean) ** 2 for v in x]) / len(x)
    scale = 1.0 / math.sqrt(var + eps)
    return [(float(v) - mean) * scale * float(w) + float(b) for v, w, b in zip(x, weight, bias)]


def softmax_traffic(n, passes=3):
    if type(n) is not int or n < 0:
        raise ValueError("n must be a non-negative integer")
    if type(passes) is not int:
        raise ValueError("passes must be 1, 2 or 3")
    if passes == 3:
        return 16 * n
    if passes == 2:
        return 12 * n
    if passes == 1:
        return 8 * n
    raise ValueError("passes must be 1, 2 or 3")


def rmsnorm_traffic(n, on_chip=True):
    if type(n) is not int or n < 0:
        raise ValueError("n must be a non-negative integer")
    return 8 * n if on_chip else 12 * n
