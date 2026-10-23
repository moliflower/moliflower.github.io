"""Reference solution for Lab 0. Try the lab yourself before reading this."""

from __future__ import annotations

from math import ceil

from simt import make_vector_gpu


def coalesced_index(tid, bid, block_dim):
    return bid * block_dim + tid


def strided_index(tid, bid, grid_dim):
    return tid * grid_dim + bid


def _add_kernel(tid, bid, bdim, gdim, gpu, a_addr, b_addr, c_addr, n, strided):
    i = tid * gdim + bid if strided else bid * bdim + tid
    if i < n:
        gpu.store(c_addr + i, gpu.load(a_addr + i) + gpu.load(b_addr + i))


def _run(a, b, block, strided):
    if not 1 <= block <= 1024:
        raise ValueError("block must be between 1 and 1024")
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    gpu, a_addr, b_addr, c_addr, n = make_vector_gpu(list(a), list(b))
    grid = ceil(n / block)
    gpu.run(_add_kernel, grid, block, a_addr, b_addr, c_addr, n, strided)
    return gpu.global_mem[c_addr : c_addr + n], gpu.stats()


def vector_add(a, b, block=128):
    return _run(a, b, block, False)


def vector_add_strided(a, b, block=128):
    return _run(a, b, block, True)


def compare_transactions(n=1024, block=128):
    _, coalesced = vector_add([0] * n, [0] * n, block)
    _, strided = vector_add_strided([0] * n, [0] * n, block)
    return coalesced["global_transactions"], strided["global_transactions"]


def roofline(bytes_moved, flops, bandwidth_gbps, peak_tflops):
    if bandwidth_gbps <= 0 or peak_tflops <= 0:
        raise ValueError("bandwidth and peak must be positive")
    memory_s = bytes_moved / (bandwidth_gbps * 1e9)
    compute_s = flops / (peak_tflops * 1e12)
    if memory_s >= compute_s:
        return memory_s, "memory"
    return compute_s, "compute"


def vector_scale(a, factor, block=128):
    if not 1 <= block <= 1024:
        raise ValueError("block must be between 1 and 1024")
    gpu, a_addr, _, c_addr, n = make_vector_gpu(a, [0] * len(a))

    def kernel(tid, bid, block_dim, grid_dim, device):
        i = coalesced_index(tid, bid, block_dim)
        if i < n:
            device.store(c_addr + i, device.load(a_addr + i) * factor)

    gpu.run(kernel, ceil(n / block), block)
    return gpu.global_mem[c_addr:c_addr + n], gpu.stats()


def vector_add_debug(a, b, block=4):
    return vector_add(a, b, block)
