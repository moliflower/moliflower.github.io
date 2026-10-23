"""Lab0 小步练习：先填索引，再复用函数，最后独立实现和修错。

从 README.md 开始；只用标准库，不改函数签名。
原有 vector_add / strided / compare / roofline 接口保留。
"""

from __future__ import annotations

from math import ceil

from simt import make_vector_gpu


def coalesced_index(tid, bid, block_dim):
    """Q1：连续分配；例如 tid=2,bid=1,block_dim=4，返回 6。"""
    raise NotImplementedError("Lab0 Q1：只填全局索引公式")


def strided_index(tid, bid, grid_dim):
    """Q2：跨组分配；例如 tid=2,bid=1,grid_dim=3，返回 7。"""
    raise NotImplementedError("Lab0 Q2：只填跨步索引公式")


def _run_add(a, b, block, strided=False):
    """给定的基础设施：安排内存、运行每个线程、读取结果。"""
    if not 1 <= block <= 1024:
        raise ValueError("block must be between 1 and 1024")
    gpu, a_addr, b_addr, c_addr, n = make_vector_gpu(a, b)
    grid = ceil(n / block)

    def kernel(tid, bid, block_dim, grid_dim, device):
        if strided:
            i = strided_index(tid, bid, grid_dim)
        else:
            i = coalesced_index(tid, bid, block_dim)
        if i < n:  # 尾部判断已提供；Q6 会让你自己修复一个错误版本。
            left = device.load(a_addr + i)
            right = device.load(b_addr + i)
            device.store(c_addr + i, left + right)

    gpu.run(kernel, grid, block)
    return gpu.global_mem[c_addr:c_addr + n], gpu.stats()


def vector_add(a: list[int], b: list[int], block: int = 128) -> tuple[list[int], dict[str, int]]:
    """Q1: coalesced vector add.

    Return ``(c, stats)`` where ``c[i] = a[i] + b[i]`` and ``stats`` is
    ``gpu.stats()``.

    Requirements:
      - one thread computes one output element
      - use ``i = bid * block + tid`` (coalesced)
      - guard with ``if i < n``
      - ``block`` must be between 1 and 1024; the last block may be partial
      - ``n == 0`` must return ``([], stats)`` with zero traffic

    Hint: use ``simt.make_vector_gpu`` to lay out ``a``, ``b`` and ``c``.
    The returned addresses are ``(a_addr, b_addr, c_addr) = (0, n, 2*n)``.
    """
    return _run_add(a, b, block)


def vector_add_strided(a: list[int], b: list[int], block: int = 128) -> tuple[list[int], dict[str, int]]:
    """Q2: deliberately uncoalesced vector add.

    Same result as Q1, but use ``i = tid * grid + bid`` instead of
    ``i = bid * block + tid``. This is the "bad" access pattern used to
    measure the cost of poor coalescing.

    Hint: the grid size is still ``ceil(n / block)``.
    """
    return _run_add(a, b, block, strided=True)


def compare_transactions(n: int = 1024, block: int = 128) -> tuple[int, int]:
    """Q3: return ``(coalesced_transactions, strided_transactions)``.

    Run both kernels on two arrays of length ``n`` and return the
    ``global_transactions`` from each ``stats`` dictionary.
    """
    raise NotImplementedError("Lab 0 Q3: implement compare_transactions")


def roofline(
    bytes_moved: float,
    flops: float,
    bandwidth_gbps: float,
    peak_tflops: float,
) -> tuple[float, str]:
    """Q4: a one-line roofline model.

    Return ``(predicted_seconds, bound)`` where ``bound`` is either
    ``"memory"`` or ``"compute"``.

    - ``bandwidth_gbps`` is in GB/s (10^9 bytes/s)
    - ``peak_tflops`` is in TFLOP/s (10^12 FLOP/s)
    - memory time = bytes_moved / (bandwidth_gbps * 1e9)
    - compute time = flops / (peak_tflops * 1e12)
    - the predicted time is the larger of the two
    - on a tie, call it ``"memory"``
    """
    raise NotImplementedError("Lab 0 Q4: implement roofline")


def vector_scale(a, factor, block=128):
    """Q5 独立实现：c[i]=a[i]*factor，返回 (c, stats)。

    通过 tracer 每元素读一次、写一次；空输入零流量；
    block 必须在 1..1024；不修改 a。
    """
    raise NotImplementedError("Lab0 Q5：独立实现 vector_scale")


def vector_add_debug(a, b, block=4):
    """Q6 修错：定位不完整 block 的越界，规格与 vector_add 相同。"""
    if not 1 <= block <= 1024:
        raise ValueError("block must be between 1 and 1024")
    gpu, a_addr, b_addr, c_addr, n = make_vector_gpu(a, b)

    def kernel(tid, bid, block_dim, grid_dim, device):
        i = bid * block_dim + tid
        if i <= n:  # TODO: 先用 n=5,block=4 最小复现，不要改内存大小掩盖错误。
            left = device.load(a_addr + i)
            right = device.load(b_addr + i)
            device.store(c_addr + i, left + right)

    gpu.run(kernel, ceil(n / block), block)
    return gpu.global_mem[c_addr:c_addr + n], gpu.stats()


if __name__ == "__main__":
    # A tiny manual smoke test. It will fail until Q1 is implemented.
    c, stats = vector_add([1, 2, 3], [4, 5, 6])
    print("c =", c)
    print("stats =", stats)
    print("compare =", compare_transactions())
    print("roofline =", roofline(1_000_000_000, 10_000_000_000, 100, 10))
