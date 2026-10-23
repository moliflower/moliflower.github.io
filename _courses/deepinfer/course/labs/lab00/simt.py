"""A tiny deterministic SIMT simulator for DeepInfer 101.

This is *not* a GPU emulator. It executes one thread at a time and records
global/shared memory accesses. That is enough to teach:

  - grid / block / thread / warp indexing
  - memory coalescing (128-byte cache lines)
  - shared-memory bank conflicts (32 banks x 4 bytes)

The simulator is intentionally small so that you can read it in one sitting.
It is also the "CPU reference" for later labs: when you port a kernel to
TileLang or CUDA, the reference output and traffic numbers come from here.
"""

from __future__ import annotations

from collections import defaultdict

WARP_SIZE = 32
CACHE_LINE_WORDS = 32  # 32 words * 4 bytes = 128 bytes
SECTOR_WORDS = 8       # 8 words * 4 bytes = 32 bytes
NUM_BANKS = 32
WORD_BYTES = 4


class Access:
    """One memory access issued by one thread."""

    __slots__ = ("bid", "tid", "pc", "space", "kind", "addr")

    def __init__(self, bid: int, tid: int, pc: int, space: str, kind: str, addr: int) -> None:
        self.bid = bid
        self.tid = tid
        self.pc = pc
        self.space = space
        self.kind = kind
        self.addr = addr

    def __repr__(self) -> str:
        return f"Access(bid={self.bid}, tid={self.tid}, pc={self.pc}, {self.space}_{self.kind}, addr={self.addr})"


class GPU:
    """A deterministic SIMT device with global and shared memory.

    Addresses are word addresses (4 bytes each), not byte addresses. This keeps
    the arithmetic simple: address 0 is the first 4-byte word.
    """

    def __init__(self, global_size: int, shared_size: int = 0) -> None:
        if global_size < 0 or shared_size < 0:
            raise ValueError("memory sizes must be non-negative")
        self.global_mem = [0] * global_size
        self.shared_mem = [0] * shared_size
        self.trace: list[Access] = []
        self._bid = -1
        self._tid = -1
        self._pc: dict[tuple[int, int], int] = {}

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(self, kernel, grid: int, block: int, *args) -> None:
        """Run ``kernel`` for every thread.

        The kernel signature is::

            kernel(tid, bid, block_dim, grid_dim, gpu, *args)

        Threads are executed sequentially in ``(bid, tid)`` order. This means
        the simulator cannot model true parallelism or barriers. It is a
        *memory-access tracer*, not a performance simulator.
        """
        if grid < 0 or block < 0:
            raise ValueError("grid and block must be non-negative")
        if block > 1024:
            raise ValueError("block size > 1024 is not supported")
        for bid in range(grid):
            for tid in range(block):
                self._bid, self._tid = bid, tid
                self._pc[(bid, tid)] = 0
                kernel(tid, bid, block, grid, self, *args)
        self._bid, self._tid = -1, -1

    # ------------------------------------------------------------------
    # Memory operations (the only way a kernel can touch memory)
    # ------------------------------------------------------------------
    def _record(self, space: str, kind: str, addr: int) -> None:
        size = len(self.global_mem) if space == "global" else len(self.shared_mem)
        if not 0 <= addr < size:
            raise IndexError(f"{space} address out of range: {addr} (size={size})")
        key = (self._bid, self._tid)
        pc = self._pc.get(key, 0)
        self._pc[key] = pc + 1
        self.trace.append(Access(self._bid, self._tid, pc, space, kind, addr))

    def load(self, addr: int) -> int:
        self._record("global", "load", addr)
        return self.global_mem[addr]

    def store(self, addr: int, value: int) -> None:
        self._record("global", "store", addr)
        self.global_mem[addr] = value

    def shared_load(self, addr: int) -> int:
        self._record("shared", "load", addr)
        return self.shared_mem[addr]

    def shared_store(self, addr: int, value: int) -> None:
        self._record("shared", "store", addr)
        self.shared_mem[addr] = value

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def reset_trace(self) -> None:
        self.trace.clear()

    @property
    def global_loads(self) -> int:
        return sum(1 for a in self.trace if a.space == "global" and a.kind == "load")

    @property
    def global_stores(self) -> int:
        return sum(1 for a in self.trace if a.space == "global" and a.kind == "store")

    @property
    def shared_loads(self) -> int:
        return sum(1 for a in self.trace if a.space == "shared" and a.kind == "load")

    @property
    def shared_stores(self) -> int:
        return sum(1 for a in self.trace if a.space == "shared" and a.kind == "store")

    def _global_groups(self, unit_words: int) -> int:
        """Count touched units per straight-line warp memory instruction.

        pc is a per-thread access ordinal, not a real program counter.
        Diverging memory paths/unequal loop counts are outside this tracer's
        model: their instruction identities cannot be inferred from ordinal.
        """
        lines: dict[tuple[int, int, int, str], set[int]] = defaultdict(set)
        for a in self.trace:
            if a.space != "global":
                continue
            key = (a.bid, a.tid // WARP_SIZE, a.pc, a.kind)
            lines[key].add(a.addr // unit_words)
        return sum(len(v) for v in lines.values())

    def transactions(self) -> int:
        """Legacy Lab0 metric: touched 128B lines, not measured DRAM traffic."""
        return self._global_groups(CACHE_LINE_WORDS)

    def sectors(self) -> int:
        """Touched 32B sectors; no cache hits, write policy or timing modeled."""
        return self._global_groups(SECTOR_WORDS)

    def bank_conflicts(self) -> int:
        """Sum of (max distinct words in a bank - 1) per warp instruction.

        Same-word reads broadcast and count once. This models scalar 4B
        accesses only. Concurrent writes to the same word are a data race;
        their resulting value is not modeled by this sequential tracer.
        """
        groups = defaultdict(lambda: defaultdict(set))
        for a in self.trace:
            if a.space != "shared":
                continue
            key = (a.bid, a.tid // WARP_SIZE, a.pc, a.kind)
            groups[key][a.addr % NUM_BANKS].add(a.addr)
        return sum(max(len(words) for words in banks.values()) - 1 for banks in groups.values())

    def stats(self) -> dict[str, int]:
        """Return a snapshot of the counters used by the labs."""
        g_loads = self.global_loads
        g_stores = self.global_stores
        return {
            "global_loads": g_loads,
            "global_stores": g_stores,
            "shared_loads": self.shared_loads,
            "shared_stores": self.shared_stores,
            "global_transactions": self.transactions(),
            "global_sectors": self.sectors(),
            "sector_bytes": self.sectors() * SECTOR_WORDS * WORD_BYTES,
            "shared_bank_conflicts": self.bank_conflicts(),
            "bytes_moved": WORD_BYTES * (g_loads + g_stores),  # requested bytes, not HBM traffic
        }


def make_vector_gpu(a: list[int], b: list[int]) -> tuple[GPU, int, int, int, int]:
    """Lay out ``a``, ``b`` and an output array in one global memory.

    Returns ``(gpu, a_addr, b_addr, c_addr, n)``. This helper is used by the
    lab so that every student uses the same memory layout.
    """
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    n = len(a)
    gpu = GPU(3 * n)
    gpu.global_mem[0:n] = list(a)
    gpu.global_mem[n : 2 * n] = list(b)
    return gpu, 0, n, 2 * n, n


def read_vector(gpu: GPU, addr: int, n: int) -> list[int]:
    """Read ``n`` words starting at ``addr`` without recording an access."""
    return gpu.global_mem[addr : addr + n]
