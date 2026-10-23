"""DeepInfer 101 - Lab 3: KV Cache (continuous + paged).

Fill in the two classes below. Read ``README.md`` first.

Rules:
  - Standard library only.
  - Keep the class/method signatures unchanged.
  - Copies: ``append`` should copy the input vectors; ``get`` should return
    copies, so the caller cannot mutate the cache by accident.
"""

from __future__ import annotations

class KVCache:
    """A single-layer KV cache with a contiguous layout.

    This is the "easy" version: token i lives at position i.
    """

    def __init__(self, head_dim: int) -> None:
        if type(head_dim) is not int or head_dim <= 0:
            raise ValueError("head_dim must be a positive integer")
        self.head_dim = head_dim
        self.k: list[list[float]] = []
        self.v: list[list[float]] = []

    def append(self, k: list[float], v: list[float]) -> int:
        """Append float copies of one token's finite K/V. Return old length.

        Each vector must have head_dim elements, else raise ValueError and
        leave the entire cache unchanged. Nonfinite input is out of scope.
        """
        raise NotImplementedError("Lab 3: KVCache.append")

    def get(self, start: int, end: int) -> tuple[list[list[float]], list[list[float]]]:
        """Return K/V for positions ``[start, end)``. Raise ``ValueError``
        unless start/end are ints (not bool) and 0<=start<=end<=len(self).
        Return fresh outer lists AND fresh vector rows; empty ranges are valid.
        """
        raise NotImplementedError("Lab 3: KVCache.get")

    def __len__(self) -> int:
        return len(self.k)

    def free(self) -> None:
        self.k.clear()
        self.v.clear()


class PagedKVCache:
    """A single-layer KV cache with paged allocation.

    ``block_table`` is the logical -> physical mapping. A new physical block
    is allocated when the current token position is a multiple of
    ``block_size``. The last block may be partially filled; that is the
    "page waste" you will calculate.
    """

    def __init__(self, block_size: int) -> None:
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        self.block_size = block_size
        self.blocks: dict[int, list[tuple[list[float], list[float]]]] = {}
        self.block_table: list[int] = []
        self.length = 0
        self._next_block = 0

    def append(self, k: list[float], v: list[float]) -> int:
        """Append float copies, allocating a page only at a page boundary.

        k/v must have equal lengths, else ValueError without state changes.
        This teaching container does not fix a head_dim between tokens.
        Input values are finite numbers; return the old logical length.
        """
        raise NotImplementedError("Lab 3: PagedKVCache.append")

    def get(self, start: int, end: int) -> tuple[list[list[float]], list[list[float]]]:
        """Gather K/V for positions ``[start, end)`` across blocks.

        Raise ValueError unless start/end are ints (not bool) satisfying
        0<=start<=end<=len(self). Return copies of both outer and inner lists.
        Look up physical IDs through block_table; do not use logical IDs
        directly as dictionary keys.
        """
        raise NotImplementedError("Lab 3: PagedKVCache.get")

    def allocated_slots(self) -> int:
        """Total slots in allocated blocks (including the last-page waste)."""
        return len(self.block_table) * self.block_size

    def waste(self) -> int:
        """Allocated slots minus used tokens."""
        return self.allocated_slots() - self.length

    def __len__(self) -> int:
        return self.length

    def free(self) -> None:
        self.blocks.clear()
        self.block_table.clear()
        self.length = 0
        self._next_block = 0
