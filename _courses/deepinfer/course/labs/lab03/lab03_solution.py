"""Reference solution for Lab 3. Try the lab yourself before reading this."""

from __future__ import annotations

class KVCache:
    def __init__(self, head_dim: int) -> None:
        if type(head_dim) is not int or head_dim <= 0:
            raise ValueError("head_dim must be a positive integer")
        self.head_dim = head_dim
        self.k: list[list[float]] = []
        self.v: list[list[float]] = []

    def append(self, k: list[float], v: list[float]) -> int:
        if len(k) != self.head_dim or len(v) != self.head_dim:
            raise ValueError("k and v must have length head_dim")
        new_k, new_v = [float(x) for x in k], [float(x) for x in v]
        self.k.append(new_k)
        self.v.append(new_v)
        return len(self.k) - 1

    def get(self, start: int, end: int):
        if type(start) is not int or type(end) is not int or start < 0 or end < start or end > len(self):
            raise ValueError("invalid range")
        return (
            [row[:] for row in self.k[start:end]],
            [row[:] for row in self.v[start:end]],
        )

    def __len__(self) -> int:
        return len(self.k)

    def free(self) -> None:
        self.k.clear()
        self.v.clear()


class PagedKVCache:
    def __init__(self, block_size: int) -> None:
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        self.block_size = block_size
        self.blocks: dict[int, list[tuple[list[float], list[float]]]] = {}
        self.block_table: list[int] = []
        self.length = 0
        self._next_block = 0

    def append(self, k: list[float], v: list[float]) -> int:
        if len(k) != len(v):
            raise ValueError("k and v must have the same length")
        new_k, new_v = [float(x) for x in k], [float(x) for x in v]
        if self.length % self.block_size == 0:
            block_id = self._next_block
            self._next_block += 1
            self.blocks[block_id] = []
            self.block_table.append(block_id)
        current = self.block_table[-1]
        self.blocks[current].append((new_k, new_v))
        self.length += 1
        return self.length - 1

    def get(self, start: int, end: int):
        if type(start) is not int or type(end) is not int or start < 0 or end < start or end > self.length:
            raise ValueError("invalid range")
        out_k = []
        out_v = []
        for pos in range(start, end):
            block_id = self.block_table[pos // self.block_size]
            offset = pos % self.block_size
            k, v = self.blocks[block_id][offset]
            out_k.append(k[:])
            out_v.append(v[:])
        return out_k, out_v

    def allocated_slots(self) -> int:
        return len(self.block_table) * self.block_size

    def waste(self) -> int:
        return self.allocated_slots() - self.length

    def __len__(self) -> int:
        return self.length

    def free(self) -> None:
        self.blocks.clear()
        self.block_table.clear()
        self.length = 0
        self._next_block = 0
