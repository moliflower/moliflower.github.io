"""Autograder for Lab 0. Run with: python -m unittest -v"""

from __future__ import annotations

import unittest

from lab00 import (
    coalesced_index, compare_transactions, roofline, strided_index,
    vector_add, vector_add_debug, vector_add_strided, vector_scale,
)
from walkthrough import run_example


class TestWalkthrough(unittest.TestCase):
    def test_complete_example(self):
        c, rows, stats = run_example()
        self.assertEqual(c, [11, 22, 33, 44, 55])
        self.assertEqual(rows[-1], (1, 3, 7, "跳过"))
        self.assertEqual(stats["bytes_moved"], 60)


class TestIndex(unittest.TestCase):
    def test_two_blocks(self):
        got = [coalesced_index(tid, bid, 4) for bid in range(2) for tid in range(4)]
        self.assertEqual(got, list(range(8)), "逐行填 bid/tid 表；第二组应从 4 开始。")


class TestVectorAdd(unittest.TestCase):
    def test_basic(self) -> None:
        c, stats = vector_add([1, 2, 3], [4, 5, 6])
        self.assertEqual(c, [5, 7, 9])
        self.assertEqual(stats["global_loads"], 6)
        self.assertEqual(stats["global_stores"], 3)
        self.assertEqual(stats["bytes_moved"], 36)

    def test_empty(self) -> None:
        c, stats = vector_add([], [])
        self.assertEqual(c, [])
        self.assertEqual(stats["global_transactions"], 0)
        self.assertEqual(stats["bytes_moved"], 0)

    def test_partial_block(self) -> None:
        n = 1000
        a = [i % 17 for i in range(n)]
        b = [i % 13 for i in range(n)]
        c, _ = vector_add(a, b)
        self.assertEqual(c, [(x + y) for x, y in zip(a, b)])

    def test_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            vector_add([1, 2], [3])

    def test_coalesced_transactions(self) -> None:
        # 1024 int32 = 4 KiB. A warp reading 32 consecutive words = 1 line.
        # 32 warps * (load a + load b + store c) = 96 transactions.
        _, stats = vector_add([0] * 1024, [0] * 1024)
        self.assertEqual(stats["global_transactions"], 96)

    def test_block_size_32(self) -> None:
        c, stats = vector_add(list(range(100)), list(range(100)), block=32)
        self.assertEqual(c, [2 * i for i in range(100)])
        # n=100 is not a multiple of 32 words, so b and c start mid-line and
        # some warps straddle two 128-byte lines. The exact count is 18 --
        # a useful reminder that alignment matters.
        self.assertEqual(stats["global_transactions"], 18)

    def test_inputs_unchanged(self):
        a, b = [1, -2, 3], [4, 5, -6]
        vector_add(a, b, block=2)
        self.assertEqual(a, [1, -2, 3])
        self.assertEqual(b, [4, 5, -6])

    def test_invalid_block(self):
        for block in (0, -1, 1025):
            with self.subTest(block=block), self.assertRaises(ValueError):
                vector_add([1], [2], block)


class TestStrided(unittest.TestCase):
    def test_index(self):
        self.assertEqual([strided_index(tid, 1, 3) for tid in range(4)], [1, 4, 7, 10])

    def test_partial_nontrivial_grid(self):
        a, b = list(range(10)), [-i for i in range(10)]
        c, stats = vector_add_strided(a, b, block=4)
        self.assertEqual(c, [0] * 10)
        self.assertEqual(stats["global_loads"], 20)

    def test_correctness(self) -> None:
        a = [1, 2, 3, 4, 5]
        b = [10, 20, 30, 40, 50]
        c, _ = vector_add_strided(a, b)
        self.assertEqual(c, [11, 22, 33, 44, 55])

    def test_strided_transactions(self) -> None:
        # stride = grid = 8 words. Each warp touches 8 lines per access.
        # 32 warps * (2 loads + 1 store) * 8 lines = 768.
        _, stats = vector_add_strided([0] * 1024, [0] * 1024)
        self.assertEqual(stats["global_transactions"], 768)


class TestCompare(unittest.TestCase):
    def test_compare_1024(self) -> None:
        self.assertEqual(compare_transactions(1024, 128), (96, 768))

    def test_compare_returns_two_ints(self) -> None:
        result = compare_transactions(256, 64)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], int)
        self.assertIsInstance(result[1], int)


class TestRoofline(unittest.TestCase):
    def test_invalid_hardware(self):
        for bandwidth, peak in ((0, 10), (100, 0), (-1, 10)):
            with self.assertRaises(ValueError):
                roofline(100, 100, bandwidth, peak)

    def test_compute_bound(self) -> None:
        seconds, bound = roofline(1_000_000_000, 1_000_000_000_000, 100, 10)
        self.assertEqual(bound, "compute")
        self.assertAlmostEqual(seconds, 0.1)

    def test_memory_bound(self) -> None:
        seconds, bound = roofline(1_000_000_000, 10_000_000_000, 100, 10)
        self.assertEqual(bound, "memory")
        self.assertAlmostEqual(seconds, 0.01)

    def test_tie_is_memory(self) -> None:
        # 1e9 bytes / 100e9 = 0.01 s; 0.1e12 flops / 10e12 = 0.01 s.
        seconds, bound = roofline(1_000_000_000, 100_000_000_000, 100, 10)
        self.assertEqual(bound, "memory")
        self.assertAlmostEqual(seconds, 0.01)


class TestScale(unittest.TestCase):
    def test_values_and_trace(self):
        a = [1, -2, 3, 0, 5]
        c, stats = vector_scale(a, -3, block=4)
        self.assertEqual(c, [-3, 6, -9, 0, -15])
        self.assertEqual(a, [1, -2, 3, 0, 5])
        self.assertEqual(stats["global_loads"], 5, "本题每元素只读一次，不要额外读占位 b。")
        self.assertEqual(stats["global_stores"], 5)
        self.assertEqual(stats["bytes_moved"], 40)

    def test_empty_and_invalid(self):
        c, stats = vector_scale([], 2)
        self.assertEqual(c, [])
        self.assertEqual(stats["global_transactions"], 0)
        for block in (0, 1025):
            with self.assertRaises(ValueError):
                vector_scale([], 2, block)


class TestDebug(unittest.TestCase):
    def test_full_block(self):
        self.assertEqual(vector_add_debug([1, 2, 3, 4], [10] * 4)[0], [11, 12, 13, 14])

    def test_tail(self):
        c, stats = vector_add_debug([1, 2, 3, 4, 5], [10] * 5)
        self.assertEqual(c, [11, 12, 13, 14, 15])
        self.assertEqual(stats["global_stores"], 5, "检查 i==n 是否仍然访问。")

    def test_empty(self):
        self.assertEqual(vector_add_debug([], [])[0], [])


if __name__ == "__main__":
    unittest.main()
