"""Autograder for Project 1. Run with: python -m unittest -v"""

from __future__ import annotations

import math
import copy
from collections import Counter
import unittest
from unittest.mock import patch

import project1 as kernels
from memory import MatrixMemory

from project1 import (
    estimate_time,
    matmul_naive,
    matmul_tiled,
    matmul_traffic,
    matmul_tiled_profile,
    rmsnorm,
    softmax,
    sum_reduce,
    vector_add,
)


class TestPhase1(unittest.TestCase):
    def test_vector_add(self) -> None:
        self.assertEqual(vector_add([1, 2, 3], [4, 5, 6]), [5.0, 7.0, 9.0])

    def test_vector_add_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            vector_add([1], [1, 2])

    def test_vector_add_contract(self):
        a, b = [1.5, -2.0, 0], [-0.5, 3.0, 0]
        self.assertEqual(vector_add(a, b), [1.0, 1.0, 0.0])
        self.assertEqual(vector_add([], []), [])
        self.assertEqual(a, [1.5, -2.0, 0])
        self.assertEqual(b, [-0.5, 3.0, 0])
        self.assertIsNot(vector_add(a, [0, 0, 0]), a)

    def test_sum_reduce(self) -> None:
        self.assertEqual(sum_reduce([1, 2, 3, 4, 5]), 15.0)
        self.assertEqual(sum_reduce([]), 0.0)

    def test_sum_reduce_tree_order(self):
        # A left fold yields 1; fsum yields 2; adjacent pairs yield 0.
        values = [1e16, 1.0, -1e16, 1.0]
        self.assertEqual(sum_reduce(values), 0.0)
        self.assertEqual(values, [1e16, 1.0, -1e16, 1.0])
        for length in (1, 3, 7, 17):
            self.assertEqual(sum_reduce([1.0] * length), float(length))

    def test_softmax(self) -> None:
        y = softmax([0.0, 0.0])
        self.assertEqual(y, [0.5, 0.5])
        self.assertAlmostEqual(sum(softmax([1.0, 2.0, 3.0])), 1.0)

    def test_softmax_large(self) -> None:
        y = softmax([1000.0, 1000.0])
        self.assertTrue(all(math.isfinite(v) for v in y))
        self.assertEqual(y, [0.5, 0.5])

    def test_softmax_contract(self):
        x = [-2.0, 0.0, 1.0]
        expected = [math.exp(v-1) / math.fsum(math.exp(w-1) for w in x) for v in x]
        for shift in (0, 1000, -1000):
            out = softmax([v + shift for v in x])
            self.assertEqual(len(out), len(x))
            for actual, target in zip(out, expected):
                self.assertAlmostEqual(actual, target, places=12)
        self.assertEqual(softmax([]), [])
        self.assertEqual(softmax([77]), [1.0])
        softmax(x)
        self.assertEqual(x, [-2.0, 0.0, 1.0])

    def test_rmsnorm(self) -> None:
        y = rmsnorm([3.0, 4.0], [1.0, 1.0])
        scale = 1.0 / math.sqrt(12.5 + 1e-6)
        self.assertAlmostEqual(y[0], 3.0 * scale)
        self.assertAlmostEqual(y[1], 4.0 * scale)

    def test_rmsnorm_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            rmsnorm([1.0], [1.0, 2.0])

    def test_rmsnorm_contract(self):
        x, weight = [-3.0, 0.0, 4.0], [2.0, -1.0, 0.5]
        scale = math.sqrt(25.0/3 + 0.25)
        out = rmsnorm(x, weight, 0.25)
        for actual, target in zip(out, [-6.0/scale, 0.0, 2.0/scale]):
            self.assertAlmostEqual(actual, target, places=12)
        self.assertEqual(x, [-3.0, 0.0, 4.0])
        self.assertEqual(weight, [2.0, -1.0, 0.5])
        self.assertEqual(rmsnorm([], []), [])
        self.assertEqual(rmsnorm([0, 0], [1, 2]), [0.0, 0.0])
        for eps in (0, -1, math.nan, math.inf):
            with self.subTest(eps=eps), self.assertRaises(ValueError):
                rmsnorm([], [], eps)

    def test_matmul_naive(self) -> None:
        a = [[1, 2, 3], [4, 5, 6]]
        b = [[7, 8], [9, 10], [11, 12]]
        self.assertEqual(matmul_naive(a, b), [[58.0, 64.0], [139.0, 154.0]])

    def test_matmul_inner_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            matmul_naive([[1, 2]], [[1, 2]])

    def test_matmul_naive_contract(self):
        for a, b in (([], []), ([[]], [[]]), ([[1]], []),
                     ([[1, 2], [3]], [[1], [2]]), ([[1]], [[1, 2], [3]]),
                     (([1],), [[1]]), ([[1]], [1])):
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                matmul_naive(a, b)
        a, b = [[1.5, -2], [0, 3]], [[2, 0.5], [-1, 2]]
        originals = copy.deepcopy((a, b))
        out = matmul_naive(a, b)
        self.assertEqual(out, [[5.0, -3.25], [-3.0, 6.0]])
        self.assertEqual((a, b), originals)
        out[0][0] = 99
        self.assertEqual(out[1], [-3.0, 6.0])


class TestPhase2(unittest.TestCase):
    def setUp(self) -> None:
        self.a = [[1, 2, 3], [4, 5, 6]]
        self.b = [[7, 8], [9, 10], [11, 12]]
        self.expected = [[58.0, 64.0], [139.0, 154.0]]

    def test_tiled_small(self) -> None:
        for tile in (1, 2, 3, 16):
            self.assertEqual(matmul_tiled(self.a, self.b, tile), self.expected)

    def test_tiled_rectangular(self) -> None:
        a = [[1, 2], [3, 4], [5, 6]]
        b = [[1, 2, 3, 4], [5, 6, 7, 8]]
        expected = [[11.0, 14.0, 17.0, 20.0], [23.0, 30.0, 37.0, 44.0], [35.0, 46.0, 57.0, 68.0]]
        self.assertEqual(matmul_tiled(a, b, tile=1), expected)
        self.assertEqual(matmul_tiled(a, b, tile=2), expected)

    def test_tiled_odd_sizes(self) -> None:
        a = [[float(i + j) for j in range(5)] for i in range(3)]
        b = [[float(i - j) for j in range(4)] for i in range(5)]
        self.assertEqual(matmul_tiled(a, b, tile=2), matmul_naive(a, b))

    def test_tiled_contract(self):
        for tile in (0, -2, 1.5, True):
            with self.subTest(tile=tile), self.assertRaises(ValueError):
                matmul_tiled(self.a, self.b, tile)
        for a, b in (([], []), ([[1]], []), ([[]], [[1]]),
                     ([[1, 2], [3]], [[1], [2]]), ([[1, 2]], [[1]])):
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                matmul_tiled(a, b)
        originals = copy.deepcopy((self.a, self.b))
        matmul_tiled(self.a, self.b, 2)
        self.assertEqual((self.a, self.b), originals)

    def test_traffic(self) -> None:
        self.assertEqual(matmul_traffic(8, tile=2, dtype_bytes=4), (4352, 2304))

    def test_traffic_tile_helps(self) -> None:
        naive, tiled = matmul_traffic(64, tile=4)
        self.assertLess(tiled, naive)
        self.assertGreater(naive / tiled, 3.5)

    def test_traffic_partial_tiles(self):
        self.assertEqual(matmul_traffic(3, 16), (252, 108))
        self.assertEqual(matmul_traffic(3, 2), (252, 180))
        self.assertEqual(matmul_traffic(0, 16), (0, 0))

    def test_traffic_invalid(self) -> None:
        for args in ((8, 0, 4), (-1, 2, 4), (1.5, 2, 4), (True, 2, 4),
                     (8, 2.5, 4), (8, True, 4), (8, 2, 0), (8, 2, 1.5)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                matmul_traffic(*args)

    def test_traffic_monotonic_and_dtype(self):
        previous = math.inf
        for tile in range(1, 10):
            naive, tiled = matmul_traffic(5, tile)
            self.assertLessEqual(tiled, previous)
            self.assertEqual(matmul_traffic(5, tile, 2), (naive//2, tiled//2))
            previous = tiled
        self.assertEqual(matmul_traffic(5, 1)[0], matmul_traffic(5, 1)[1])
        self.assertEqual(matmul_traffic(5, 8)[1], 3*25*4)


class TestPhase3(unittest.TestCase):
    def test_compute_bound(self) -> None:
        seconds, bound = estimate_time(1_000_000_000, 1_000_000_000_000, 100, 10)
        self.assertEqual(bound, "compute")
        self.assertAlmostEqual(seconds, 0.1)

    def test_memory_bound(self) -> None:
        seconds, bound = estimate_time(1_000_000_000, 10_000_000_000, 100, 10)
        self.assertEqual(bound, "memory")
        self.assertAlmostEqual(seconds, 0.01)

    def test_invalid(self) -> None:
        for args in ((1, 1, 0, 10), (1, 1, 10, -1), (-1, 1, 1, 1),
                     (1, -1, 1, 1), (math.nan, 1, 1, 1),
                     (1, math.inf, 1, 1), (1, 1, math.inf, 1), (1, 1, 1, math.nan)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                estimate_time(*args)

    def test_zero_and_tie(self):
        self.assertEqual(estimate_time(0, 0), (0.0, "memory"))
        self.assertEqual(estimate_time(1e9, 1e12, 1000, 1000), (0.001, "memory"))


class TestProfile(unittest.TestCase):
    def test_rectangular_counter(self):
        a = [[1, 2, 3], [4, 5, 6]]
        b = [[7, 8], [9, 10], [11, 12]]
        out, stats = matmul_tiled_profile(a, b, 2)
        self.assertEqual(out, [[58, 64], [139, 154]])
        self.assertEqual(stats, {"a_reads": 6, "b_reads": 6, "c_writes": 4, "requested_bytes": 64})

    def test_trace_agrees_with_model(self):
        for n in (1, 3, 5):
            a = [[i-j for j in range(n)] for i in range(n)]
            b = [[2*i+j for j in range(n)] for i in range(n)]
            for tile in (1, 2, 16):
                with self.subTest(n=n, tile=tile):
                    out, stats = matmul_tiled_profile(a, b, tile)
                    self.assertEqual(out, matmul_naive(a, b))
                    self.assertEqual(stats["requested_bytes"], matmul_traffic(n, tile)[1])
                    self.assertEqual(stats["c_writes"], n*n)

    def test_invalid(self):
        for a, b, tile in (([], [[1]], 1), ([[1]], [[1]], 0), ([[1, 2]], [[1]], 1),
                           ([[1]], [[1]], True), ([[1]], [[1]], 1.5),
                           ([[1], []], [[1]], 2), ([[1]], [[]], 2)):
            with self.subTest(a=a, b=b, tile=tile), self.assertRaises(ValueError):
                matmul_tiled_profile(a, b, tile)

    def test_independent_oracle_and_rectangular_tails(self):
        # All three dimensions have tails, and the oracle is not student Q5.
        m, k, n, tile = 3, 5, 7, 2
        a = [[(i*3-p)/7 for p in range(k)] for i in range(m)]
        b = [[(p-j*2)/9 for j in range(n)] for p in range(k)]
        originals = copy.deepcopy((a, b))
        out, stats = matmul_tiled_profile(a, b, tile)
        self.assertEqual(len(out), m)
        for i in range(m):
            self.assertEqual(len(out[i]), n)
            for j in range(n):
                target = math.fsum(a[i][p]*b[p][j] for p in range(k))
                self.assertTrue(math.isclose(out[i][j], target, rel_tol=1e-10, abs_tol=1e-12))
        self.assertEqual(stats, {"a_reads": 60, "b_reads": 70, "c_writes": 21, "requested_bytes": 604})
        self.assertEqual((a, b), originals)
        out[0][0] = 999
        self.assertNotEqual(out[1][0], 999)

    def test_real_memory_boundary(self):
        # Instrument the supplied abstraction; a correct product plus made-up
        # statistics cannot pass. A shifted load checks that returned values
        # actually reach the multiplication, not just a decorative trace.
        instances = []

        class ObservedMemory(MatrixMemory):
            def __init__(self, a, b):
                super().__init__(a, b)
                self.addresses = Counter()
                instances.append(self)

            def load_a(self, i, p):
                self.addresses[("a", i, p)] += 1
                return super().load_a(i, p) + 1.0

            def load_b(self, p, j):
                self.addresses[("b", p, j)] += 1
                return super().load_b(p, j)

            def store_c(self, i, j, value):
                self.addresses[("c", i, j)] += 1
                super().store_c(i, j, value)

        a, b = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        with patch.object(kernels, "MatrixMemory", ObservedMemory):
            out, stats = matmul_tiled_profile(a, b, 2)
        self.assertEqual(len(instances), 1, "construct exactly one MatrixMemory per product")
        memory = instances[0]
        self.assertEqual(out, [[sum((a[i][p]+1)*b[p][j] for p in range(3)) for j in range(3)] for i in range(3)])
        self.assertEqual(stats, memory.stats())
        for i in range(3):
            for j in range(3):
                self.assertEqual(memory.addresses[("a", i, j)], 2)
                self.assertEqual(memory.addresses[("b", i, j)], 2)
                self.assertEqual(memory.addresses[("c", i, j)], 1)


if __name__ == "__main__":
    unittest.main()
