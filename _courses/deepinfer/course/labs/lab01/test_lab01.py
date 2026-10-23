"""Autograder for Lab 1. Run with: python -m unittest -v"""

from __future__ import annotations

import math
import unittest

from lab01 import (
    layernorm,
    online_softmax_state,
    rmsnorm,
    rmsnorm_traffic,
    softmax,
    softmax_traffic,
    sum_reduce,
)


class TestReduce(unittest.TestCase):
    def test_even(self) -> None:
        self.assertEqual(sum_reduce([1, 2, 3, 4]), 10)

    def test_odd(self) -> None:
        self.assertEqual(sum_reduce([1, 2, 3]), 6)

    def test_empty(self) -> None:
        self.assertEqual(sum_reduce([]), 0.0)

    def test_floats(self) -> None:
        self.assertAlmostEqual(sum_reduce([0.1, 0.2, 0.3]), 0.6)

    def test_pairwise_order(self):
        # Adjacent binary tree gives (1e16+1)+(-1e16+1) == 0.
        # Left fold gives 1; compensated sum gives 2. Order is the contract.
        values = [1e16, 1.0, -1e16, 1.0]
        self.assertEqual(sum_reduce(values), 0.0)
        self.assertEqual(values, [1e16, 1.0, -1e16, 1.0])


class TestSoftmax(unittest.TestCase):
    def test_nonuniform_and_shift(self):
        expected = [1 / (1 + math.e), math.e / (1 + math.e)]
        for x in ([0, 1], [1000, 1001], [-1001, -1000]):
            actual = softmax(x)
            self.assertEqual(len(actual), len(expected))
            for got, want in zip(actual, expected):
                self.assertAlmostEqual(got, want)

    def test_uniform(self) -> None:
        self.assertEqual(softmax([0.0, 0.0]), [0.5, 0.5])

    def test_large_values(self) -> None:
        y = softmax([1000.0, 1000.0])
        self.assertEqual(y, [0.5, 0.5])
        self.assertTrue(all(math.isfinite(v) for v in y))

    def test_sums_to_one(self) -> None:
        y = softmax([1.0, 2.0, 3.0, -4.0])
        self.assertAlmostEqual(sum(y), 1.0)

    def test_empty(self) -> None:
        self.assertEqual(softmax([]), [])

    def test_online_state(self) -> None:
        m, s = online_softmax_state([1.0, 2.0, 3.0])
        self.assertEqual(m, 3.0)
        self.assertAlmostEqual(s, math.exp(-2) + math.exp(-1) + 1.0)

    def test_online_state_empty(self) -> None:
        m, s = online_softmax_state([])
        self.assertEqual(m, -math.inf)
        self.assertEqual(s, 0.0)

    def test_online_state_prefix_invariant(self):
        values = [-1000.0, -1001.0, 0.0, -2.0, 3.0, 3.0]
        for end in range(1, len(values) + 1):
            prefix = values[:end]
            before = prefix[:]
            m, s = online_softmax_state(prefix)
            self.assertEqual(prefix, before)
            self.assertEqual(m, max(prefix))
            self.assertAlmostEqual(s, math.fsum(math.exp(v - m) for v in prefix))

    def test_softmax_extreme_gap_and_copy(self):
        x = [1000.0, -1000.0, 1000.0]
        y = softmax(x)
        self.assertEqual(y, [0.5, 0.0, 0.5])
        self.assertEqual(x, [1000.0, -1000.0, 1000.0])
        self.assertIsNot(y, x)


class TestNorm(unittest.TestCase):
    def test_rmsnorm(self) -> None:
        y = rmsnorm([3.0, 4.0], [1.0, 1.0])
        scale = 1.0 / math.sqrt(12.5 + 1e-6)
        self.assertAlmostEqual(y[0], 3.0 * scale)
        self.assertAlmostEqual(y[1], 4.0 * scale)

    def test_rmsnorm_weight(self) -> None:
        y = rmsnorm([1.0, 1.0], [2.0, 3.0])
        scale = 1.0 / math.sqrt(1.0 + 1e-6)
        self.assertAlmostEqual(y[0], 2.0 * scale)
        self.assertAlmostEqual(y[1], 3.0 * scale)

    def test_rmsnorm_empty(self) -> None:
        self.assertEqual(rmsnorm([], []), [])

    def test_rmsnorm_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            rmsnorm([1.0, 2.0], [1.0])

    def test_layernorm(self) -> None:
        y = layernorm([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
        scale = 1.0 / math.sqrt(2.0 / 3.0 + 1e-5)
        self.assertAlmostEqual(y[0], -1.0 * scale)
        self.assertAlmostEqual(y[1], 0.0)
        self.assertAlmostEqual(y[2], 1.0 * scale)

    def test_layernorm_bias(self) -> None:
        y = layernorm([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [10.0, 10.0, 10.0])
        self.assertAlmostEqual(y[1], 10.0)

    def test_rmsnorm_zero_eps_and_copy(self):
        x, w = [0.0, 0.0], [2.0, -3.0]
        self.assertEqual(rmsnorm(x, w), [0.0, 0.0])
        self.assertEqual((x, w), ([0.0, 0.0], [2.0, -3.0]))
        for eps in (0, -1, math.inf, math.nan):
            with self.subTest(eps=eps), self.assertRaises(ValueError):
                rmsnorm([], [], eps)
        # Epsilon belongs inside sqrt; this case makes the distinction large.
        self.assertEqual(rmsnorm([0.0, 2.0], [1.0, 2.0], 2.0), [0.0, 2.0])

    def test_layernorm_contract(self):
        x, w, bias = [5.0, 5.0], [2.0, -1.0], [3.0, -4.0]
        self.assertEqual(layernorm(x, w, bias), bias)
        self.assertEqual((x, w, bias), ([5.0, 5.0], [2.0, -1.0], [3.0, -4.0]))
        self.assertEqual(layernorm([], [], []), [])
        for args in (([1], [], [1]), ([1], [1], [])):
            with self.assertRaises(ValueError):
                layernorm(*args)
        for eps in (0, -1, math.inf, math.nan):
            with self.subTest(eps=eps), self.assertRaises(ValueError):
                layernorm([], [], [], eps)


class TestTraffic(unittest.TestCase):
    def test_softmax_traffic(self) -> None:
        self.assertEqual(softmax_traffic(1024, 3), 16 * 1024)
        self.assertEqual(softmax_traffic(1024, 2), 12 * 1024)
        self.assertEqual(softmax_traffic(1024, 1), 8 * 1024)

    def test_softmax_traffic_invalid(self) -> None:
        with self.assertRaises(ValueError):
            softmax_traffic(1024, 4)

    def test_rmsnorm_traffic(self) -> None:
        self.assertEqual(rmsnorm_traffic(1024, on_chip=True), 8 * 1024)
        self.assertEqual(rmsnorm_traffic(1024, on_chip=False), 12 * 1024)

    def test_softmax_traffic_boundaries(self):
        self.assertEqual(softmax_traffic(0), 0)
        for n in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                softmax_traffic(n)
        for passes in (0, -1, 2.0, True):
            with self.assertRaises(ValueError):
                softmax_traffic(3, passes)

    def test_rmsnorm_traffic_boundaries(self):
        self.assertEqual(rmsnorm_traffic(0), 0)
        for n in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                rmsnorm_traffic(n)


if __name__ == "__main__":
    unittest.main()
