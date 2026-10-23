"""Autograder for Lab 2. Run with: python -m unittest -v"""

from __future__ import annotations

import math
import unittest

from lab02 import (
    attention,
    attention_bytes,
    attention_flops,
    causal_mask,
    matmul,
    multi_head_attention,
    softmax_rows,
    transpose,
)


class TestBasics(unittest.TestCase):
    def test_matmul(self) -> None:
        self.assertEqual(
            matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
            [[19.0, 22.0], [43.0, 50.0]],
        )

    def test_matmul_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            matmul([[1, 2]], [[1, 2]])

    def test_matmul_rectangular_and_copy(self):
        a, b = [[1, 2, 3], [-1, 0, 2]], [[1], [0], [-1]]
        out = matmul(a, b)
        self.assertEqual(out, [[-2.0], [-3.0]])
        out[0][0] = 99
        self.assertEqual(out[1], [-3.0])
        self.assertEqual((a, b), ([[1, 2, 3], [-1, 0, 2]], [[1], [0], [-1]]))
        self.assertEqual(matmul([], b), [])

    def test_transpose(self) -> None:
        self.assertEqual(transpose([[1, 2, 3], [4, 5, 6]]), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    def test_transpose_roundtrip_and_copy(self):
        a = [[1, 2, 3]]
        out = transpose(a)
        self.assertEqual(transpose(out), a)
        out[0][0] = 99
        self.assertEqual(a, [[1, 2, 3]])
        self.assertEqual(transpose([]), [])

    def test_softmax_rows(self) -> None:
        y = softmax_rows([[0.0, 0.0], [1.0, 1.0]])
        self.assertEqual(y, [[0.5, 0.5], [0.5, 0.5]])
        y2 = softmax_rows([[0.0, 1.0]])[0]
        self.assertAlmostEqual(y2[0], 1.0 / (1.0 + math.e))
        self.assertAlmostEqual(y2[1], math.e / (1.0 + math.e))

    def test_causal_mask(self) -> None:
        masked = causal_mask([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        self.assertEqual(masked[0][0], 1.0)
        self.assertEqual(masked[0][1], -math.inf)
        self.assertEqual(masked[0][2], -math.inf)
        self.assertEqual(masked[1][1], 5.0)
        self.assertEqual(masked[1][2], -math.inf)
        self.assertEqual(masked[2][2], 9.0)

    def test_softmax_rows_masked_and_invalid(self):
        rows = [[1000, -math.inf, 1000], [-1000, -1000, -math.inf]]
        self.assertEqual(softmax_rows(rows), [[0.5, 0, 0.5], [0.5, 0.5, 0]])
        self.assertEqual(rows[0], [1000, -math.inf, 1000])
        self.assertEqual(softmax_rows([]), [])
        for row in ([], [-math.inf], [0, math.inf], [0, math.nan]):
            with self.subTest(row=row), self.assertRaises(ValueError):
                softmax_rows([row])

    def test_causal_mask_shape_and_copy(self):
        scores = [[1, 2], [3, 4]]
        got = causal_mask(scores)
        self.assertEqual(got, [[1, -math.inf], [3, 4]])
        got[1][0] = 99
        self.assertEqual(scores, [[1, 2], [3, 4]])
        self.assertEqual(causal_mask([]), [])
        with self.assertRaises(ValueError):
            causal_mask([[1, 2, 3], [4, 5, 6]])


class TestAttention(unittest.TestCase):
    def test_attention_rectangular_and_validation(self):
        self.assertEqual(attention([[0]], [[1], [2]], [[2], [6]], False), [[4.0]])
        self.assertEqual(attention([], [[1]], [[2]]), [])
        for q, k, v in (([[1]], [[1]], []), ([[1]], [[1]], [[2], [3]]),
                        ([[1, 2]], [[1]], [[2]]), ([[]], [[]], [[]])):
            with self.subTest(q=q, k=k, v=v), self.assertRaises(ValueError):
                attention(q, k, v)
        with self.assertRaises(ValueError):
            attention([[0]], [[1], [2]], [[2], [6]])

    def test_multi_head_contract(self):
        for heads in (0, -1, 1.5, True):
            with self.subTest(heads=heads), self.assertRaises(ValueError):
                multi_head_attention([[1]], [[1]], [[1]], heads)
        self.assertEqual(multi_head_attention([], [], [], 2), [])
        for q, k, v in (([[1, 2]], [[1]], [[1, 2]]), ([[1]], [], [[1]]),
                        ([[]], [[]], [[]])):
            with self.assertRaises(ValueError):
                multi_head_attention(q, k, v, 1)

    def test_identity(self) -> None:
        x = [[1.0, 0.0], [0.0, 1.0]]
        out = attention(x, x, x, causal=True)
        self.assertEqual(out[0], [1.0, 0.0])
        # row 1 sees both positions; scores are [0, 1/sqrt(2)].
        p = math.exp(1 / math.sqrt(2))
        self.assertAlmostEqual(out[1][0], 1.0 / (1.0 + p))
        self.assertAlmostEqual(out[1][1], p / (1.0 + p))

    def test_non_causal(self) -> None:
        x = [[1.0, 0.0], [0.0, 1.0]]
        out = attention(x, x, x, causal=False)
        p = math.exp(1 / math.sqrt(2))
        self.assertAlmostEqual(out[0][0], p / (1.0 + p))
        self.assertAlmostEqual(out[0][1], 1.0 / (1.0 + p))
        self.assertAlmostEqual(out[1][0], 1.0 / (1.0 + p))
        self.assertAlmostEqual(out[1][1], p / (1.0 + p))

    def test_multi_head(self) -> None:
        # All rows identical -> all scores are identical -> uniform attention
        # -> output equals the (identical) value row.
        row = [1.0, 2.0, 3.0, 4.0]
        x = [row, row, row]
        out = multi_head_attention(x, x, x, num_heads=2)
        for got in out:
            self.assertEqual(got, row)

    def test_multi_head_invalid(self) -> None:
        with self.assertRaises(ValueError):
            multi_head_attention([[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]], 2)

    def test_multi_head_distinct_subspaces(self):
        # d_head=1: independent logits [0,1] and [0,2] on the second row.
        q = [[0, 0], [1, 1]]
        k = [[0, 0], [1, 2]]
        v = [[2, 10], [6, 30]]
        got = multi_head_attention(q, k, v, 2)
        self.assertEqual(got[0], [2, 10])
        self.assertAlmostEqual(got[1][0], (2 + 6 * math.e) / (1 + math.e))
        self.assertAlmostEqual(got[1][1], (10 + 30 * math.exp(2)) / (1 + math.exp(2)))

    def test_future_values_cannot_change_past(self):
        q, k = [[1], [2], [3]], [[-1], [1], [2]]
        v = [[2], [4], [8]]
        before = attention(q, k, v)
        v[-1] = [999]
        after = attention(q, k, v)
        self.assertEqual(before[:2], after[:2])
        self.assertNotEqual(before[-1], after[-1])


class TestCost(unittest.TestCase):
    def test_flops_boundaries(self):
        self.assertEqual(attention_flops(1, 1, 2, 3, 4, False), 96)
        self.assertEqual(attention_flops(1, 1, 0, 0, 4), 0)
        for index in range(5):
            for bad in (-1, 0.5, True):
                args = [1, 1, 1, 1, 1]
                args[index] = bad
                with self.subTest(args=args), self.assertRaises(ValueError):
                    attention_flops(*args, causal=False)

    def test_bytes_boundaries(self):
        # Q/O: 16, K/V: 24, S/P: 24 elements; 2B each.
        self.assertEqual(attention_bytes(1, 1, 2, 3, 4), 128)
        self.assertEqual(attention_bytes(0, 1, 2, 3, 4), 0)
        for index in range(6):
            args = [1, 1, 1, 1, 1, 2]
            args[index] = -1
            with self.subTest(args=args), self.assertRaises(ValueError):
                attention_bytes(*args)

    def test_flops(self) -> None:
        self.assertEqual(attention_flops(1, 8, 128, 128, 64, causal=False), 33_554_432)
        self.assertEqual(attention_flops(1, 8, 128, 128, 64, causal=True), 16_908_288)

    def test_flops_causal_rectangular(self) -> None:
        with self.assertRaises(ValueError):
            attention_flops(1, 8, 128, 64, 64, causal=True)

    def test_bytes_materialized(self) -> None:
        self.assertEqual(
            attention_bytes(1, 8, 128, 128, 64, bytes_per_elem=2, materialized=True),
            1_572_864,
        )

    def test_bytes_read_once_lower_bound(self) -> None:
        self.assertEqual(
            attention_bytes(1, 8, 128, 128, 64, bytes_per_elem=2, materialized=False),
            524_288,
        )

    def test_causal_diagonal(self):
        self.assertEqual(attention_flops(1, 1, 1, 1, 2), 8)
        self.assertEqual(attention_flops(1, 1, 3, 3, 2), 48)

    def test_io_tiny(self):
        self.assertEqual(attention_bytes(1, 1, 2, 2, 1, 4), 96)
        self.assertEqual(attention_bytes(1, 1, 2, 2, 1, 4, False), 32)


if __name__ == "__main__":
    unittest.main()
