"""Public contract tests. Each class belongs to exactly one HW1 question."""

import unittest

from hw01 import (arithmetic_intensity, attention_flops, dequantize_per_tensor,
                  kv_cache_bytes, max_kv_tokens, mla_kv_cache_bytes, quant_error,
                  quantize_per_tensor, roofline_time)


class TestIntensity(unittest.TestCase):
    def test_units_and_fraction(self):
        self.assertEqual(arithmetic_intensity(1_000_000_000, 100_000_000), 10.0)
        self.assertEqual(arithmetic_intensity(3, 2), 1.5)
        self.assertEqual(arithmetic_intensity(0, 2), 0.0)

    def test_invalid(self):
        for flops, moved in ((10, 0), (10, -1), (-1, 10)):
            with self.subTest(flops=flops, moved=moved), self.assertRaises(ValueError):
                arithmetic_intensity(flops, moved)


class TestRoofline(unittest.TestCase):
    def test_compute_bound(self):
        seconds, bound = roofline_time(1e9, 1e12, 100, 10)
        self.assertEqual(bound, "compute")
        self.assertAlmostEqual(seconds, 0.1)

    def test_memory_bound(self):
        seconds, bound = roofline_time(1e9, 1e10, 100, 10)
        self.assertEqual(bound, "memory")
        self.assertAlmostEqual(seconds, 0.01)

    def test_tie_and_zero(self):
        self.assertEqual(roofline_time(1e9, 1e11, 100, 10), (0.01, "memory"))
        self.assertEqual(roofline_time(0, 0, 100, 10), (0.0, "memory"))

    def test_invalid(self):
        for args in ((1, 1, 0, 1), (1, 1, 1, -1), (-1, 1, 1, 1), (1, -1, 1, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                roofline_time(*args)


class TestKV(unittest.TestCase):
    def test_kv(self):
        self.assertEqual(kv_cache_bytes(32, 8, 128, 1024, 2, 2), 268_435_456)
        self.assertEqual(kv_cache_bytes(2, 3, 5, 7, 11, 4), 18_480)

    def test_zero_and_invalid(self):
        for index in range(6):
            args = [2, 3, 5, 7, 11, 4]
            args[index] = 0
            self.assertEqual(kv_cache_bytes(*args), 0)
            args[index] = -1
            with self.subTest(index=index), self.assertRaises(ValueError):
                kv_cache_bytes(*args)


class TestMLA(unittest.TestCase):
    def test_shared_latent(self):
        self.assertEqual(mla_kv_cache_bytes(32, 512, 64, 1024, 2, 1), 37_748_736)
        self.assertEqual(mla_kv_cache_bytes(2, 3, 5, 7, 11, 4), 4_928)
        self.assertEqual(mla_kv_cache_bytes(2, 3, 0, 7, 1, 2), 84)

    def test_invalid(self):
        for index in range(6):
            args = [2, 3, 5, 7, 11, 4]
            args[index] = -1
            with self.subTest(index=index), self.assertRaises(ValueError):
                mla_kv_cache_bytes(*args)


class TestAttention(unittest.TestCase):
    def test_full_and_causal(self):
        self.assertEqual(attention_flops(1, 128, 8, 64, False), 33_554_432)
        self.assertEqual(attention_flops(1, 128, 8, 64, True), 16_908_288)

    def test_diagonal_and_empty(self):
        self.assertEqual(attention_flops(2, 1, 3, 5), 120)
        self.assertEqual(attention_flops(2, 0, 3, 5), 0)

    def test_count_pairs_by_rows(self):
        for seq in (2, 3, 7):
            self.assertEqual(attention_flops(2, seq, 3, 5),
                             sum(4 * 2 * 3 * 5 * (row + 1) for row in range(seq)))

    def test_invalid(self):
        for index in range(4):
            args = [2, 3, 5, 7]
            args[index] = -1
            with self.subTest(index=index), self.assertRaises(ValueError):
                attention_flops(*args)


class TestQuantize(unittest.TestCase):
    def test_symmetric_grid(self):
        q, scale = quantize_per_tensor([-1.0, 0.0, 1.0], 8)
        self.assertEqual(q, [-127, 0, 127])
        self.assertAlmostEqual(scale, 1 / 127)
        self.assertTrue(all(type(value) is int for value in q))

    def test_ties_to_even_and_no_mutation(self):
        x = [-3.0, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.0]
        before = x[:]
        q, scale = quantize_per_tensor(x, 3)
        self.assertEqual((q, scale), ([-3, -2, -2, 0, 0, 2, 2, 3], 1.0))
        self.assertEqual(x, before)
        self.assertIsNot(q, x)

    def test_empty_zeros_and_bit_endpoints(self):
        self.assertEqual(quantize_per_tensor([], 8), ([], 1.0))
        self.assertEqual(quantize_per_tensor([0, 0], 8), ([0, 0], 1.0))
        self.assertEqual(quantize_per_tensor([-1, 1], 2), ([-1, 1], 1.0))
        self.assertEqual(quantize_per_tensor([-1, 1], 16)[0], [-32767, 32767])

    def test_invalid_bits(self):
        for bits in (1, 17, 3.5, True):
            for x in ([], [1.0]):
                with self.subTest(bits=bits, x=x), self.assertRaises(ValueError):
                    quantize_per_tensor(x, bits)


class TestDequantize(unittest.TestCase):
    def test_independent_codes(self):
        # Q7 must be runnable before Q6: supply codes directly.
        q = [0, 1, -2, 3]
        result = dequantize_per_tensor(q, 0.25)
        self.assertEqual(result, [0.0, 0.25, -0.5, 0.75])
        self.assertTrue(all(type(value) is float for value in result))
        self.assertEqual(q, [0, 1, -2, 3])
        self.assertIsNot(result, q)

    def test_empty(self):
        self.assertEqual(dequantize_per_tensor([], 1.0), [])


class TestQuantError(unittest.TestCase):
    def test_nonzero_max_error(self):
        self.assertAlmostEqual(quant_error([0.0, 0.2, 1.0], 3), 2 / 15)
        self.assertEqual(quant_error([-1.0, 0.0, 1.0], 8), 0.0)

    def test_error_bound(self):
        for bits in (2, 3, 8, 16):
            x = [0.1, -0.2, 0.3, 7.25]
            self.assertLessEqual(quant_error(x, bits), 7.25 / (2**(bits - 1) - 1) / 2 + 1e-12)

    def test_empty_and_invalid_bits(self):
        self.assertEqual(quant_error([], 8), 0.0)
        with self.assertRaises(ValueError):
            quant_error([], 1)


class TestBudget(unittest.TestCase):
    def test_exact_budget_and_one_byte_short(self):
        self.assertEqual(max_kv_tokens(512 * 2**20, 32, 8, 128), 4096)
        self.assertEqual(max_kv_tokens(512 * 2**20 - 1, 32, 8, 128), 4095)
        self.assertEqual(max_kv_tokens(0, 32, 8, 128), 0)

    def test_maximality_and_large_integer(self):
        for budget in (1, 119, 120, 121, 1000, 2**60 + 123):
            count = max_kv_tokens(budget, 2, 3, 5, 2)
            self.assertIs(type(count), int)
            self.assertLessEqual(120 * count, budget)
            self.assertGreater(120 * (count + 1), budget)

    def test_invalid(self):
        for args in ((-1, 2, 3, 5, 2), (10, 0, 3, 5, 2), (10, 2, 3, 5, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                max_kv_tokens(*args)


if __name__ == "__main__":
    unittest.main()
