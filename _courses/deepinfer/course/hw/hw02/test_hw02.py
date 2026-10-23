"""Public contract tests. Each class belongs to exactly one HW2 question."""

import unittest

from hw02 import (attention_ai, attention_bytes, attention_flops, bound,
                  kv_cache_bytes, kv_cache_growth, num_pages, page_waste,
                  paged_batch_bytes, paged_kv_bytes, prefix_cache_savings)


class TestFlops(unittest.TestCase):
    def test_full_and_causal(self):
        self.assertEqual(attention_flops(1, 8, 128, 128, 64, False), 33_554_432)
        self.assertEqual(attention_flops(1, 8, 128, 128, 64, True), 16_908_288)

    def test_rectangular_diagonal_and_empty(self):
        self.assertEqual(attention_flops(2, 3, 2, 5, 7, False), 1680)
        self.assertEqual(attention_flops(2, 3, 1, 1, 7), 168)
        self.assertEqual(attention_flops(1, 1, 0, 0, 8), 0)

    def test_invalid(self):
        for args in ((1, 1, 2, 3, 4), (-1, 1, 2, 2, 4)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                attention_flops(*args)


class TestTraffic(unittest.TestCase):
    def test_bytes(self):
        self.assertEqual(attention_bytes(1, 8, 128, 128, 64, materialized=True), 1_572_864)
        self.assertEqual(attention_bytes(1, 8, 128, 128, 64, materialized=False), 524_288)

    def test_rectangular_traffic_ledger(self):
        self.assertEqual(attention_bytes(2, 3, 2, 5, 7, 4, False), 2352)
        self.assertEqual(attention_bytes(2, 3, 2, 5, 7, 4, True), 3312)

    def test_zero_and_invalid(self):
        self.assertEqual(attention_bytes(0, 3, 2, 5, 7), 0)
        self.assertEqual(attention_bytes(2, 3, 2, 5, 7, 0), 0)
        for index in range(6):
            args = [2, 3, 2, 5, 7, 4]
            args[index] = -1
            with self.subTest(index=index), self.assertRaises(ValueError):
                attention_bytes(*args)


class TestAI(unittest.TestCase):
    def test_flags_and_rectangular_inputs(self):
        self.assertAlmostEqual(attention_ai(1, 8, 128, 128, 64, causal=False), 64 / 3)
        self.assertEqual(attention_ai(1, 8, 128, 128, 64, causal=False, materialized=False), 64)
        self.assertAlmostEqual(attention_ai(2, 3, 2, 5, 7, 4, False, True), 1680 / 3312)
        self.assertAlmostEqual(attention_ai(1, 1, 3, 3, 2, 2, True, True), 48 / 120)

    def test_zero_bytes_and_invalid_causal_shape(self):
        for args in ((0, 1, 3, 3, 2), (1, 1, 0, 0, 2), (1, 1, 2, 3, 2)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                attention_ai(*args)


class TestBound(unittest.TestCase):
    def test_units_and_tie(self):
        self.assertEqual(bound(42.67, 100, 1000), "memory")
        self.assertEqual(bound(200, 100, 1000), "compute")
        self.assertEqual(bound(100, 100, 1000), "compute")
        self.assertEqual(bound(0, 100, 1000), "memory")

    def test_invalid(self):
        for args in ((10, 0, 1000), (10, 100, -1), (-1, 100, 1000)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                bound(*args)


class TestKV(unittest.TestCase):
    def test_kv_bytes(self):
        self.assertEqual(kv_cache_bytes(32, 8, 128, 1024, 2, 2), 268_435_456)
        self.assertEqual(kv_cache_bytes(2, 3, 5, 7, 11, 4), 18_480)

    def test_zero_and_invalid(self):
        self.assertEqual(kv_cache_bytes(2, 3, 5, 0), 0)
        for index in range(6):
            args = [2, 3, 5, 7, 11, 4]
            args[index] = -1
            with self.subTest(index=index), self.assertRaises(ValueError):
                kv_cache_bytes(*args)


class TestGrowth(unittest.TestCase):
    def test_growth_is_increment_per_sequence(self):
        self.assertEqual(kv_cache_growth(32, 8, 128, 100), 13_107_200)
        self.assertEqual(kv_cache_growth(2, 3, 5, 7, 11, 4), 18_480)
        self.assertEqual(kv_cache_growth(2, 3, 5, 0), 0)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            kv_cache_growth(2, 3, 5, -1)


class TestPages(unittest.TestCase):
    def test_page_boundaries(self):
        for length, expected in ((0, 0), (1, 1), (16, 1), (17, 2), (1000, 63)):
            with self.subTest(length=length):
                self.assertEqual(num_pages(length, 16), expected)

    def test_large_integer_and_invalid(self):
        self.assertEqual(num_pages(2**60 + 1, 16), 2**56 + 1)
        for args in ((10, 0), (10, -1), (-1, 16)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                num_pages(*args)


class TestWaste(unittest.TestCase):
    def test_waste_boundaries(self):
        for length, expected in ((0, 0), (1, 15), (16, 0), (17, 15), (1000, 8)):
            with self.subTest(length=length):
                self.assertEqual(page_waste(length, 16), expected)

    def test_invalid(self):
        for args in ((10, 0), (-1, 16)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                page_waste(*args)


class TestPagedBytes(unittest.TestCase):
    def test_last_page_and_batch(self):
        self.assertEqual(paged_kv_bytes(32, 8, 128, 1000, 16), 132_120_576)
        self.assertEqual(paged_kv_bytes(1, 1, 1, 1, 16, batch=2), 128)
        self.assertEqual(paged_kv_bytes(1, 1, 1, 0, 16), 0)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            paged_kv_bytes(1, 1, 1, 10, 0)
        with self.assertRaises(ValueError):
            paged_kv_bytes(1, 1, 1, 10, 16, batch=-1)


class TestPrefixSavings(unittest.TestCase):
    def test_new_consumers_and_existing_copy(self):
        self.assertEqual(prefix_cache_savings(32, 8, 128, 512), 67_108_864)
        self.assertEqual(prefix_cache_savings(32, 8, 128, 512, batch=3), 201_326_592)
        self.assertEqual(prefix_cache_savings(1, 1, 1, 35, batch=2), 280)
        self.assertEqual(prefix_cache_savings(1, 1, 1, 35, batch=0), 0)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            prefix_cache_savings(1, 1, 1, -1)


class TestRaggedBatch(unittest.TestCase):
    def test_independent_last_pages(self):
        lengths = [1, 16, 17]
        self.assertEqual(paged_batch_bytes(lengths, 16, 1, 1, 1), 256)
        self.assertEqual(lengths, [1, 16, 17])

    def test_reordering_zeros_and_empty(self):
        self.assertEqual(paged_batch_bytes([17, 0, 1, 16], 16, 2, 3, 5, 4), 15_360)
        self.assertEqual(paged_batch_bytes([], 16, 2, 3, 5), 0)
        self.assertEqual(paged_batch_bytes([0, 0], 16, 2, 3, 5), 0)

    def test_equal_lengths_agree_with_uniform_batch(self):
        self.assertEqual(paged_batch_bytes([17, 17, 17], 16, 2, 3, 5, 4), 23_040)

    def test_invalid_even_for_empty_batch(self):
        for args in (([-1], 16, 1, 1, 1), ([], 0, 1, 1, 1), ([], 16, -1, 1, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                paged_batch_bytes(*args)


if __name__ == "__main__":
    unittest.main()
