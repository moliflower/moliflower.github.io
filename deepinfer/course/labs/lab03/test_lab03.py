"""Autograder for Lab 3. Run with: python -m unittest -v"""

from __future__ import annotations

import unittest

from lab03 import KVCache, PagedKVCache


class TestContinuous(unittest.TestCase):
    def test_append_input_copy_and_atomicity(self):
        cache = KVCache(2)
        k, v = [1, 2], [3, 4]
        cache.append(k, v)
        k[0], v[0] = 99, 99
        self.assertEqual((cache.k, cache.v), ([[1.0, 2.0]], [[3.0, 4.0]]))
        for k, v in (([1], [2, 3]), ([1, 2], [3])):
            with self.assertRaises(ValueError):
                cache.append(k, v)
            self.assertEqual(len(cache), 1)
            self.assertEqual((cache.k, cache.v), ([[1.0, 2.0]], [[3.0, 4.0]]))

    def test_get_range_contract(self):
        cache = KVCache(1)
        cache.append([2], [3])
        for start, end in ((1, 0), (-1, 0), (2, 2), (0.0, 1), (False, 1)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                cache.get(start, end)
        self.assertEqual(cache.get(1, 1), ([], []))
        k, v = cache.get(0, 1)
        k.clear()
        v.append([9])
        self.assertEqual(cache.get(0, 1), ([[2]], [[3]]))

    def test_append_get(self) -> None:
        cache = KVCache(2)
        self.assertEqual(cache.append([1, 2], [3, 4]), 0)
        self.assertEqual(cache.append([5, 6], [7, 8]), 1)
        k, v = cache.get(0, 2)
        self.assertEqual(k, [[1.0, 2.0], [5.0, 6.0]])
        self.assertEqual(v, [[3.0, 4.0], [7.0, 8.0]])
        self.assertEqual(len(cache), 2)

    def test_copy_semantics(self) -> None:
        cache = KVCache(2)
        cache.append([1, 2], [3, 4])
        k, v = cache.get(0, 1)
        k[0][0] = 99.0
        v[0][0] = 99.0
        k2, v2 = cache.get(0, 1)
        self.assertEqual(k2, [[1.0, 2.0]])
        self.assertEqual(v2, [[3.0, 4.0]])

    def test_append_invalid(self) -> None:
        cache = KVCache(2)
        with self.assertRaises(ValueError):
            cache.append([1], [2, 3])

    def test_get_invalid(self) -> None:
        cache = KVCache(2)
        cache.append([1, 2], [3, 4])
        with self.assertRaises(ValueError):
            cache.get(0, 2)
        with self.assertRaises(ValueError):
            cache.get(-1, 1)

    def test_free(self) -> None:
        cache = KVCache(2)
        cache.append([1, 2], [3, 4])
        cache.free()
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.get(0, 0), ([], []))


class TestPaged(unittest.TestCase):
    def test_append_boundary_and_atomicity(self):
        cache = PagedKVCache(2)
        for i in range(2):
            self.assertEqual(cache.append([i], [-i]), i)
        self.assertEqual((cache.allocated_slots(), cache.waste()), (2, 0))
        before_ids = cache.block_table[:]
        with self.assertRaises(ValueError):
            cache.append([2, 3], [4])
        self.assertEqual((len(cache), cache.allocated_slots(), cache.waste()), (2, 2, 0))
        self.assertEqual(cache.block_table, before_ids)
        self.assertEqual(cache.append([2], [-2]), 2)
        self.assertEqual((cache.allocated_slots(), cache.waste()), (4, 1))

    def test_get_all_ranges(self):
        # Compare gather against a small independent list oracle at every cut.
        for block_size in (1, 2, 4):
            cache = PagedKVCache(block_size)
            for i in range(7):
                cache.append([i, i + 0.5], [-i, -i - 0.5])
            for start in range(8):
                for end in range(start, 8):
                    with self.subTest(page=block_size, start=start, end=end):
                        self.assertEqual(cache.get(start, end),
                            ([[i, i + 0.5] for i in range(start, end)],
                             [[-i, -i - 0.5] for i in range(start, end)]))
            for start, end in ((2, 1), (-1, 1), (8, 8), (0.0, 1), (False, 1)):
                with self.assertRaises(ValueError):
                    cache.get(start, end)

    def test_physical_ids_are_not_logical_indices(self):
        cache = PagedKVCache(2)
        for i in range(5):
            cache.append([i], [-i])
        old = cache.blocks
        ids = [9, 2, 7]
        cache.blocks = {new: old[logical] for logical, new in enumerate(ids)}
        cache.block_table = ids
        self.assertEqual(cache.get(1, 5), ([[1], [2], [3], [4]], [[-1], [-2], [-3], [-4]]))

    def test_copies_on_append_and_get(self):
        cache = PagedKVCache(2)
        k, v = [1], [2]
        cache.append(k, v)
        k[0], v[0] = 99, 99
        out_k, out_v = cache.get(0, 1)
        self.assertEqual((out_k, out_v), ([[1]], [[2]]))
        out_k[0][0] = 77
        self.assertEqual(cache.get(0, 1), ([[1]], [[2]]))

    def test_block_allocation(self) -> None:
        cache = PagedKVCache(4)
        for i in range(10):
            cache.append([i, i], [-i, -i])
        self.assertEqual(len(cache), 10)
        self.assertEqual(len(cache.block_table), 3)
        self.assertEqual(cache.allocated_slots(), 12)
        self.assertEqual(cache.waste(), 2)

    def test_gather_across_blocks(self) -> None:
        cache = PagedKVCache(4)
        for i in range(10):
            cache.append([i, i], [-i, -i])
        k, v = cache.get(2, 7)
        self.assertEqual([row[0] for row in k], [2, 3, 4, 5, 6])
        self.assertEqual([row[0] for row in v], [-2, -3, -4, -5, -6])

    def test_empty_range(self) -> None:
        cache = PagedKVCache(4)
        self.assertEqual(cache.get(0, 0), ([], []))
        cache.append([1, 1], [2, 2])
        self.assertEqual(cache.get(1, 1), ([], []))

    def test_invalid_range(self) -> None:
        cache = PagedKVCache(4)
        cache.append([1, 1], [2, 2])
        with self.assertRaises(ValueError):
            cache.get(0, 2)

    def test_free(self) -> None:
        cache = PagedKVCache(4)
        for i in range(5):
            cache.append([i], [i])
        cache.free()
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.block_table, [])
        self.assertEqual(cache.allocated_slots(), 0)

    def test_new_block_id_after_free(self) -> None:
        cache = PagedKVCache(2)
        cache.append([1], [1])
        first = cache.block_table[0]
        cache.free()
        cache.append([2], [2])
        self.assertEqual(cache.block_table[0], 0)
        self.assertNotEqual(cache.blocks[0][0][0], [1.0])


if __name__ == "__main__":
    unittest.main()
