import random
import unittest
from lab04 import BlockPool, locate


class TestAllocate(unittest.TestCase):
    def test_empty_and_exhaustion(self):
        pool = BlockPool(3)
        self.assertEqual(pool.allocate(2), [0, 1])
        with self.assertRaises(MemoryError):
            pool.allocate(2)
        self.assertEqual(pool.refs, [1, 1, 0])
        self.assertEqual(pool.allocate(0), [])
        self.assertEqual(pool.allocate(1), [2])
        self.assertEqual(BlockPool(0).allocate(0), [])

    def test_invalid(self):
        for value in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                BlockPool(value)
            with self.assertRaises(ValueError):
                BlockPool(2).allocate(value)


class TestRetain(unittest.TestCase):
    def test_share_and_atomic_validation(self):
        pool = BlockPool(3)
        pool.refs[:] = [1, 1, 0]
        pool.retain([1])
        self.assertEqual(pool.refs, [1, 2, 0])
        for ids in ([0, 0], [0, 2], [-1], [3], [True], [0.0], None):
            with self.assertRaises(ValueError):
                pool.retain(ids)
            self.assertEqual(pool.refs, [1, 2, 0])
        pool.retain([])


class TestRelease(unittest.TestCase):
    def test_shared_lifetime_and_reuse(self):
        pool = BlockPool(3)
        pool.refs[:] = [2, 1, 1]
        pool.release([0, 1])
        self.assertEqual(pool.refs, [1, 0, 1])
        self.assertEqual(pool.allocate(1), [1])
        pool.release([0])
        self.assertEqual(pool.allocate(1), [0])

    def test_invalid_release_is_atomic(self):
        pool = BlockPool(2)
        pool.refs[:] = [1, 0]
        for ids in ([0, 1], [0, 0], [-1], [2], [True], [0.0], None):
            with self.assertRaises(ValueError):
                pool.release(ids)
            self.assertEqual(pool.refs, [1, 0])
        pool.release([])

    def test_seeded_ownership_trace(self):
        generator = random.Random(61)
        pool, owners = BlockPool(7), []
        for step in range(150):
            if owners and generator.random() < 0.55:
                ids = owners.pop(generator.randrange(len(owners)))
                pool.release(ids)
            elif pool.available():
                ids = pool.allocate(1)
                owners.append(ids)
                if generator.random() < 0.5:
                    pool.retain(ids)
                    owners.append(ids[:])
            expected = [sum(block in ids for ids in owners) for block in range(7)]
            self.assertEqual(pool.refs, expected, f"step {step}")
            self.assertEqual(pool.available(), 7 - len({block for ids in owners for block in ids}))


class TestLocate(unittest.TestCase):
    def test_noncontiguous_mapping_and_boundaries(self):
        table = [9, 2, 7]
        self.assertEqual(locate(table, 4, 4), (2, 0))
        self.assertEqual(locate(table, 11, 4), (7, 3))
        self.assertEqual(table, [9, 2, 7])

    def test_invalid(self):
        for args in (([], 0, 4), ([2], -1, 4), ([2], 4, 4), ([2], True, 4), ([2], 0, 0), ([2], 0, True), ([False], 0, 4), ([-1], 0, 4)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                locate(*args)
