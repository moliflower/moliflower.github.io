import unittest
from hw05 import prefix_hit_tokens, shared_prefix_slots, eviction_order


class TestHit(unittest.TestCase):
    def test_prefix_not_substring_or_partial_block(self):
        self.assertEqual(prefix_hit_tokens([1, 2, 3, 4, 5], [1, 2, 3, 9], 2), 2)
        self.assertEqual(prefix_hit_tokens([9, 2, 3, 4], [1, 2, 3, 4], 2), 0)
        self.assertEqual(prefix_hit_tokens([1, 2, 3], [1, 2, 3], 2), 2)
        self.assertEqual(prefix_hit_tokens([], [1], 2), 0)

    def test_preserve_and_validate(self):
        tokens = [1, 2, 3, 4]
        self.assertEqual(prefix_hit_tokens(tokens, tokens, 2), 4)
        self.assertEqual(tokens, [1, 2, 3, 4])
        for args in (([], [], 0), ([], [], True), ([True], [], 2), ([-1], [], 2), (None, [], 2)):
            with self.assertRaises(ValueError):
                prefix_hit_tokens(*args)


class TestSharing(unittest.TestCase):
    def test_shared_once_independent_tails(self):
        lengths = [5, 8, 9]
        self.assertEqual(shared_prefix_slots(lengths, 4, 4), 20)
        self.assertEqual(shared_prefix_slots(lengths, 0, 4), 28)
        self.assertEqual(shared_prefix_slots([4, 4], 4, 4), 4)
        self.assertEqual(shared_prefix_slots([], 4, 4), 0)
        self.assertEqual(shared_prefix_slots([2**60 + 1], 0, 16), 2**60 + 16)
        self.assertEqual(lengths, [5, 8, 9])

    def test_invalid_even_empty(self):
        for args in (([], 0, 0), ([], 1, 4), ([3], 4, 4), ([True], 0, 4), ([-1], 0, 4), ([], -4, 4)):
            with self.assertRaises(ValueError):
                shared_prefix_slots(*args)


class TestEviction(unittest.TestCase):
    def test_only_unreferenced_lru_with_ties(self):
        refs, times = [1, 0, 0, 0], [0, 9, 3, 3]
        self.assertEqual(eviction_order(refs, times), [2, 3, 1])
        self.assertEqual(refs, [1, 0, 0, 0])
        self.assertEqual(times, [0, 9, 3, 3])
        self.assertEqual(eviction_order([1], [0]), [])
        self.assertEqual(eviction_order([], []), [])

    def test_invalid(self):
        for args in (([0], []), ([-1], [0]), ([0], [-1]), ([True], [0]), ([], None)):
            with self.assertRaises(ValueError):
                eviction_order(*args)
