"""公开规格：从常规输入、边界和性质三方面检查，不要求某种写法。"""

import unittest
import foundations as student


class TestExample(unittest.TestCase):
    def test_returned_value(self):
        self.assertEqual(student.add_example(2, 5), 7)
        self.assertEqual(student.add_example(-2, 2), 0)


class TestExpressions(unittest.TestCase):
    def test_units(self):
        self.assertEqual(student.bytes_for_elements(7, 2), 14)
        self.assertEqual(student.bytes_for_elements(3, 4), 12)

    def test_empty(self):
        self.assertEqual(student.bytes_for_elements(0, 4), 0)

    def test_large_integer_stays_exact(self):
        self.assertEqual(student.bytes_for_elements(2**60 + 1, 4), 2**62 + 4)

    def test_invalid_domain(self):
        for args in [(-1, 2), (1, 0), (1, -2)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                student.bytes_for_elements(*args)


class TestControl(unittest.TestCase):
    def test_exact_and_partial(self):
        for count, size, expected in [(7, 3, 3), (6, 3, 2), (1, 8, 1), (8, 1, 8)]:
            with self.subTest(count=count, size=size):
                self.assertEqual(student.ceil_groups(count, size), expected)

    def test_no_work_needs_no_groups(self):
        self.assertEqual(student.ceil_groups(0, 8), 0)

    def test_large_integer(self):
        self.assertEqual(student.ceil_groups(2**60 + 1, 2), 2**59 + 1)

    def test_coverage_and_minimality(self):
        for count in range(24):
            for size in [1, 3, 8]:
                with self.subTest(count=count, size=size):
                    groups = student.ceil_groups(count, size)
                    self.assertIs(type(groups), int)
                    self.assertGreaterEqual(groups * size, count)
                    if groups:
                        self.assertLess((groups - 1) * size, count)

    def test_invalid_domain(self):
        for args in [(-1, 2), (1, 0), (1, -2)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                student.ceil_groups(*args)


class TestSequences(unittest.TestCase):
    def test_below_equal_above(self):
        self.assertEqual(student.clamp_lengths([2, 5, 8, 0], 5), [2, 5, 5, 0])

    def test_empty_and_zero_limit(self):
        self.assertEqual(student.clamp_lengths([], 3), [])
        self.assertEqual(student.clamp_lengths([0, 2], 0), [0, 0])

    def test_new_list_and_no_mutation(self):
        lengths = [1, 8]
        result = student.clamp_lengths(lengths, 3)
        self.assertEqual(lengths, [1, 8])
        self.assertIsNot(result, lengths)
        result[0] = 99
        self.assertEqual(lengths, [1, 8])

    def test_idempotent(self):
        once = student.clamp_lengths([0, 5, 2, 9, 5], 4)
        self.assertEqual(once, [0, 4, 2, 4, 4])
        self.assertEqual(student.clamp_lengths(once, 4), once)

    def test_invalid_domain(self):
        for lengths, limit in [([-1], 3), ([1], -1)]:
            with self.subTest(lengths=lengths, limit=limit), self.assertRaises(ValueError):
                student.clamp_lengths(lengths, limit)


class TestComposition(unittest.TestCase):
    def test_offsets_include_final_boundary(self):
        self.assertEqual(student.prefix_offsets([2, 0, 3]), [0, 2, 2, 5])
        self.assertEqual(student.prefix_offsets([4]), [0, 4])

    def test_empty_and_zeros(self):
        self.assertEqual(student.prefix_offsets([]), [0])
        self.assertEqual(student.prefix_offsets([0, 0]), [0, 0, 0])

    def test_segment_lengths_and_no_mutation(self):
        lengths = [3, 1, 4, 0, 2]
        offsets = student.prefix_offsets(lengths)
        self.assertEqual(lengths, [3, 1, 4, 0, 2])
        self.assertEqual(len(offsets), len(lengths) + 1)
        self.assertEqual(offsets[0], 0)
        self.assertEqual(offsets[-1], sum(lengths))
        self.assertEqual([offsets[i + 1] - offsets[i] for i in range(len(lengths))], lengths)

    def test_calls_do_not_share_state(self):
        self.assertEqual(student.prefix_offsets([1, 2]), [0, 1, 3])
        self.assertEqual(student.prefix_offsets([5]), [0, 5])

    def test_invalid_domain(self):
        with self.assertRaises(ValueError):
            student.prefix_offsets([1, -1])


if __name__ == "__main__":
    unittest.main()
