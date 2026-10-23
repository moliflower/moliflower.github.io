"""预备单元公开测试。每条断言尽量说明输入、期望和排查方向。"""

import unittest

from demo import sum_with_trace
from prep import assert_vector_close, copy_matrix, matmul, maximum, vector_add


class TestRun(unittest.TestCase):
    def test_demo(self):
        total, trace = sum_with_trace([2, -1, 4])
        self.assertEqual(total, 5)
        self.assertEqual(trace, [(0, 2, 0, 2), (1, -1, 2, 1), (2, 4, 1, 5)])

    def test_empty_trace(self):
        self.assertEqual(sum_with_trace([]), (0, []))


class TestMaximum(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(maximum([2, 7, 3]), 7, "输入 [2,7,3]：期望 7")

    def test_negative(self):
        self.assertEqual(maximum([-5, -2, -9]), -2, "全负数返回了什么？检查 best 的初值。")

    def test_single(self):
        self.assertEqual(maximum([-8]), -8)

    def test_empty(self):
        with self.assertRaises(ValueError, msg="空列表没有最大值；规格要求 ValueError"):
            maximum([])

    def test_no_mutation(self):
        values = [2, -7, 3]
        maximum(values)
        self.assertEqual(values, [2, -7, 3])


class TestMatrix(unittest.TestCase):
    def test_copy_values(self):
        self.assertEqual(copy_matrix([[1, 2], [3, 4]]), [[1, 2], [3, 4]])

    def test_copy_independence(self):
        original = [[1, 2], [3, 4]]
        copied = copy_matrix(original)
        copied[0][0] = 99
        self.assertEqual(original[0][0], 1, "改副本影响了原矩阵：每一行也需要复制。")

    def test_empty_copy(self):
        self.assertEqual(copy_matrix([]), [])

    def test_matmul_values(self):
        a, b = [[1, 2], [3, 4]], [[5, 6], [7, 8]]
        self.assertEqual(matmul(a, b), [[19, 22], [43, 50]], "逐格对照；total 是否在每个输出开始前清零？")

    def test_rectangular_and_no_mutation(self):
        a, b = [[1, 2, 3], [4, 5, 6]], [[1], [0], [-1]]
        self.assertEqual(matmul(a, b), [[-2], [-2]])
        self.assertEqual(a, [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(b, [[1], [0], [-1]])

    def test_rows_independent(self):
        out = matmul([[1], [2]], [[3, 4]])
        out[0][0] = 99
        self.assertEqual(out[1][0], 6, "输出行不应该引用同一列表。")

    def test_mismatch(self):
        with self.assertRaises(ValueError):
            matmul([[1, 2]], [[3, 4]])


class TestTesting(unittest.TestCase):
    def test_close_accepts_roundoff(self):
        assert_vector_close([0.1 + 0.2], [0.3])

    def test_close_rejects_wrong_value(self):
        with self.assertRaises(AssertionError):
            assert_vector_close([0.4], [0.3])

    def test_close_rejects_length(self):
        with self.assertRaises(AssertionError):
            assert_vector_close([1, 2], [1])

    def test_exact_and_empty(self):
        assert_vector_close([], [])
        assert_vector_close([1], [1], atol=0)

    def test_negative_tolerance(self):
        with self.assertRaises(ValueError):
            assert_vector_close([1], [1], atol=-1)

    def test_vector_add(self):
        self.assertEqual(vector_add([2, -1], [3, 4]), [5, 3])
        self.assertEqual(vector_add([], []), [])

    def test_vector_mismatch(self):
        with self.assertRaises(ValueError):
            vector_add([1], [])

    def test_vector_no_mutation(self):
        a, b = [1, 2], [3, 4]
        result = vector_add(a, b)
        self.assertEqual(a, [1, 2])
        self.assertEqual(b, [3, 4])
        result[0] = 99
        self.assertEqual(a, [1, 2])


if __name__ == "__main__":
    unittest.main()
