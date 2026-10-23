import importlib.util
import math
from pathlib import Path
import re
import sys
import unittest

sys.dont_write_bytecode = True
COURSE = Path(__file__).resolve().parents[1] / "deepinfer" / "course"
SPECIFICATION = importlib.util.spec_from_file_location("textbook_lab02", COURSE / "labs/lab02/lab02_solution.py")
REFERENCE = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(REFERENCE)


def partial_state(scores, values):
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    return maximum, sum(weights), sum(weight * value for weight, value in zip(weights, values))


def merge_states(states):
    maximum = max(state[0] for state in states)
    normalizer = sum(state[1] * math.exp(state[0] - maximum) for state in states)
    numerator = sum(state[2] * math.exp(state[0] - maximum) for state in states)
    return maximum, normalizer, numerator


class TestAttentionTextbook(unittest.TestCase):
    def assert_matrix_close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for actual_row, expected_row in zip(actual, expected):
            self.assertEqual(len(actual_row), len(expected_row))
            for actual_value, expected_value in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual_value, expected_value)

    def test_nonuniform_full_attention_calculation(self):
        queries = [[1], [2], [1]]
        keys = [[math.log(2)], [math.log(4)], [math.log(8)]]
        values = [[2], [6], [10]]
        scores = REFERENCE.matmul(queries, REFERENCE.transpose(keys))
        self.assert_matrix_close(scores, [
            [math.log(2), math.log(4), math.log(8)],
            [math.log(4), math.log(16), math.log(64)],
            [math.log(2), math.log(4), math.log(8)],
        ])
        probabilities = REFERENCE.softmax_rows(REFERENCE.causal_mask(scores))
        self.assert_matrix_close(probabilities, [[1, 0, 0], [1 / 5, 4 / 5, 0], [1 / 7, 2 / 7, 4 / 7]])
        self.assert_matrix_close(REFERENCE.attention(queries, keys, values), [[2], [26 / 5], [54 / 7]])

    def test_mask_after_softmax_counterexample(self):
        full = REFERENCE.softmax_rows([[math.log(4), math.log(16), math.log(64)]])[0]
        masked_after = [full[0], full[1], 0]
        self.assertAlmostEqual(sum(masked_after), 5 / 21)
        wrong_output = sum(weight * value for weight, value in zip(masked_after, [2, 6, 10]))
        self.assertAlmostEqual(wrong_output, 26 / 21)
        self.assertLess(wrong_output, 2)

    def test_future_value_and_changed_query_exercises(self):
        keys = [[math.log(2)], [math.log(4)], [math.log(8)]]
        queries = [[1], [2], [1]]
        self.assert_matrix_close(REFERENCE.attention(queries, keys, [[2], [6], [100]]), [[2], [26 / 5], [414 / 7]])
        self.assert_matrix_close(REFERENCE.attention([[1], [1], [1]], keys, [[2], [6], [10]]), [[2], [14 / 3], [54 / 7]])

    def test_two_distinct_heads(self):
        output = REFERENCE.multi_head_attention([[0, 0], [1, 1]], [[0, 0], [1, 2]], [[2, 10], [6, 30]], 2)
        self.assert_matrix_close(output, [[2, 10], [(2 + 6 * math.e) / (1 + math.e), (10 + 30 * math.exp(2)) / (1 + math.exp(2))]])
        self.assertAlmostEqual(output[1][0], 4.924234, places=6)
        self.assertAlmostEqual(output[1][1], 27.615942, places=6)

    def test_flops_and_lab_io_examples(self):
        self.assertEqual(REFERENCE.attention_flops(1, 1, 3, 3, 2, False), 72)
        self.assertEqual(REFERENCE.attention_flops(1, 1, 3, 3, 2, True), 48)
        self.assertEqual(REFERENCE.attention_flops(1, 8, 128, 128, 64, False), 33_554_432)
        self.assertEqual(REFERENCE.attention_flops(1, 8, 128, 128, 64, True), 16_908_288)
        self.assertEqual(REFERENCE.attention_bytes(1, 1, 2, 2, 1, 2), 48)
        self.assertEqual(REFERENCE.attention_bytes(1, 1, 2, 2, 1, 2, False), 16)
        self.assertEqual(REFERENCE.attention_bytes(1, 1, 2, 2, 1, 4), 96)
        self.assertEqual(REFERENCE.attention_bytes(1, 1, 2, 3, 4, 2), 128)
        self.assertEqual(REFERENCE.attention_bytes(1, 8, 128, 128, 64), 1_572_864)
        self.assertEqual(REFERENCE.attention_bytes(1, 8, 128, 128, 64, 2, False), 524_288)

    def test_capacity_traffic_and_q_major_rereads(self):
        sequence, head_dim, element_bytes = 4096, 64, 2
        score_capacity = sequence**2 * element_bytes
        self.assertEqual(score_capacity, 32 * 2**20)
        self.assertEqual(4 * score_capacity, 128 * 2**20)
        self.assertEqual(REFERENCE.attention_bytes(1, 1, sequence, sequence, head_dim), 130 * 2**20)
        sequence, head_dim, query_tile = 128, 64, 32
        reread_elements = 2 * sequence * head_dim * (1 + math.ceil(sequence / query_tile))
        self.assertEqual(reread_elements, 81_920)
        self.assertEqual(reread_elements * element_bytes, 163_840)
        self.assertEqual(REFERENCE.attention_bytes(1, 1, sequence, sequence, head_dim, 2, False), 65_536)
        self.assertEqual(REFERENCE.attention_bytes(1, 1, sequence, sequence, head_dim, 2, True), 196_608)

    def test_two_tiles_and_wrong_numerator_rescale(self):
        first = partial_state([0, math.log(2)], [2, 6])
        second = partial_state([math.log(4)], [10])
        self.assertEqual(first, (math.log(2), 1.5, 7))
        merged = merge_states([first, second])
        self.assertEqual(merged, (math.log(4), 1.75, 13.5))
        self.assertAlmostEqual(merged[2] / merged[1], 54 / 7)
        self.assertAlmostEqual((first[2] + second[2]) / merged[1], 68 / 7)
        singles = [partial_state([score], [value]) for score, value in zip([0, math.log(2), math.log(4)], [2, 6, 10])]
        self.assertEqual(merge_states(singles), merged)
        changed = merge_states([first, partial_state([math.log(4)], [14])])
        self.assertEqual(changed[2] / changed[1], 10)

    def test_split_weighting_counterexample(self):
        first = partial_state([0], [0])
        second = partial_state([0, 0, 0], [4, 4, 4])
        self.assertEqual(first, (0, 1, 0))
        self.assertEqual(second, (0, 3, 12))
        merged = merge_states([first, second])
        self.assertEqual(merged[2] / merged[1], 3)
        wrong = (first[2] / first[1] + second[2] / second[1]) / 2
        self.assertEqual(wrong, 2)

    def test_standalone_python_examples_are_runnable(self):
        for lecture in ("lec05.md", "lec06.md"):
            document = (COURSE / "lectures" / lecture).read_text(encoding="utf-8")
            snippets = re.findall(r"```python\n(.*?)\n```", document, flags=re.DOTALL)
            self.assertTrue(snippets)
            for snippet in snippets:
                with self.subTest(lecture=lecture):
                    exec(compile(snippet, lecture, "exec"), {})


if __name__ == "__main__":
    unittest.main()
