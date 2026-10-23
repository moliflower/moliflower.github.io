import contextlib
import io
import math
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ElementTree


COURSE = Path(__file__).resolve().parents[1] / "deepinfer" / "course"


def python_examples(chapter):
    source = (COURSE / "lectures" / chapter).read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)\n```", source, re.DOTALL)


def run_example(source):
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(source, "textbook-example", "exec"), namespace)
    return namespace, output.getvalue()


def online_states(values):
    maximum = -math.inf
    normalizer = 0.0
    states = []
    for value in values:
        next_maximum = max(maximum, value)
        normalizer = normalizer * math.exp(maximum - next_maximum) + math.exp(value - next_maximum)
        maximum = next_maximum
        states.append((maximum, normalizer))
    return states


def request_counts(rows, depth, cols, tile):
    left_reads = rows * depth * ((cols + tile - 1) // tile)
    right_reads = depth * cols * ((rows + tile - 1) // tile)
    writes = rows * cols
    return left_reads, right_reads, writes


class TestPriorTextbook(unittest.TestCase):
    def test_execution_tail_and_unique_coverage(self):
        for length in (0, 1, 5, 8, 9, 10):
            block = 4
            grid = (length + block - 1) // block
            contiguous = [group * block + thread for group in range(grid) for thread in range(block)]
            strided = [thread * grid + group for group in range(grid) for thread in range(block)]
            self.assertEqual(sorted(index for index in contiguous if index < length), list(range(length)))
            self.assertEqual(sorted(index for index in strided if index < length), list(range(length)))
        self.assertEqual((4, 9, 14), (0 + 4, 5 + 4, 10 + 4))

    def test_execution_table_example_is_runnable(self):
        example = next(source for source in python_examples("lec01.md") if "group_count =" in source)
        namespace, output = run_example(example)
        rows = output.splitlines()
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0], "0 0 0 work")
        self.assertEqual(rows[4], "1 0 4 work")
        self.assertEqual(rows[-1], "1 3 7 skip")
        self.assertEqual(namespace["length"], 5)

    def test_memory_example_is_runnable(self):
        examples = python_examples("lec02.md")
        example = next(source for source in examples if "def address_counts" in source)
        namespace, output = run_example(example)
        counter = namespace["address_counts"]
        self.assertEqual(counter(0, 1, 32), (1, 4))
        self.assertEqual(counter(1, 1, 32), (2, 5))
        self.assertEqual(counter(16, 1, 32), (2, 4))
        self.assertEqual(counter(0, 3, 32), (3, 12))
        self.assertEqual(counter(0, 1, 8), (1, 1))
        self.assertEqual(counter(0, 0, 32), (1, 1))
        self.assertEqual(len(output.splitlines()), 4)

    def test_reduction_work_and_odd_tail(self):
        values = [2, -1, 4, 3, -2, 5, 8]
        states = []
        additions = 0
        while len(values) > 1:
            next_values = []
            for index in range(0, len(values), 2):
                if index + 1 < len(values):
                    next_values.append(values[index] + values[index + 1])
                    additions += 1
                else:
                    next_values.append(values[index])
            values = next_values
            states.append(values)
        self.assertEqual(states, [[1, 7, 3, 8], [8, 11], [19]])
        self.assertEqual(additions, 6)

    def test_float_reduction_example_is_runnable(self):
        example = next(source for source in python_examples("lec03.md") if "paired =" in source)
        namespace, output = run_example(example)
        self.assertEqual(namespace["paired"], 0.0)
        self.assertEqual(namespace["serial"], 1.0)
        self.assertEqual(output.strip(), "0.0 1.0")

    def test_stable_softmax_example_is_runnable(self):
        example = next(source for source in python_examples("lec03.md") if "def stable_probabilities" in source)
        namespace, output = run_example(example)
        probabilities = namespace["stable_probabilities"]
        self.assertEqual(probabilities([1000.0, 1000.0]), [0.5, 0.5])
        self.assertEqual(probabilities([]), [])
        original = [0.0, math.log(2), math.log(4)]
        shifted = [value + 1000 for value in original]
        for actual, expected in zip(probabilities(original), (1 / 7, 2 / 7, 4 / 7)):
            self.assertAlmostEqual(actual, expected, places=12)
        for actual, expected in zip(probabilities(shifted), probabilities(original)):
            self.assertAlmostEqual(actual, expected, places=12)
        self.assertEqual(len(output.splitlines()), 2)

    def test_online_state_tables(self):
        increasing = online_states([0.0, math.log(2), math.log(4)])
        self.assertEqual([round(state[1], 12) for state in increasing], [1.0, 1.5, 1.75])
        decreasing = online_states([math.log(4), 0.0, math.log(2)])
        self.assertEqual([round(state[1], 12) for state in decreasing], [1.0, 1.25, 1.75])
        for values in ([0, -1000, 2], [1000, 1001], [-4, -2, -8], [3, 3, 3]):
            maximum, normalizer = online_states(values)[-1]
            self.assertEqual(maximum, max(values))
            self.assertAlmostEqual(normalizer, math.fsum(math.exp(value - maximum) for value in values))

    def test_norm_hand_calculations(self):
        rms = [value * weight / math.sqrt(12.5 + 0.5) for value, weight in zip([3, 4], [1, 2])]
        self.assertAlmostEqual(rms[0], 0.8320502943378437)
        self.assertAlmostEqual(rms[1], 2.2188007849009166)
        values = [1, 2, 3]
        mean = math.fsum(values) / len(values)
        variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
        normalized = [(value - mean) / math.sqrt(variance + 1 / 3) for value in values]
        self.assertEqual(normalized, [-1.0, 0.0, 1.0])

    def test_activation_traffic_and_roofline(self):
        self.assertEqual([4096 * 4 * (passes + 1) for passes in (3, 2, 1)], [65536, 49152, 32768])
        self.assertAlmostEqual(16384 / 2e12 * 1e9, 8.192)
        self.assertEqual(4 * 8, 32)
        self.assertEqual(4 * 12, 48)
        self.assertEqual(max(2e9 / 1000e9, 100e9 / 100e12), 0.002)
        self.assertEqual(max(0.5e9 / 1000e9, 100e9 / 100e12), 0.001)

    def test_tiled_accumulator_example_is_runnable(self):
        examples = python_examples("lec04.md")
        example = next(source for source in examples if "inner_start in range" in source)
        namespace, output = run_example(example)
        self.assertEqual(namespace["accumulator"], [[58, 64], [139, 154]])
        self.assertEqual(output.splitlines(), ["0 [[25, 28], [73, 82]]", "2 [[58, 64], [139, 154]]"])
        self.assertEqual(namespace["left"], [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(namespace["right"], [[7, 8], [9, 10], [11, 12]])

    def test_tiled_traffic_tables(self):
        self.assertEqual([sum(request_counts(3, 3, 3, tile)) * 4 for tile in (1, 2, 16)], [252, 180, 108])
        self.assertEqual(request_counts(3, 5, 4, 2), (30, 40, 12))
        self.assertEqual(sum(request_counts(3, 5, 4, 2)) * 4, 328)
        self.assertEqual(sum(request_counts(3, 5, 4, 1)) * 4, 528)
        self.assertEqual(sum(request_counts(3, 5, 4, 8)) * 4, 188)
        self.assertEqual(sum(request_counts(1, 3, 2, 2)) * 4, 44)

    def test_migration_dot_products(self):
        left = [2, -1, 3]
        right = [[4, 0], [5, 2], [-1, 1]]
        first = [math.fsum(left[inner] * right[inner][col] for inner in range(2)) for col in range(2)]
        tail = [left[2] * right[2][col] for col in range(2)]
        self.assertEqual(first, [3, -2])
        self.assertEqual(tail, [-3, 3])
        self.assertEqual([earlier + later for earlier, later in zip(first, tail)], [0, 1])

    def test_unit1_figures_are_local_accessible_svg(self):
        figures = [
            "unit1-execution-tail.svg",
            "unit1-memory-sectors.svg",
            "unit1-numerics-online.svg",
            "unit1-tiling-accumulator.svg",
        ]
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        for figure in figures:
            with self.subTest(figure=figure):
                source = (COURSE / "figures" / figure).read_text(encoding="utf-8")
                root = ElementTree.fromstring(source)
                self.assertEqual(root.attrib["role"], "img")
                self.assertIsNotNone(root.find("svg:title", namespace))
                self.assertIsNotNone(root.find("svg:desc", namespace))
                self.assertLessEqual(int(root.attrib["width"]), 640)
                self.assertGreaterEqual(int(root.attrib["width"]), 560)
                self.assertNotIn("<script", source)
                self.assertNotIn("href=", source)
                for size in re.findall(r'font-size="([0-9]+)"', source):
                    self.assertGreaterEqual(int(size), 20)


if __name__ == "__main__":
    unittest.main()
