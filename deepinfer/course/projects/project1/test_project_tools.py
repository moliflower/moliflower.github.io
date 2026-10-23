"""Tests of supplied tooling: these run before student kernels are finished."""

import contextlib
import io
import math
from types import SimpleNamespace
import unittest

from experiment import main, make_inputs, run_case, validate_result
from memory import MatrixMemory


class TestMemory(unittest.TestCase):
    def test_invalid_addresses_do_not_count(self):
        memory = MatrixMemory([[1]], [[2]])
        for operation in (lambda: memory.load_a(-1, 0), lambda: memory.load_b(1, 0),
                          lambda: memory.store_c(0, 1, 5), lambda: memory.load_a(True, 0)):
            with self.assertRaises(IndexError):
                operation()
        self.assertEqual(memory.stats(), {"a_reads": 0, "b_reads": 0, "c_writes": 0, "requested_bytes": 0})

    def test_stats_snapshot_and_storage(self):
        memory = MatrixMemory([[1]], [[2]])
        old = memory.stats()
        memory.store_c(0, 0, memory.load_a(0, 0)*memory.load_b(0, 0))
        self.assertEqual(old["requested_bytes"], 0)
        self.assertEqual(memory.output, [[2.0]])
        self.assertEqual(memory.stats()["requested_bytes"], 12)


class TestExperiment(unittest.TestCase):
    def test_seed_reproduces_inputs(self):
        self.assertEqual(make_inputs(3, 61), make_inputs(3, 61))
        self.assertNotEqual(make_inputs(3, 61), make_inputs(3, 62))

    def test_nonfinite_output_and_wrong_component_counts(self):
        correct = {"a_reads": 1, "b_reads": 1, "c_writes": 1, "requested_bytes": 12}
        with self.assertRaises(ValueError):
            validate_result([[math.nan]], correct, [[1]], 1, 1)
        wrong = {**correct, "a_reads": 2, "b_reads": 0}
        with self.assertRaises(ValueError):
            validate_result([[1]], wrong, [[1]], 1, 1)

    def test_timed_repetitions_are_also_validated(self):
        calls = []

        def profile(a, b, tile):
            calls.append(None)
            value = a[0][0]*b[0][0] if len(calls) == 1 else 1e20
            return [[value]], {"a_reads": 1, "b_reads": 1, "c_writes": 1, "requested_bytes": 12}

        with self.assertRaisesRegex(ValueError, "numerical mismatch"):
            run_case(SimpleNamespace(matmul_tiled_profile=profile), 1, 1,
                     seed=61, repeats=2, warmups=1, bandwidth_gbps=1000, peak_tflops=100)
        self.assertEqual(len(calls), 2)

    def test_invalid_cli_rates(self):
        for args in (["--bandwidth-gbps", "nan"], ["--peak-tflops", "0"], ["--warmups", "-1"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(args)
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
