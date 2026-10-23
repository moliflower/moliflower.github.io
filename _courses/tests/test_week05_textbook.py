import importlib.util
from pathlib import Path
import sys
import unittest


sys.dont_write_bytecode = True
COURSE = Path(__file__).resolve().parents[1] / "deepinfer" / "course"


def reference(name, relative_path):
    specification = importlib.util.spec_from_file_location(name, COURSE / relative_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


blocks = reference("week05_blocks", "labs/lab04/lab04_solution.py")
prefixes = reference("week05_prefixes", "hw/hw05/hw05_solution.py")
engine = reference("week05_engine", "projects/project2/project2_solution.py")


class Week05Textbook(unittest.TestCase):
    def test_address_table_and_allocated_vs_written_boundary(self):
        expected = [(9, 0), (9, 3), (2, 0), (2, 1), (7, 3)]
        actual = [blocks.locate([9, 2, 7], position, 4) for position in [0, 3, 4, 5, 11]]
        self.assertEqual(actual, expected)
        self.assertEqual(blocks.locate([9, 2, 7], 10, 4), (7, 2))
        with self.assertRaises(ValueError):
            blocks.locate([9, 2, 7], 12, 4)
        self.assertEqual(blocks.locate([5, 1], 5, 4), (1, 1))

    def test_allocation_failure_leaves_capacity_available(self):
        pool = blocks.BlockPool(3)
        self.assertEqual(pool.allocate(2), [0, 1])
        with self.assertRaises(MemoryError):
            pool.allocate(2)
        self.assertEqual(pool.refs, [1, 1, 0])
        self.assertEqual(pool.allocate(1), [2])
        self.assertEqual(blocks.BlockPool(0).allocate(0), [])

    def test_full_shared_lifecycle(self):
        pool = blocks.BlockPool(3)
        first = pool.allocate(2)
        pool.retain([0])
        self.assertEqual(pool.refs, [2, 1, 0])
        second_tail = pool.allocate(1)
        self.assertEqual(pool.refs, [2, 1, 1])
        self.assertEqual(sum(pool.refs), 4)
        self.assertEqual(len(pool.refs) - pool.available(), 3)
        pool.release(first)
        self.assertEqual(pool.refs, [1, 0, 1])
        pool.release([0] + second_tail)
        self.assertEqual(pool.refs, [0, 0, 0])

    def test_atomic_release_and_self_check(self):
        pool = blocks.BlockPool(2)
        pool.allocate(1)
        with self.assertRaises(ValueError):
            pool.release([0, 1])
        self.assertEqual(pool.refs, [1, 0])
        pool = blocks.BlockPool(3)
        pool.allocate(2)
        pool.retain([0])
        pool.release([0, 1])
        self.assertEqual(pool.refs, [1, 0, 0])
        self.assertEqual(pool.allocate(2), [1, 2])

    def test_prefix_examples_and_context_counterexample(self):
        self.assertEqual(prefixes.prefix_hit_tokens([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 9], 4), 4)
        self.assertEqual(prefixes.prefix_hit_tokens([1, 2, 3], [1, 2, 3], 2), 2)
        self.assertEqual(prefixes.prefix_hit_tokens([9, 2, 3, 4], [1, 2, 3, 4], 2), 0)
        self.assertEqual(prefixes.prefix_hit_tokens([], [1], 2), 0)
        with self.assertRaises(ValueError):
            prefixes.prefix_hit_tokens([True], [], 2)

    def test_sharing_tables_and_independent_rounding(self):
        lengths = [5, 8, 9]
        tails = [length - 4 for length in lengths]
        tail_slots = [(length + 3) // 4 * 4 for length in tails]
        self.assertEqual(tails, [1, 4, 5])
        self.assertEqual(tail_slots, [4, 4, 8])
        self.assertEqual([slots - length for slots, length in zip(tail_slots, tails)], [3, 0, 3])
        self.assertEqual(prefixes.shared_prefix_slots(lengths, 4, 4), 20)
        self.assertEqual(prefixes.shared_prefix_slots(lengths, 0, 4), 28)
        self.assertEqual(prefixes.shared_prefix_slots([1, 1], 0, 4), 8)
        self.assertEqual(prefixes.shared_prefix_slots([4, 4], 4, 4), 4)
        self.assertEqual(prefixes.shared_prefix_slots([], 4, 4), 0)
        self.assertEqual(prefixes.shared_prefix_slots([2**60 + 1], 0, 16), 2**60 + 16)
        self.assertEqual(prefixes.shared_prefix_slots([6, 7], 4, 4), 12)

    def test_aligned_prefix_savings_identity(self):
        for block_size in [1, 2, 4, 16]:
            for prefix in [0, block_size, 3 * block_size]:
                for lengths in [[prefix], [prefix + 1, prefix + 3], [prefix, prefix + 1, prefix + 2 * block_size]]:
                    baseline = prefixes.shared_prefix_slots(lengths, 0, block_size)
                    shared = prefixes.shared_prefix_slots(lengths, prefix, block_size)
                    self.assertEqual(baseline - shared, (len(lengths) - 1) * prefix)

    def test_eviction_eligibility_and_ties(self):
        self.assertEqual(prefixes.eviction_order([1, 0, 0, 0], [0, 9, 3, 3]), [2, 3, 1])
        self.assertEqual(prefixes.eviction_order([0, 2, 0], [7, 0, 7]), [0, 2])
        self.assertEqual(prefixes.eviction_order([1, 2], [0, 1]), [])

    def test_fifo_examples_and_blocked_tail_validation(self):
        waiting = [("A", 5), ("B", 1)]
        self.assertEqual(engine.admit_fifo(waiting, 1, 4), ([], 1))
        self.assertEqual(engine.admit_fifo(waiting, 2, 4), (["A"], 0))
        self.assertEqual(engine.admit_fifo(waiting, 3, 4), (["A", "B"], 0))
        with self.assertRaises(ValueError):
            engine.admit_fifo([("A", 5), ("B", 0)], 0, 4)

    def test_six_event_trace_and_snapshot_ownership(self):
        events = [("submit", "A", 5), ("submit", "B", 5), ("submit", "C", 1), ("finish", "A"), ("finish", "B"), ("finish", "C")]
        expected = [
            {"active": {"A": 2}, "waiting": [], "free_blocks": 1},
            {"active": {"A": 2}, "waiting": ["B"], "free_blocks": 1},
            {"active": {"A": 2}, "waiting": ["B", "C"], "free_blocks": 1},
            {"active": {"B": 2, "C": 1}, "waiting": [], "free_blocks": 0},
            {"active": {"C": 1}, "waiting": [], "free_blocks": 2},
            {"active": {}, "waiting": [], "free_blocks": 3},
        ]
        trace = engine.replay(events, 3, 4)
        self.assertEqual(trace, expected)
        for snapshot in trace:
            self.assertEqual(sum(snapshot["active"].values()) + snapshot["free_blocks"], 3)
            self.assertFalse(set(snapshot["active"]) & set(snapshot["waiting"]))
        trace[0]["active"].clear()
        trace[0]["waiting"].append("not-real")
        self.assertEqual(trace[1], expected[1])


if __name__ == "__main__":
    unittest.main()
