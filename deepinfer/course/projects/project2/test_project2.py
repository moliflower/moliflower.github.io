import copy
import unittest
from project2 import admit_fifo, replay


class TestAdmission(unittest.TestCase):
    def test_head_of_line_and_exact_pages(self):
        queue = [("A", 5), ("B", 1)]
        self.assertEqual(admit_fifo(queue, 1, 4), ([], 1))
        self.assertEqual(admit_fifo(queue, 3, 4), (["A", "B"], 0))
        self.assertEqual(admit_fifo(queue, 2, 4), (["A"], 0))
        self.assertEqual(queue, [("A", 5), ("B", 1)])
        self.assertEqual(admit_fifo([], 0, 4), ([], 0))

    def test_validation_includes_blocked_tail(self):
        for args in (([("A", 5), ("B", 0)], 0, 4), ([("A", 1), ("A", 2)], 3, 4), ([], -1, 4), ([], 0, True), ([("", 1)], 2, 4)):
            with self.assertRaises(ValueError):
                admit_fifo(*args)


class TestReplay(unittest.TestCase):
    def test_release_retries_waiting_and_conserves_capacity(self):
        events = [("submit", "A", 5), ("submit", "B", 5), ("submit", "C", 1), ("finish", "A"), ("finish", "B"), ("finish", "C")]
        original = copy.deepcopy(events)
        trace = replay(events, 3, 4)
        self.assertEqual(trace[2], {"active": {"A": 2}, "waiting": ["B", "C"], "free_blocks": 1})
        self.assertEqual(trace[3], {"active": {"B": 2, "C": 1}, "waiting": [], "free_blocks": 0})
        self.assertEqual(trace[-1], {"active": {}, "waiting": [], "free_blocks": 3})
        for row in trace:
            self.assertEqual(sum(row["active"].values()) + row["free_blocks"], 3)
            self.assertFalse(set(row["active"]) & set(row["waiting"]))
        trace[0]["active"].clear()
        trace[0]["waiting"].append("changed")
        self.assertEqual(trace[1]["active"], {"A": 2})
        self.assertEqual(trace[1]["waiting"], ["B"])
        self.assertEqual(events, original)

    def test_reject_unknown_finished_duplicate_and_impossible(self):
        cases = [[("finish", "A")], [("submit", "A", 13)], [("submit", "A", 1), ("submit", "A", 1)],
                 [("submit", "A", 1), ("finish", "A"), ("submit", "A", 1)],
                 [("submit", "A", 9), ("submit", "B", 1), ("finish", "B")], [("bogus",)], [()]]
        for events in cases:
            with self.subTest(events=events), self.assertRaises(ValueError):
                replay(events, 3, 4)
        for capacity, size in ((0, 4), (3, 0), (True, 4), (3, 1.5)):
            with self.assertRaises(ValueError):
                replay([], capacity, size)
        self.assertEqual(replay([], 3, 4), [])
