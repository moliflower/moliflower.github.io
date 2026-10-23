import copy
import random
import unittest
from lab05 import decode_first, plan_step


class TestDecode(unittest.TestCase):
    def test_ready_order_and_budget(self):
        active = [("P", 5, 2), ("D", 0, 3), ("E", 0, 1)]
        self.assertEqual(decode_first(active, 1), ["D"])
        self.assertEqual(decode_first(active, 3), ["D", "E"])
        self.assertEqual(decode_first([], 1), [])

    def test_invalid_tail_is_not_hidden_by_budget(self):
        for record in (("D", 0, 1), ("", 0, 1), ("X", -1, 1), ("X", True, 1),
                       ("X", 0, 0), ("X", 0, True), ("X", 0.5, 1), ("X", 0, 1.5), None, ("X", 0)):
            active = [("D", 0, 1), record]
            before = copy.deepcopy(active)
            with self.subTest(record=record), self.assertRaises(ValueError):
                decode_first(active, 1)
            self.assertEqual(active, before)

    def test_invalid_container_budget_and_purity(self):
        for active in (None, (), "abc"):
            with self.assertRaises(ValueError):
                decode_first(active, 2)
        for budget in (0, -1, True, 2.0, None):
            with self.assertRaises(ValueError):
                decode_first([], budget)
        active = [["D", 0, 2]]
        result = decode_first(active, 1)
        result.append("Z")
        self.assertEqual(active, [["D", 0, 2]])


class TestPlan(unittest.TestCase):
    def test_decode_priority_and_partial_chunk(self):
        active = [("A", 5, 3), ("B", 0, 2), ("C", 2, 1)]
        self.assertEqual(plan_step(active, 4, 2), [("B", "decode", 1), ("A", "prefill", 2), ("C", "prefill", 1)])
        self.assertEqual(plan_step(active, 1, 2), [("B", "decode", 1)])

    def test_no_same_tick_decode_and_no_second_chunk(self):
        self.assertEqual(plan_step([("A", 1, 3)], 8, 2), [("A", "prefill", 1)])
        self.assertEqual(plan_step([("A", 9, 3)], 8, 2), [("A", "prefill", 2)])
        self.assertEqual(plan_step([], 1, 1), [])

    def test_invalid_parameters_and_tail(self):
        for value in (0, -1, True, 1.5, None):
            with self.assertRaises(ValueError):
                plan_step([], value, 1)
            with self.assertRaises(ValueError):
                plan_step([], 1, value)
        with self.assertRaises(ValueError):
            plan_step([("A", 0, 1), ("B", 0, 0)], 1, 1)

    def test_seeded_exact_policy_and_preserved_inputs(self):
        generator = random.Random(106)
        for trial in range(100):
            active = [(str(index), generator.randrange(6), generator.randrange(1, 5))
                      for index in range(generator.randrange(8))]
            budget, chunk = generator.randrange(1, 9), generator.randrange(1, 6)
            before = copy.deepcopy(active)
            expected = []
            used = 0
            for phase in ("decode", "prefill"):
                for request_id, prefill, output in active:
                    if used == budget or (prefill == 0) != (phase == "decode"):
                        continue
                    count = 1 if phase == "decode" else min(prefill, chunk, budget - used)
                    expected.append((request_id, phase, count))
                    used += count
            self.assertEqual(plan_step(active, budget, chunk), expected, f"trial {trial}")
            self.assertEqual(active, before)
