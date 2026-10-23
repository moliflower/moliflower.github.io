import copy
import random
import unittest
from project2_scheduler import plan_step, simulate


class TestPolicy(unittest.TestCase):
    def test_port_and_chunk_limit(self):
        active = [("A", 5, 3), ("B", 0, 2), ("C", 2, 1)]
        self.assertEqual(plan_step(active, 4, 2), [("B", "decode", 1), ("A", "prefill", 2), ("C", "prefill", 1)])
        self.assertEqual(plan_step([("A", 1, 2)], 4, 4), [("A", "prefill", 1)])
        self.assertEqual(plan_step([("A", 9, 2)], 4, 2), [("A", "prefill", 2)])

    def test_validation_and_purity(self):
        for active in (None, (), [("A", -1, 1)], [("A", 0, False)], [("A", 0, 1), ("A", 2, 1)]):
            with self.assertRaises(ValueError):
                plan_step(active, 1, 1)
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                plan_step([], value, 1)
            with self.assertRaises(ValueError):
                plan_step([], 1, value)
        active = [["A", 0, 1], ["B", 2, 1]]
        before = copy.deepcopy(active)
        plan_step(active, 2, 2)
        self.assertEqual(active, before)


class TestSimulation(unittest.TestCase):
    def test_hand_trace_and_first_token_from_prefill(self):
        result = simulate([("A", 0, 3, 2), ("B", 1, 1, 1), ("C", 1, 1, 2)], 2, 3, 2)
        self.assertEqual(result["requests"], {
            "A": {"arrival": 0, "token_times": [2, 3]},
            "B": {"arrival": 1, "token_times": [2]},
            "C": {"arrival": 1, "token_times": [3, 4]},
        })
        self.assertEqual(result["trace"], [
            {"start": 0, "end": 1, "admitted": ["A"], "plan": [("A", "prefill", 2)],
             "emitted": [], "finished": [], "active": {"A": (1, 2)}, "waiting": []},
            {"start": 1, "end": 2, "admitted": ["B"], "plan": [("A", "prefill", 1), ("B", "prefill", 1)],
             "emitted": ["A", "B"], "finished": ["B"], "active": {"A": (0, 1)}, "waiting": ["C"]},
            {"start": 2, "end": 3, "admitted": ["C"], "plan": [("A", "decode", 1), ("C", "prefill", 1)],
             "emitted": ["A", "C"], "finished": ["A"], "active": {"C": (0, 1)}, "waiting": []},
            {"start": 3, "end": 4, "admitted": [], "plan": [("C", "decode", 1)],
             "emitted": ["C"], "finished": ["C"], "active": {}, "waiting": []},
        ])

    def test_idle_jump_stable_arrivals_and_fifo_refill(self):
        requests = [("late", 10**9, 1, 1), ("B", 2, 1, 2), ("A", 2, 1, 1)]
        result = simulate(requests, 1, 2, 2)
        self.assertEqual([step["start"] for step in result["trace"]], [2, 3, 4, 10**9])
        self.assertEqual([step["admitted"] for step in result["trace"]], [["B"], [], ["A"], ["late"]])
        self.assertEqual(result["requests"]["late"]["token_times"], [10**9 + 1])
        self.assertEqual(simulate([], 1, 1, 1), {"trace": [], "requests": {}})

    def test_invalid_full_input(self):
        for record in (("A", 0, 1, 1), ("", 0, 1, 1), ("B", -1, 1, 1), ("B", True, 1, 1),
                       ("B", 0, 0, 1), ("B", 0, 1, 0), ("B", 0, 1, True), ("B", 0, 1.5, 1), None):
            with self.subTest(record=record), self.assertRaises(ValueError):
                simulate([("A", 0, 1, 1), record], 1, 1, 1)
        with self.assertRaises(ValueError):
            simulate(None, 1, 1, 1)
        with self.assertRaises(ValueError):
            simulate([], 2, 1, 1)
        for value in (0, -1, True, 1.5):
            for args in ((value, 2, 1), (1, value, 1), (1, 2, value)):
                with self.assertRaises(ValueError):
                    simulate([], *args)

    def test_input_output_and_snapshot_isolation(self):
        requests = [["A", 0, 3, 3], ["B", 0, 2, 1]]
        before = copy.deepcopy(requests)
        result = simulate(requests, 1, 2, 2)
        self.assertEqual(requests, before)
        baseline = copy.deepcopy(result)
        result["trace"][0]["active"]["A"] = (99, 99)
        result["trace"][0]["waiting"].clear()
        result["requests"]["A"]["token_times"].append(99)
        self.assertEqual(result["trace"][1:], baseline["trace"][1:])
        self.assertEqual(simulate(requests, 1, 2, 2), baseline)

    def test_seeded_work_conservation_and_lifecycle(self):
        generator = random.Random(206)
        for trial in range(40):
            requests = [(str(index), generator.randrange(5), generator.randrange(1, 8), generator.randrange(1, 5))
                        for index in range(generator.randrange(1, 7))]
            result = simulate(requests, 2, 3, 2)
            consumed = {record[0]: {"prefill": 0, "decode": 0} for record in requests}
            admitted, finished = set(), set()
            for step in result["trace"]:
                self.assertLessEqual(sum(item[2] for item in step["plan"]), 3)
                self.assertEqual(len({item[0] for item in step["plan"]}), len(step["plan"]))
                self.assertFalse(admitted.intersection(step["admitted"]))
                admitted.update(step["admitted"])
                self.assertFalse(finished.intersection(step["finished"]))
                finished.update(step["finished"])
                self.assertLessEqual(len(admitted - finished), 2)
                self.assertEqual(set(step["active"]), admitted - finished)
                self.assertFalse(set(step["active"]).intersection(step["waiting"]))
                for request_id, phase, count in step["plan"]:
                    consumed[request_id][phase] += count
            for request_id, arrival, prompt, output in requests:
                self.assertEqual(consumed[request_id], {"prefill": prompt, "decode": output - 1})
                times = result["requests"][request_id]["token_times"]
                self.assertEqual(len(times), output)
                self.assertGreater(times[0], arrival)
                self.assertTrue(all(left < right for left, right in zip(times, times[1:])))
            self.assertEqual(finished, set(consumed))
