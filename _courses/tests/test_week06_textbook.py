import copy
import importlib.util
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
COURSE = Path(__file__).resolve().parents[1] / "deepinfer" / "course"


def load_reference(name, relative):
    specification = importlib.util.spec_from_file_location(name, COURSE / relative)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


POLICY = load_reference("textbook_lab05", "labs/lab05/lab05_solution.py")
METRICS = load_reference("textbook_hw06", "hw/hw06/hw06_solution.py")
ENGINE = load_reference("textbook_phase2", "projects/project2_scheduler/project2_scheduler_solution.py")


def token_times(result):
    return {request_id: record["token_times"] for request_id, record in result["requests"].items()}


def fixed_batch_times(requests, max_active, budget, chunk):
    pending = sorted(requests, key=lambda record: record[1])
    waiting = []
    active = {}
    times = {record[0]: [] for record in requests}
    cursor = 0
    now = 0
    while cursor < len(pending) or waiting or active:
        if not waiting and not active:
            now = max(now, pending[cursor][1])
        while cursor < len(pending) and pending[cursor][1] <= now:
            waiting.append(pending[cursor])
            cursor += 1
        if not active:
            for _ in range(min(max_active, len(waiting))):
                request_id, arrival, prompt, output = waiting.pop(0)
                active[request_id] = [prompt, output]
        plan = POLICY.plan_step([(request_id, *state) for request_id, state in active.items()], budget, chunk)
        finished = []
        for request_id, phase, count in plan:
            state = active[request_id]
            if phase == "prefill":
                state[0] -= count
            if state[0] == 0:
                times[request_id].append(now + 1)
                state[1] -= 1
            if state[1] == 0:
                finished.append(request_id)
        for request_id in finished:
            del active[request_id]
        now += 1
    return times


class TestWeek06Textbook(unittest.TestCase):
    def test_single_request_prefill_emits_first_token(self):
        result = ENGINE.simulate([("A", 0, 3, 3)], 1, 2, 2)
        self.assertEqual(token_times(result), {"A": [2, 3, 4]})
        self.assertEqual([step["plan"] for step in result["trace"]], [
            [("A", "prefill", 2)], [("A", "prefill", 1)],
            [("A", "decode", 1)], [("A", "decode", 1)],
        ])
        self.assertEqual(sum(count for step in result["trace"] for _, _, count in step["plan"]), 5)
        singleton = ENGINE.simulate([("A", 0, 3, 1)], 1, 2, 2)
        self.assertEqual(token_times(singleton), {"A": [2]})
        self.assertEqual(len(singleton["trace"]), 2)

    def test_same_workload_fixed_vs_continuous(self):
        requests = [("A", 0, 1, 1), ("B", 0, 1, 4), ("C", 1, 1, 2)]
        continuous = ENGINE.simulate(requests, 2, 2, 1)
        fixed = fixed_batch_times(requests, 2, 2, 1)
        self.assertEqual(fixed, {"A": [1], "B": [1, 2, 3, 4], "C": [5, 6]})
        self.assertEqual(token_times(continuous), {"A": [1], "B": [1, 2, 3, 4], "C": [2, 3]})
        fixed_records = [(arrival, fixed[request_id]) for request_id, arrival, _, _ in requests]
        continuous_records = [(record["arrival"], record["token_times"]) for record in continuous["requests"].values()]
        self.assertEqual(METRICS.aggregate_metrics(fixed_records)["mean_ttft"], 2)
        self.assertEqual(METRICS.aggregate_metrics(continuous_records)["mean_ttft"], 1)
        self.assertEqual(sum(map(len, fixed.values())), 7)
        self.assertEqual(continuous["trace"][-1]["end"], 4)
        delayed = [("A", 0, 1, 1), ("B", 0, 1, 4), ("C", 4, 1, 2)]
        self.assertEqual(fixed_batch_times(delayed, 2, 2, 1), token_times(ENGINE.simulate(delayed, 2, 2, 1)))

    def test_abc_full_state_and_work(self):
        result = ENGINE.simulate([("A", 0, 3, 2), ("B", 1, 1, 1), ("C", 1, 1, 2)], 2, 3, 2)
        self.assertEqual(token_times(result), {"A": [2, 3], "B": [2], "C": [3, 4]})
        self.assertEqual([step["active"] for step in result["trace"]], [
            {"A": (1, 2)}, {"A": (0, 1)}, {"C": (0, 1)}, {},
        ])
        self.assertEqual([step["waiting"] for step in result["trace"]], [[], ["C"], [], []])
        self.assertEqual([sum(count for _, _, count in step["plan"]) for step in result["trace"]], [2, 2, 2, 1])

    def test_budget_example_complete_suffix(self):
        active = {"A": [5, 3], "B": [0, 2], "C": [2, 1]}
        outputs = {request_id: [] for request_id in active}
        states = []
        plans = []
        for now in range(5):
            plan = POLICY.plan_step([(request_id, *state) for request_id, state in active.items()], 4, 2)
            plans.append(plan)
            for request_id, phase, count in plan:
                if phase == "prefill":
                    active[request_id][0] -= count
                if active[request_id][0] == 0:
                    active[request_id][1] -= 1
                    outputs[request_id].append(now + 1)
            active = {request_id: state for request_id, state in active.items() if state[1] > 0}
            states.append(copy.deepcopy(active))
        self.assertEqual(plans[:2], [[("B", "decode", 1), ("A", "prefill", 2), ("C", "prefill", 1)]] * 2)
        self.assertEqual(states, [
            {"A": [3, 3], "B": [0, 1], "C": [1, 1]}, {"A": [1, 3]},
            {"A": [0, 2]}, {"A": [0, 1]}, {},
        ])
        self.assertEqual(outputs, {"A": [3, 4, 5], "B": [1, 2], "C": [2]})

    def test_chunk_comparison_and_exercise(self):
        requests = [("long", 0, 5, 2), ("short", 1, 1, 1)]
        for chunk, expected_long, mean_ttft, duration in ((1, [5, 6], 3, 6), (2, [3, 4], 2, 4), (3, [2, 3], 1.5, 3)):
            with self.subTest(chunk=chunk):
                result = ENGINE.simulate(requests, 2, 3, chunk)
                self.assertEqual(token_times(result), {"long": expected_long, "short": [2]})
                metrics = METRICS.aggregate_metrics([(0, expected_long), (1, [2])])
                self.assertEqual(metrics, {"mean_ttft": mean_ttft, "mean_tpot": 1})
                self.assertEqual(result["trace"][-1]["end"], duration)

    def test_latency_counterexample_and_request_weighting(self):
        self.assertEqual(METRICS.request_metrics(2, [5, 7, 11]), {"ttft": 3, "tpot": 3, "e2e": 9})
        shifted = METRICS.request_metrics(0, [5, 7, 11])
        self.assertEqual(shifted, {"ttft": 5, "tpot": 3, "e2e": 11})
        self.assertNotEqual(shifted["tpot"], shifted["e2e"] / 3)
        self.assertEqual(METRICS.request_metrics(1, [4]), {"ttft": 3, "tpot": None, "e2e": 3})
        records = [(0, [1, 5]), (0, [2, 3, 4, 5]), (1, [2])]
        self.assertEqual(METRICS.aggregate_metrics(records), {"mean_ttft": 4 / 3, "mean_tpot": 2.5})
        self.assertEqual(sum([4, 1, 1, 1]) / 4, 1.75)

    def test_goodput_window_and_units(self):
        records = [(0, [2, 4]), (0, [2, 6]), (1, [2])]
        self.assertEqual(METRICS.slo_goodput(records, 2, 2, 6), 2 / 6)
        self.assertEqual(METRICS.slo_goodput(records, 2, 2, 8), 2 / 8)
        with self.assertRaises(ValueError):
            METRICS.slo_goodput(records, 2, 2, 5)
        converted = [(arrival * 1000, [timestamp * 1000 for timestamp in times]) for arrival, times in records]
        self.assertEqual(METRICS.slo_goodput(converted, 2000, 2000, 6000), 2 / 6000)

    def test_project_steps_idle_fifo_and_refill(self):
        self.assertEqual(token_times(ENGINE.simulate([("A", 0, 1, 1)], 1, 1, 1)), {"A": [1]})
        self.assertEqual(token_times(ENGINE.simulate([("A", 0, 1, 1), ("B", 0, 1, 1)], 1, 1, 1)), {"A": [1], "B": [2]})
        result = ENGINE.simulate([("late", 10**9, 1, 1), ("B", 2, 1, 2), ("A", 2, 1, 1)], 1, 2, 2)
        self.assertEqual([step["start"] for step in result["trace"]], [2, 3, 4, 10**9])
        self.assertEqual([step["admitted"] for step in result["trace"]], [["B"], [], ["A"], ["late"]])


if __name__ == "__main__":
    unittest.main()
