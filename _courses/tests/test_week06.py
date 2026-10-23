"""Cross-assignment checks, an independent replay oracle, and mutation checks."""

import contextlib
import copy
import csv
import importlib.util
import io
from pathlib import Path
import random
import subprocess
import sys
import types
import unittest

sys.dont_write_bytecode = True
COURSE = Path(__file__).resolve().parents[1] / "deepinfer" / "course"


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, COURSE / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LAB = load("labs/lab05/lab05_solution.py", "lab_reference")
HW = load("hw/hw06/hw06_solution.py", "hw_reference")
PROJECT = load("projects/project2_scheduler/project2_scheduler_solution.py", "project_reference")


def oracle(requests, slots, budget, chunk):
    ordered = sorted(requests, key=lambda record: record[1])
    prompt_done = {record[0]: 0 for record in requests}
    outputs = {record[0]: [] for record in requests}
    admitted_ids, done_ids, running = set(), set(), []
    trace = []
    for now in range(200):
        candidates = [record for record in ordered if record[1] <= now and record[0] not in admitted_ids]
        newly = candidates[:slots - len(running)]
        running.extend(newly)
        admitted_ids.update(record[0] for record in newly)
        if not running:
            if len(done_ids) == len(requests):
                break
            continue
        ready = [record for record in running if prompt_done[record[0]] == record[2]]
        filling = [record for record in running if prompt_done[record[0]] < record[2]]
        plan, emitted, finished = [], [], []
        available = budget
        for record in ready + filling:
            request_id, arrival, prompt, output = record
            if not available:
                break
            phase = "decode" if record in ready else "prefill"
            count = 1 if phase == "decode" else min(prompt - prompt_done[request_id], chunk, available)
            plan.append((request_id, phase, count))
            available -= count
            if phase == "prefill":
                prompt_done[request_id] += count
            if prompt_done[request_id] == prompt:
                outputs[request_id].append(now + 1)
                emitted.append(request_id)
            if len(outputs[request_id]) == output:
                finished.append(request_id)
                done_ids.add(request_id)
        running = [record for record in running if record[0] not in done_ids]
        trace.append({"start": now, "end": now + 1, "admitted": [record[0] for record in newly],
                      "plan": plan, "emitted": emitted, "finished": finished,
                      "active": {record[0]: (record[2] - prompt_done[record[0]], record[3] - len(outputs[record[0]]))
                                 for record in running},
                      "waiting": [record[0] for record in candidates if record[0] not in admitted_ids]})
    else:
        raise AssertionError("oracle exceeded bounded fixture horizon")
    return {"trace": trace, "requests": {record[0]: {"arrival": record[1], "token_times": outputs[record[0]]}
                                         for record in requests}}


class TestWeekSixIntegration(unittest.TestCase):
    def test_300_independent_replays_and_ported_policy(self):
        generator = random.Random(20261024)
        for trial in range(300):
            slots = generator.randrange(1, 5)
            budget, chunk = generator.randrange(slots, slots + 6), generator.randrange(1, 7)
            requests = [(str(index), generator.randrange(9), generator.randrange(1, 10), generator.randrange(1, 6))
                        for index in range(generator.randrange(9))]
            before = copy.deepcopy(requests)
            expected = oracle(requests, slots, budget, chunk)
            self.assertEqual(PROJECT.simulate(requests, slots, budget, chunk), expected, f"trial {trial}")
            self.assertEqual(requests, before)
            active = [(request_id, arrival, output) for request_id, arrival, prompt, output in requests]
            self.assertEqual(LAB.plan_step(active, budget, chunk), PROJECT.plan_step(active, budget, chunk))

    def test_18_experiment_rows_against_homework(self):
        directory = COURSE / "projects/project2_scheduler"
        output = subprocess.check_output([sys.executable, "-B", str(directory / "experiment.py"), "--solution"], text=True)
        rows = list(csv.DictReader(io.StringIO(output)))
        self.assertEqual(len(rows), 18)
        workloads = load("projects/project2_scheduler/experiment.py", "experiment_fixtures").WORKLOADS
        for row in rows:
            result = PROJECT.simulate(workloads[row["workload"]], int(row["max_active"]), int(row["budget"]), int(row["chunk"]))
            records = [(record["arrival"], record["token_times"]) for record in result["requests"].values()]
            metrics = HW.aggregate_metrics(records)
            duration = result["trace"][-1]["end"]
            self.assertEqual(int(row["duration_ticks"]), duration)
            self.assertAlmostEqual(float(row["mean_ttft_ticks"]), metrics["mean_ttft"], places=5)
            self.assertEqual(float(row["mean_tpot_ticks"]), metrics["mean_tpot"])
            self.assertAlmostEqual(float(row["slo_requests_per_tick"]), HW.slo_goodput(records, 4, 1, duration), places=5)
            self.assertAlmostEqual(float(row["output_tokens_per_tick"]), sum(len(times) for arrival, times in records) / duration, places=5)

    def test_public_tests_reject_six_semantic_mutations(self):
        mutations = [
            ("lab05", "labs/lab05", "count = min(prefill, chunk_size, remaining)", "count = min(prefill, remaining)"),
            ("lab05", "labs/lab05", "decode_ids = decode_first(active, token_budget)", "decode_ids = []"),
            ("hw06", "hw/hw06", "(len(token_times) - 1)", "len(token_times)"),
            ("hw06", "hw/hw06", "metric[\"ttft\"] <= ttft_limit", "metric[\"ttft\"] < ttft_limit"),
            ("project2_scheduler", "projects/project2_scheduler", ".append(now + 1)", ".append(now)"),
            ("project2_scheduler", "projects/project2_scheduler", "if state[0] == 0:", "if phase == \"decode\":"),
        ]
        for name, relative, original, replacement in mutations:
            source = (COURSE / relative / f"{name}_solution.py").read_text(encoding="utf-8")
            self.assertIn(original, source)
            mutant = types.ModuleType(name)
            exec(compile(source.replace(original, replacement), f"mutant_{name}", "exec"), mutant.__dict__)
            previous = sys.modules.get(name)
            sys.modules[name] = mutant
            try:
                tests = load(f"{relative}/test_{name}.py", f"mutation_tests_{name}")
                with contextlib.redirect_stdout(io.StringIO()):
                    result = unittest.TextTestRunner(stream=io.StringIO()).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
                self.assertFalse(result.wasSuccessful(), f"tests missed {name}: {replacement}")
            finally:
                if previous is None:
                    del sys.modules[name]
                else:
                    sys.modules[name] = previous


if __name__ == "__main__":
    unittest.main(verbosity=2)
