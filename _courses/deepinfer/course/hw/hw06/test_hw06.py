import copy
import unittest
from hw06 import request_metrics, aggregate_metrics, slo_goodput


class TestLatency(unittest.TestCase):
    def test_example_singleton_and_zero_ttft(self):
        self.assertEqual(request_metrics(2, [5, 7, 11]), {"ttft": 3, "tpot": 3, "e2e": 9})
        self.assertEqual(request_metrics(0, [0]), {"ttft": 0, "tpot": None, "e2e": 0})
        self.assertAlmostEqual(request_metrics(0.1, [0.3, 0.5])["ttft"], 0.2)

    def test_invalid_arrival_times_and_order(self):
        for arrival in (-1, True, float("nan"), float("inf"), "0", None):
            with self.assertRaises(ValueError):
                request_metrics(arrival, [1])
        for times in ([], None, (1,), [1, 1], [2, 1], [0], [True], [float("nan")], [float("inf")], ["2"]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                request_metrics(1, times)

    def test_preserves_timestamp_list(self):
        times = [2, 3, 6]
        metrics = request_metrics(1, times)
        metrics["ttft"] = 100
        self.assertEqual(times, [2, 3, 6])


class TestAggregate(unittest.TestCase):
    def test_request_mean_not_gap_weighted(self):
        records = [(0, [1, 5]), (0, [2, 3, 4, 5]), (1, [2])]
        self.assertEqual(aggregate_metrics(records), {"mean_ttft": 4 / 3, "mean_tpot": 2.5})

    def test_empty_and_singletons(self):
        self.assertEqual(aggregate_metrics([]), {"mean_ttft": None, "mean_tpot": None})
        self.assertEqual(aggregate_metrics([(1, [4])]), {"mean_ttft": 3, "mean_tpot": None})

    def test_invalid_and_purity(self):
        for records in (None, (), [None], [(0, [])], [(0, [1], 2)], [(0, [1]), (2, [1])]):
            with self.assertRaises(ValueError):
                aggregate_metrics(records)
        records = [[0, [1, 4]]]
        before = copy.deepcopy(records)
        aggregate_metrics(records)
        self.assertEqual(records, before)


class TestGoodput(unittest.TestCase):
    def test_inclusive_limits_and_singleton(self):
        records = [(0, [2, 4]), (0, [2, 6]), (1, [2])]
        self.assertEqual(slo_goodput(records, 2, 2, 6), 2 / 6)
        self.assertEqual(slo_goodput([(0, [0])], 0, 0, 1), 1)
        self.assertEqual(slo_goodput([], 0, 0, 3), 0)

    def test_window_validation_even_for_failed_slo(self):
        with self.assertRaises(ValueError):
            slo_goodput([(1, [9])], 0, 0, 8)
        for value in (-1, True, float("nan"), float("inf"), None):
            for args in ((value, 1, 1), (1, value, 1), (1, 1, value)):
                with self.assertRaises(ValueError):
                    slo_goodput([], *args)
        with self.assertRaises(ValueError):
            slo_goodput([], 1, 1, 0)

    def test_bad_record_cannot_be_filtered_away(self):
        records = [(0, [1]), (0, [4, 3])]
        before = copy.deepcopy(records)
        with self.assertRaises(ValueError):
            slo_goodput(records, 0, 0, 5)
        self.assertEqual(records, before)
