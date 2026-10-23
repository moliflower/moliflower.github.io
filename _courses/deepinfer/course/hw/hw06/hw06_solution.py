"""Reference metrics with explicit finite-number and singleton contracts."""

import math


def nonnegative_number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("expected a finite nonnegative number")


def request_metrics(arrival, token_times):
    nonnegative_number(arrival)
    if not isinstance(token_times, list) or not token_times:
        raise ValueError("token_times must be a nonempty list")
    for index, timestamp in enumerate(token_times):
        nonnegative_number(timestamp)
        if timestamp < arrival or index and timestamp <= token_times[index - 1]:
            raise ValueError("tokens must be strictly ordered after arrival")
    tpot = None if len(token_times) == 1 else (token_times[-1] - token_times[0]) / (len(token_times) - 1)
    return {"ttft": token_times[0] - arrival, "tpot": tpot, "e2e": token_times[-1] - arrival}


def all_metrics(records):
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    metrics = []
    for record in records:
        if not isinstance(record, (tuple, list)) or len(record) != 2:
            raise ValueError("expected (arrival, token_times)")
        metrics.append(request_metrics(*record))
    return metrics


def aggregate_metrics(records):
    metrics = all_metrics(records)
    tpots = [metric["tpot"] for metric in metrics if metric["tpot"] is not None]
    return {
        "mean_ttft": sum(metric["ttft"] for metric in metrics) / len(metrics) if metrics else None,
        "mean_tpot": sum(tpots) / len(tpots) if tpots else None,
    }


def slo_goodput(records, ttft_limit, tpot_limit, duration):
    for value in (ttft_limit, tpot_limit, duration):
        nonnegative_number(value)
    if duration == 0:
        raise ValueError("duration must be positive")
    metrics = all_metrics(records)
    if any(arrival > duration or times[-1] > duration for arrival, times in records):
        raise ValueError("all records must fit observation window [0, duration]")
    successes = sum(metric["ttft"] <= ttft_limit and
                    (metric["tpot"] is None or metric["tpot"] <= tpot_limit)
                    for metric in metrics)
    return successes / duration
