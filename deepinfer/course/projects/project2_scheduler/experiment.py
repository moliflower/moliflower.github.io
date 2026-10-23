"""Print reproducible virtual-tick experiments as CSV; no files are overwritten."""

import argparse
import csv
import sys


WORKLOADS = {
    "short": [("A", 0, 2, 3), ("B", 0, 2, 2), ("C", 1, 1, 1)],
    "long_then_short": [("long", 0, 12, 3), ("short", 1, 1, 2), ("late", 2, 2, 1)],
    "idle_gap": [("A", 0, 2, 2), ("B", 10, 2, 2)],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", action="store_true", help="explicitly run the public reference solution")
    args = parser.parse_args()
    if args.solution:
        from project2_scheduler_solution import simulate
    else:
        from project2_scheduler import simulate
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["workload", "max_active", "budget", "chunk", "duration_ticks", "mean_ttft_ticks",
                     "mean_tpot_ticks", "output_tokens_per_tick", "slo_requests_per_tick"])
    for name, requests in WORKLOADS.items():
        for slots in (1, 2):
            for chunk in (1, 2, 4):
                result = simulate(requests, slots, 4, chunk)
                duration = result["trace"][-1]["end"]
                ttfts, tpots, successes, tokens = [], [], 0, 0
                for record in result["requests"].values():
                    times = record["token_times"]
                    ttft = times[0] - record["arrival"]
                    tpot = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else None
                    ttfts.append(ttft)
                    if tpot is not None:
                        tpots.append(tpot)
                    successes += ttft <= 4 and (tpot is None or tpot <= 1)
                    tokens += len(times)
                writer.writerow([name, slots, 4, chunk, duration, round(sum(ttfts) / len(ttfts), 6),
                                 round(sum(tpots) / len(tpots), 6) if tpots else "NA",
                                 round(tokens / duration, 6), round(successes / duration, 6)])


if __name__ == "__main__":
    main()
