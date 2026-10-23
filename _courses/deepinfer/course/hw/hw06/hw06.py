"""HW6 latency and SLO arithmetic, independent of any GPU or model."""


def request_metrics(arrival, token_times):
    """Q1: return ttft, tpot (None for one token), and e2e."""
    raise NotImplementedError("HW6 Q1")


def aggregate_metrics(records):
    """Q2: request-weighted mean_ttft and mean_tpot; exclude undefined TPOT."""
    raise NotImplementedError("HW6 Q2")


def slo_goodput(records, ttft_limit, tpot_limit, duration):
    """Q3: requests meeting both applicable limits per unit observation time."""
    raise NotImplementedError("HW6 Q3")
