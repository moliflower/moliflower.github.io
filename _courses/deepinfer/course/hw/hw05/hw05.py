"""HW5 exact prefix reuse and eviction; read the contracts in README.md."""


def prefix_hit_tokens(cached, query, block_size):
    """Q1: length of common prefix rounded DOWN to full blocks; preserve inputs."""
    raise NotImplementedError("HW5 Q1")


def shared_prefix_slots(lengths, prefix_tokens, block_size):
    """Q2: total allocated slots when all requests share one full-block prefix."""
    raise NotImplementedError("HW5 Q2")


def eviction_order(refcounts, last_used):
    """Q3: free IDs only, sorted by (last_used timestamp, physical ID)."""
    raise NotImplementedError("HW5 Q3")
