"""Reference for explicit teaching assumptions, not a production cache."""


def count(value, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError("invalid count")


def prefix_hit_tokens(cached, query, block_size):
    count(block_size, True)
    for tokens in (cached, query):
        if not isinstance(tokens, list):
            raise ValueError("tokens must be a list")
        for token in tokens:
            count(token)
    common = 0
    for old, new in zip(cached, query):
        if old != new:
            break
        common += 1
    return common // block_size * block_size


def shared_prefix_slots(lengths, prefix_tokens, block_size):
    count(block_size, True)
    count(prefix_tokens)
    if not isinstance(lengths, list) or prefix_tokens % block_size:
        raise ValueError("invalid prefix")
    for length in lengths:
        count(length)
        if length < prefix_tokens:
            raise ValueError("prefix exceeds request")
    if not lengths:
        return 0
    return prefix_tokens + sum((length - prefix_tokens + block_size - 1) // block_size * block_size for length in lengths)


def eviction_order(refcounts, last_used):
    if not isinstance(refcounts, list) or not isinstance(last_used, list) or len(refcounts) != len(last_used):
        raise ValueError("incompatible lists")
    for value in refcounts + last_used:
        count(value)
    return sorted((block for block, refs in enumerate(refcounts) if refs == 0), key=lambda block: (last_used[block], block))
