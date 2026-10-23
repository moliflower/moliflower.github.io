"""Reference policy for the explicitly simplified Lab5 model."""


def positive_integer(value):
    if type(value) is not int or value <= 0:
        raise ValueError("expected a positive integer")


def validate_active(active):
    if not isinstance(active, list):
        raise ValueError("active must be a list")
    seen = set()
    for record in active:
        if not isinstance(record, (tuple, list)) or len(record) != 3:
            raise ValueError("expected (id, prefill_remaining, output_remaining)")
        request_id, prefill, output = record
        if not isinstance(request_id, str) or not request_id or request_id in seen:
            raise ValueError("IDs must be nonempty and unique")
        if type(prefill) is not int or prefill < 0:
            raise ValueError("prefill must be a nonnegative integer")
        positive_integer(output)
        seen.add(request_id)


def decode_first(active, token_budget):
    positive_integer(token_budget)
    validate_active(active)
    return [request_id for request_id, prefill, output in active if prefill == 0][:token_budget]


def plan_step(active, token_budget, chunk_size):
    positive_integer(chunk_size)
    decode_ids = decode_first(active, token_budget)
    plan = [(request_id, "decode", 1) for request_id in decode_ids]
    remaining = token_budget - len(plan)
    for request_id, prefill, output in active:
        if prefill and remaining:
            count = min(prefill, chunk_size, remaining)
            plan.append((request_id, "prefill", count))
            remaining -= count
    return plan
