"""Reference implementation; all times are virtual ticks, not GPU latency."""


def positive_integer(value):
    if type(value) is not int or value <= 0:
        raise ValueError("expected a positive integer")


def plan_step(active, token_budget, chunk_size):
    positive_integer(token_budget)
    positive_integer(chunk_size)
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
    plan = []
    remaining = token_budget
    for request_id, prefill, output in active:
        if prefill == 0 and remaining:
            plan.append((request_id, "decode", 1))
            remaining -= 1
    for request_id, prefill, output in active:
        if prefill and remaining:
            count = min(prefill, chunk_size, remaining)
            plan.append((request_id, "prefill", count))
            remaining -= count
    return plan


def simulate(requests, max_active, token_budget, chunk_size):
    for value in (max_active, token_budget, chunk_size):
        positive_integer(value)
    if token_budget < max_active:
        raise ValueError("token_budget must cover every active decode")
    if not isinstance(requests, list):
        raise ValueError("requests must be a list")
    results = {}
    for record in requests:
        if not isinstance(record, (tuple, list)) or len(record) != 4:
            raise ValueError("expected (id, arrival, prompt_tokens, output_tokens)")
        request_id, arrival, prompt, output = record
        if not isinstance(request_id, str) or not request_id or request_id in results:
            raise ValueError("IDs must be globally unique and nonempty")
        if type(arrival) is not int or arrival < 0:
            raise ValueError("arrival must be a nonnegative integer")
        positive_integer(prompt)
        positive_integer(output)
        results[request_id] = {"arrival": arrival, "token_times": []}
    pending = sorted(requests, key=lambda record: record[1])
    cursor, now = 0, 0
    waiting, active, trace = [], {}, []
    while cursor < len(pending) or waiting or active:
        if not waiting and not active:
            now = max(now, pending[cursor][1])
        while cursor < len(pending) and pending[cursor][1] <= now:
            waiting.append(pending[cursor])
            cursor += 1
        admitted = []
        while waiting and len(active) < max_active:
            request_id, arrival, prompt, output = waiting.pop(0)
            active[request_id] = [prompt, output]
            admitted.append(request_id)
        plan = plan_step([(request_id, *state) for request_id, state in active.items()],
                         token_budget, chunk_size)
        emitted, finished = [], []
        for request_id, phase, count in plan:
            state = active[request_id]
            if phase == "prefill":
                state[0] -= count
            if state[0] == 0:
                state[1] -= 1
                emitted.append(request_id)
                results[request_id]["token_times"].append(now + 1)
            if state[1] == 0:
                finished.append(request_id)
        for request_id in finished:
            del active[request_id]
        trace.append({
            "start": now, "end": now + 1, "admitted": admitted, "plan": plan,
            "emitted": emitted, "finished": finished,
            "active": {request_id: tuple(state) for request_id, state in active.items()},
            "waiting": [record[0] for record in waiting],
        })
        now += 1
    return {"trace": trace, "requests": results}
