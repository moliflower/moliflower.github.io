"""Phase 1 reference. Simulates reserved page counts, not tokens or GPU time."""


def integer(value, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid count")


def validate_requests(waiting, block_size):
    integer(block_size, 1)
    seen = set()
    for request_id, length in waiting:
        if not isinstance(request_id, str) or not request_id or request_id in seen:
            raise ValueError("invalid or duplicate ID")
        integer(length, 1)
        seen.add(request_id)


def admit_fifo(waiting, free_blocks, block_size):
    integer(free_blocks, 0)
    validate_requests(waiting, block_size)
    accepted = []
    for request_id, length in waiting:
        pages = (length + block_size - 1) // block_size
        if pages > free_blocks:
            break
        free_blocks -= pages
        accepted.append(request_id)
    return accepted, free_blocks


def replay(events, capacity, block_size):
    integer(capacity, 1)
    integer(block_size, 1)
    waiting, active, seen, snapshots = [], {}, set(), []
    for event in events:
        if not isinstance(event, (tuple, list)) or not event:
            raise ValueError("invalid event")
        if event[0] == "submit" and len(event) == 3:
            _, request_id, length = event
            validate_requests([(request_id, length)], block_size)
            if request_id in seen or (length + block_size - 1) // block_size > capacity:
                raise ValueError("duplicate or impossible request")
            seen.add(request_id)
            waiting.append((request_id, length))
        elif event[0] == "finish" and len(event) == 2:
            request_id = event[1]
            if not isinstance(request_id, str) or request_id not in active:
                raise ValueError("request is not active")
            del active[request_id]
        else:
            raise ValueError("unknown event")
        free = capacity - sum(active.values())
        accepted, free = admit_fifo(waiting, free, block_size)
        for request_id, length in waiting[:len(accepted)]:
            active[request_id] = (length + block_size - 1) // block_size
        waiting = waiting[len(accepted):]
        snapshots.append({"active": dict(active), "waiting": [request_id for request_id, _ in waiting], "free_blocks": free})
    return snapshots
