"""自测后阅读的参考解；grader 在临时副本中使用，不覆盖学生文件。"""


def add_example(left, right):
    total = left + right
    return total


def bytes_for_elements(count, element_bytes):
    if count < 0 or element_bytes <= 0:
        raise ValueError("count must be >= 0 and element_bytes must be > 0")
    return count * element_bytes


def ceil_groups(count, group_size):
    if count < 0 or group_size <= 0:
        raise ValueError("count must be >= 0 and group_size must be > 0")
    full_groups = count // group_size
    remainder = count % group_size
    if remainder == 0:
        return full_groups
    return full_groups + 1


def clamp_lengths(lengths, limit):
    if limit < 0 or any(length < 0 for length in lengths):
        raise ValueError("lengths and limit must be >= 0")
    result = []
    for length in lengths:
        if length > limit:
            result.append(limit)
        else:
            result.append(length)
    return result


def prefix_offsets(lengths):
    if any(length < 0 for length in lengths):
        raise ValueError("lengths must be >= 0")
    offsets = [0]
    total = 0
    for length in lengths:
        total = total + length
        offsets.append(total)
    return offsets
