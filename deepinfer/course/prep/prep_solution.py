"""预备单元参考解法：先记录定位过程，再对照。"""


def maximum(values):
    if not values:
        raise ValueError("maximum needs at least one value")
    best = values[0]
    for value in values[1:]:
        if value > best:
            best = value
    return best


def copy_matrix(matrix):
    return [row[:] for row in matrix]


def matmul(a, b):
    m, k, n = len(a), len(a[0]), len(b[0])
    if k != len(b):
        raise ValueError("inner dimensions do not match")
    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            total = 0.0
            for p in range(k):
                total += a[i][p] * b[p][j]
            out[i][j] = total
    return out


def assert_vector_close(actual, expected, atol=1e-6):
    if atol < 0:
        raise ValueError("atol must be non-negative")
    assert len(actual) == len(expected), "长度不一致"
    for i, (got, want) in enumerate(zip(actual, expected)):
        assert abs(got - want) <= atol, f"位置 {i}: actual={got}, expected={want}"


def vector_add(a, b):
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    return [x + y for x, y in zip(a, b)]
