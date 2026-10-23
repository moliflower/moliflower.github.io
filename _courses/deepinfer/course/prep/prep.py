"""预备单元：有意保留三个错误和一个待实现函数。

按 01-debug.md → 02-matrix.md → 03-tests.md 修复。
函数接口不改；demo.py 和测试文件无需修改。
"""


def maximum(values):
    """Q1 修错：非空列表返回最大值；空列表抛 ValueError。"""
    best = 0  # TODO: 用全负数和空列表检查这个初值。
    for value in values:
        if value > best:
            best = value
    return best


def copy_matrix(matrix):
    """Q2 修错：返回每一行都独立的副本，不修改输入。"""
    return matrix[:]  # TODO: 外层复制后，内层的行还是同一个对象吗？


def matmul(a, b):
    """Q2 修错：非空矩形 A(m,k) @ B(k,n)，返回新矩阵。

    本题只要求非空矩形输入，内维不匹配时抛 ValueError。
    """
    m, k, n = len(a), len(a[0]), len(b[0])
    if k != len(b):
        raise ValueError("inner dimensions do not match")
    out = [[0.0] * n for _ in range(m)]
    total = 0.0  # TODO: 用手算的四个输出检查应该在哪一层清零。
    for i in range(m):
        for j in range(n):
            for p in range(k):
                total += a[i][p] * b[p][j]
            out[i][j] = total
    return out


def assert_vector_close(actual, expected, atol=1e-6):
    """Q3 修错：长度不同或任一绝对误差大于 atol 时抛 AssertionError。

    输入为有限数值列表；atol 必须为非负数，否则抛 ValueError。
    """
    if atol < 0:
        raise ValueError("atol must be non-negative")
    assert len(actual) == len(expected), "长度不一致"
    for i, (got, want) in enumerate(zip(actual, expected)):
        assert got == want, f"位置 {i}: actual={got}, expected={want}"  # TODO: 浮点数如何比较？


def vector_add(a, b):
    """Q3 独立实现：返回逐项和；长度不匹配抛 ValueError；不修改输入。"""
    raise NotImplementedError("预备 Q3：实现 vector_add")
