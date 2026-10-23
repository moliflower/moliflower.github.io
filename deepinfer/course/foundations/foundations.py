"""Python 起步练习。只修改 TODO 区域；先读 README，再按题运行测试。"""


def add_example(left, right):
    """Q0 已完成示例：返回两个数的和，不负责打印。"""
    total = left + right
    return total


def bytes_for_elements(count, element_bytes):
    """Q1: count 个元素的字节数。count>=0, element_bytes>0，均为整数。"""
    if count < 0 or element_bytes <= 0:
        raise ValueError("count must be >= 0 and element_bytes must be > 0")
    # TODO: 返回元素总字节数。
    raise NotImplementedError("Q1: bytes_for_elements")


def ceil_groups(count, group_size):
    """Q2: 容纳 count 个元素至少需要多少组，每组容量 group_size。

    整数输入；count>=0、group_size>0。不用浮点除法或 math.ceil。
    例如 (7, 3)->3，(6, 3)->2，(0, 3)->0。
    """
    if count < 0 or group_size <= 0:
        raise ValueError("count must be >= 0 and group_size must be > 0")
    full_groups = count // group_size
    remainder = count % group_size
    # TODO: 根据 remainder 决定是否还需要一组。
    raise NotImplementedError("Q2: ceil_groups")


def clamp_lengths(lengths, limit):
    """Q3: 返回新列表，将每个长度限制在 limit 以内；不改输入。

    lengths 为非负整数列表，limit 为非负整数；空列表返回新的 []。
    例如 ([2, 8, 0], 5)->[2, 5, 0]。
    """
    if limit < 0 or any(length < 0 for length in lengths):
        raise ValueError("lengths and limit must be >= 0")
    result = []
    # TODO: 用 for/if/append 填充 result，再 return；不要求列表推导式。
    raise NotImplementedError("Q3: clamp_lengths")


def prefix_offsets(lengths):
    """Q4: 返回每段的起点及最后终点，共 len(lengths)+1 个数。

    lengths 为非负整数列表。不改输入，空列表返回 [0]。
    例如 [2, 0, 3]->[0, 2, 2, 5]。独立实现，无前题函数依赖。
    """
    if any(length < 0 for length in lengths):
        raise ValueError("lengths must be >= 0")
    # TODO: 自己选择累计变量、结果列表和循环的位置。
    raise NotImplementedError("Q4: prefix_offsets")
