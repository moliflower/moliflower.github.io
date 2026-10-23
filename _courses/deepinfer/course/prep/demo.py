"""已完成的示例：观察累加器，不需要修改。"""


def sum_with_trace(values):
    total = 0
    rows = []
    for index, value in enumerate(values):
        before = total
        total += value
        rows.append((index, value, before, total))
    return total, rows


if __name__ == "__main__":
    result, trace = sum_with_trace([2, -1, 4])
    print("index value before after")
    for row in trace:
        print(*row)
    print("result =", result)
