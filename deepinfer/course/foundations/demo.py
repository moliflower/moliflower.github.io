"""从仓库根目录运行：python course/foundations/demo.py。"""

from foundations import add_example


def main():
    count = 3
    element_bytes = 2
    print("count =", count)
    print("bytes =", count * element_bytes)
    print("returned =", add_example(2, 5))
    total = 0
    for length in [2, 0, 3]:
        before = total
        total = total + length
        print("length/before/after =", length, before, total)


if __name__ == "__main__":
    main()
