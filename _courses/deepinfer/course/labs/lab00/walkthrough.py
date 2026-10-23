"""Q0：已完成的可运行例子，不依赖学生文件，不需要修改。"""

from simt import make_vector_gpu


def run_example():
    a, b, block = [1, 2, 3, 4, 5], [10, 20, 30, 40, 50], 4
    gpu, a_addr, b_addr, c_addr, n = make_vector_gpu(a, b)
    rows = []

    def kernel(tid, bid, block_dim, grid_dim, device):
        i = bid * block_dim + tid
        rows.append((bid, tid, i, "处理" if i < n else "跳过"))
        if i < n:
            left = device.load(a_addr + i)
            right = device.load(b_addr + i)
            device.store(c_addr + i, left + right)

    gpu.run(kernel, grid=2, block=block)
    return gpu.global_mem[c_addr:c_addr + n], rows, gpu.stats()


if __name__ == "__main__":
    result, rows, stats = run_example()
    print("bid tid i action")
    for row in rows:
        print(*row)
    print("c =", result)
    print("stats =", stats)
