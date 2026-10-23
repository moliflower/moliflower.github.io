# Lab0：先预测地址，再写一个小算子

> **核心必修**：纯 Python；不需要 C++、CUDA、GPU。  
> **前置**：[预备单元](../../prep/README.md)、[第 1 讲](../../lectures/lec01.md)。  
> **节奏**：建议分两三次完成，每次只做一两题，按理解推进而不是赶时间。  
> **修改**：`lab00.py`；测试、tracer、示例不需要改。

## 1. 路线与文件

| 步骤 | 任务 | 本次新增的能力 |
| :--- | :--- | :--- |
| Q0 | 运行完成的例子 | 读编号表，不从空白开始 |
| Q1 | 填连续索引 | 把分组关系写成公式 |
| Q2 | 填跨步索引 | 改分配顺序但不改结果 |
| Q3 | 组合前两题 | 从统计结果提取信息 |
| Q4 | 写时间下界模型 | 对齐字节、FLOPs 与时间单位 |
| Q5 | 独立写 vector_scale | 迁移到相近但未提供骨架的任务 |
| Q6 | 修复尾部越界 | 用最小复现解释一个 bug |

`walkthrough.py` 是完整示例，`simt.py` 是访问追踪工具，`lab00_solution.py` 是完整解法。**先会用 tracer，不要求读懂它的内部实现。**

所有 `python course/...` 命令从 DeepInfer 根目录运行；使用 WSL 且命令名为 `python3` 时统一替换即可。

## 2. Q0：一个确定能运行的起点

```bash
python course/labs/lab00/walkthrough.py
python course/tools/grade.py lab00 --question q0
```

结果应为 `c=[11,22,33,44,55]`。最后一个执行者 `bid=1,tid=3,i=7` 被标为“跳过”。这一步不依赖任何待实现函数，应该直接通过。

先自己解释 `(bid=1,tid=0)` 为什么处理第五个元素；再进入 Q1。

## 3. Q1：只填一个索引公式

在 `lab00.py` 找到 `coalesced_index(tid,bid,block_dim)`，替换它的 `raise NotImplementedError`。其余 `vector_add` 骨架已经给出。

先填表：

| bid | tid | block_dim | 元素位置 i |
| ---: | ---: | ---: | :--- |
| 0 | 0 | 4 | 自己填 |
| 0 | 3 | 4 | 自己填 |
| 1 | 0 | 4 | 自己填 |
| 1 | 2 | 4 | 自己填 |

计算完再写公式。骨架会检查 `i<n`，并通过 `device.load/store` 读写。`make_vector_gpu(a,b)` 返回的五个值是：设备对象、a 的起点、b 的起点、c 的起点、长度 n。

```bash
python course/tools/grade.py lab00 --question q1 --failfast
```

<details>
<summary>提示 1：从每组起点想</summary>

第 bid 组前面已有 bid 个完整组，每组 block_dim 个元素。

</details>

<details>
<summary>提示 2：完整公式</summary>

`bid*block_dim+tid`。这里的乘数是每组人数，不是组数。

</details>

测试含 `n=1000` 的尾部、空输入、负数、输入不被修改。若只修公式仍报越界，先确认没有改动已经提供的 `if i<n`。

## 4. Q2：换分配顺序，不换输出顺序

实现 `strided_index(tid,bid,grid_dim)`。先用 `grid=3,block=4` 列出每组负责的元素：

```text
bid=0: 0,3,6,9
bid=1: 1,4,7,10
bid=2: 2,5,8,11
```

输出的 `c[i]` 仍然对应第 i 个输入；不要把结果按执行顺序追加到列表。

```bash
python course/tools/grade.py lab00 --question q2 --failfast
```

<details>
<summary>提示 1：相邻 tid 距离多少？</summary>

固定 bid 后，tid 每增加 1，元素位置增加 grid_dim。

</details>

<details>
<summary>提示 2：公式</summary>

`tid*grid_dim+bid`。对于 `n=10,block=4,grid=3`，仍要跳过大于等于 10 的索引；现成骨架已经处理。

</details>

## 5. Q3：先预测统计，再提取字段

实现 `compare_transactions(n,block)`，返回 `(连续line数,跨步line数)`。复用 Q1/Q2；统计字典的 `global_transactions` 字段保留旧名，指的是 **128B line 触及次数**。

对 `n=1024,block=128`：

```text
grid=8
warps 总数=8*(128/32)=32
每 warp 两次 load，一次 store
连续：32*3*1=96 lines
跨步：32*3*8=768 lines
```

```bash
python course/tools/grade.py lab00 --question q3
```

8 倍是本模型在这个输入下的计数比，不是“GPU 一定快 8 倍”。当 grid=1 时，两种索引相同。

## 6. 统计字段怎样读

| 字段 | 含义 | 不能把它当成什么 |
| :--- | :--- | :--- |
| global_loads / stores | 各线程显式读写的元素次数 | 缓存 miss 次数 |
| bytes_moved | 请求字节：4*(loads+stores) | 实测 HBM traffic |
| global_transactions | 按 warp 指令累计的 128B line 数 | 完整 line 一定被传输 |
| global_sectors | 同样分组的 32B sector 数 | 完整显存协议的请求数 |
| sector_bytes | sector 数乘 32B | profiler 测得的 DRAM 字节数 |

tracer 依次运行线程、不模拟 barrier 或缓存；内部用“线程第几次内存访问”近似指令身份。本 Lab 的直线读写与统一尾部 guard 满足用途；带有不同访存分支或不同循环次数的任意 kernel 不在模型支持范围。

word 16–47 触及两条 line、四个 sector，不能按两条完整 line 算成 256B。这一点在基础设施回归测试里固定下来。

## 7. Q4：实现带单位的 Roofline 下界

`roofline` 返回 `(秒数,bound)`。规格：

```text
memory_time = bytes_moved/(bandwidth_gbps*1e9)
compute_time = flops/(peak_tflops*1e12)
seconds = 两者最大值
```

带宽和算力必须为正，否则 `ValueError`；相等时 bound 取 `"memory"`，否则取较大时间对应的 `"memory"` 或 `"compute"`。本题输入的 bytes/FLOPs 为非负数。

手算搬 1GB、算 1TFLOP、100GB/s、10TFLOP/s：两个时间是 0.01 秒和 0.1 秒，答案 `(0.1,"compute")`。

```bash
python course/tools/grade.py lab00 --question q4
```

这不是模拟器计时结果，也不包含 kernel 启动延迟。

## 8. Q5：独立实现相近的新算子

这次实现 `vector_scale(a,factor,block=128)`，返回 `(c,stats)`，其中 `c[i]=a[i]*factor`。不再提供函数体；可以回看 `_run_add`，但先自己列步骤。

先读懂提供的骨架，不用猜 Python 新语法：

- `def kernel(...)` 是普通函数，只是定义在另一个函数内部。
- 内层函数可以读取外层已有的 `n`、`a_addr` 等变量；这里不需要给外层变量重新赋值。
- `gpu.run(kernel,grid,block)` 把这个函数交给 tracer，tracer 为每个线程填入 `tid,bid,block_dim,grid_dim,device` 后调用它。写 `kernel` 是传函数，写 `kernel(...)` 才是立刻调用。
- `device.load/store` 是已经提供的读写工具；它们多做了一件“记录地址”的事。

这与下面这段普通 Python 循环的“重复调用函数”相似；区别是 tracer 会额外设置记录上下文，所以实际练习要调用 `gpu.run`，不要自己替换它：

```python
for bid in range(grid):
    for tid in range(block):
        kernel(tid, bid, block, grid, gpu)
```

如果传函数的写法还不熟，先在纸上推演 `grid=2,block=2` 时的四次调用，再开始独立实现。

要求：通过 tracer 每元素读一次、写一次；不修改 a；空输入零访问；block 在 1–1024。可以用 `make_vector_gpu(a,[0]*len(a))` 获得一维布局，把占位 b 留着不读，不必修改 tracer。

```text
输入 [1,-2,3,0,5]，factor=-3
期望 [-3,6,-9,0,-15]
loads=5，stores=5，requested_bytes=40
```

```bash
python course/tools/grade.py lab00 --question q5 --failfast
python course/tools/grade.py lab00 --question q5 --hint 1
```

测试会同时检查输出和访问计数。直接用列表推导计算输出却不经 tracer，结果正确也不能满足本题目标。

## 9. Q6：修复尾部错误

`vector_add_debug` 是完整但有 bug 的版本。先只运行本题：

```bash
python course/tools/grade.py lab00 --question q6 --failfast
```

输入长度 5，block=4，分配的内存地址是 0–14。查看异常中的地址，回到源码定位触发读写的 `i`。

先回答：`i==5` 对 a/b/c 都是合法元素位置吗？不要通过增大分配内存或捕获并忽略 `IndexError` 来掩盖错误。

<details>
<summary>定位过程</summary>

合法位置是 0–4；第 1 组 tid=1 算出 i=5。错误 guard 允许它继续执行，写 c 的地址为 10+5=15，超出范围。修复 guard，而不是跳过错误测试。

</details>

修复后确认：完整 block、尾部 block、空输入均通过。写下一个“修复前失败、修复后通过”的输入。

## 10. 出口与下一步

```bash
python course/tools/grade.py lab00
```

全部运行还会包含 `test_simt.py`，它检查 tracer 的 sector 和广播规则，不需要你修改。

能解释以下三个问题就完成本实验；卡住的点标记为“需要复习”，不抹去已经完成的进度：

1. n=10、block=4 时，最后一组谁应该跳过？
2. Q5 为什么只有一次 load，而 Q1 有两次？
3. 为什么 line 计数、sector 计数、有用字节不能混为一个指标？

下一步是 [第 3 讲](../../lectures/lec03.md) 和 Lab1 的 Python 算法部分。也可先读 [第 2 讲](../../lectures/lec02.md) 的访存推导；CUDA、occupancy、profiling 都可留待后续。现在不要求做 KV/量化作业，也不要求实现完整 Project1。
