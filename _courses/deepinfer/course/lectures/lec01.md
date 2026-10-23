# 第 1 讲：从 Python 建立 GPU 执行模型

> **核心必修**：Python 函数、循环、列表；不要求 C++、指针、CUDA 或 GPU。  
> **先修检查**：能修复简单索引错误，能看懂测试中的输入与期望；不熟悉就先做 [预备单元](../prep/README.md)。  
> **建议投入**：分两次，各 30–45 分钟，边读边手算；随后做 Lab0。

## 1. 今天只解决一个问题

输入两个等长列表，让很多“编号工人”共同完成：

```text
a = [1, 2, 3, 4, 5]
b = [10, 20, 30, 40, 50]
c = [11, 22, 33, 44, 55]
```

你在预备单元已经会写逐项加法。这一讲不增加新的数学，而是改变“谁负责哪个元素”。完成后你应该能：

- 根据组号和组内编号，算出负责的元素位置。
- 解释为什么最后一组需要跳过某些编号。
- 跟踪一次读、读、写的地址。
- 区分请求字节数、触及的 line 数和 sector 数。

Tensor Core、occupancy、指针和 CUDA 语法都不是本讲验收条件。

## 2. 从一个循环到多组编号

先看你熟悉的写法：

```python
c = []
for i in range(len(a)):
    c.append(a[i] + b[i])
```

这里一个循环依次处理所有 `i`。GPU 的想法是让多个线程分别处理不同的 `i`。课程先用 Python 依次调用各线程的函数，以便观察地址；它并没有真的并行，也不能用它的运行秒数推断 GPU 性能。

四个术语先记用途，不背硬件规格：

| 名称 | 本讲怎样理解 | 代码里的编号 |
| :--- | :--- | :--- |
| thread | 负责一个元素的执行者 | tid，组内编号 |
| block | 一组 thread | bid，组号 |
| grid | 本次启动的所有 block | grid，组的数量 |
| warp | NVIDIA 上用于理解指令发射的一组 32 个 lane | 本讲用来分组计访存 |

演示中的每组只有 4 个线程，是为了方便手算；真实 NVIDIA warp 的大小仍然是 32。小 block 会使用不完整的 warp。

## 3. 推导索引，而不是背索引

假设每组 4 个线程。第 0 组负责 0–3，第 1 组负责 4–7，第 2 组负责 8–11。

```text
组起点 = bid * block_dim
元素位置 i = 组起点 + tid
```

`block_dim` 是每组的线程数，`grid_dim` 是组数。不要把它们当成同一个量。

| bid | tid | block_dim | i |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 4 | 0 |
| 0 | 3 | 4 | 3 |
| 1 | 0 | 4 | 4 |
| 1 | 2 | 4 | 6 |

**停下来预测**：`bid=2, tid=1, block_dim=4` 应得到几？如果写成 `bid+tid`，哪些线程会重复处理同一个位置？

<details>
<summary>手算后检查</summary>

正确位置是 `2*4+1=9`。若写成 `bid+tid`，`(bid=0,tid=1)` 和 `(bid=1,tid=0)` 都得到 1；位置重复，另外一些位置则没人处理。

</details>

## 4. 为什么不能让最后一组随便访问

`n=5, block_dim=4`，需要两组。向上取整不是为了制造额外数据，而是为了提供足够的执行者。

```text
grid = ceil(n / block_dim) = ceil(5/4) = 2
```

| bid | tid | i | 是否小于 n | 动作 |
| ---: | ---: | ---: | :--- | :--- |
| 0 | 0..3 | 0..3 | 是 | 处理 |
| 1 | 0 | 4 | 是 | 处理 |
| 1 | 1 | 5 | 否 | 跳过 |
| 1 | 2 | 6 | 否 | 跳过 |
| 1 | 3 | 7 | 否 | 跳过 |

尾部条件是 `i < n`，不是 `i <= n`。长度 5 的合法索引为 0–4。

下面的代码片段展示算法，完整可运行示例在 `walkthrough.py`：

```python
i = bid * block_dim + tid
if i < n:
    c[i] = a[i] + b[i]
```

运行示例，不需要先实现学生题：

```bash
python course/labs/lab00/walkthrough.py
python course/tools/grade.py lab00 --question q0
```

对照输出中的 `bid tid i action`。如果与预测不同，先解释其中一行，不要立即改整个程序。

## 5. 列表索引怎样变成内存地址

为数地址，tracer 把三个数组放进一条一维存储空间。这里地址的单位为 **word**，一个 word 规定为 4 字节：

```text
n=5
a 在 0..4
b 在 5..9
c 在 10..14
```

处理 `i=2` 时：

```text
读 a：地址 0+2=2，得到 3
读 b：地址 5+2=7，得到 30
写 c：地址 10+2=12，写入 33
```

为了记录访问，示例不直接操作 Python 列表，而是调用下面两个方法：

```python
left = device.load(a_addr + i)
right = device.load(b_addr + i)
device.store(c_addr + i, left + right)
```

把 `device.load` 理解成“读值并记一笔”，`device.store` 理解成“写值并记一笔”。现在不用阅读 `GPU` 类内部，也不用学习 C++ 指针。

## 6. 结果相同，访存分组可以不同

连续分配使用 `i=bid*block_dim+tid`；另一种分配使用：

```text
i = tid * grid_dim + bid
```

例如 `n=12, block_dim=4, grid_dim=3`：

| bid | 连续分配：tid=0..3 | 跨步分配：tid=0..3 |
| ---: | :--- | :--- |
| 0 | 0,1,2,3 | 0,3,6,9 |
| 1 | 4,5,6,7 | 1,4,7,10 |
| 2 | 8,9,10,11 | 2,5,8,11 |

两种方案都覆盖 0–11，结果都可以正确。但同一组线程访问的地址距离不同。GPU 会把同一条内存指令、同一 warp 中落在同一 sector 的访问合并；地址分散通常需要更多 sector。

**不要先记“跨步一定慢几倍”**。当 grid=1 时两种分配相同；输入很小时访问也可能仍在少数 sector 内。后面用固定规模做可复现比较。

## 7. 三种计数，先分清单位

我们只讨论每 lane 读取一个 4 字节值的情况：

```text
word 地址 → byte 地址：word * 4
128B line 编号：word // 32
32B sector 编号：word // 8
```

同一 warp 的一次读取，数不同编号的个数：

| 32 个 lane 读取的 word 地址 | 有用字节 | line 数 | sector 数 |
| :--- | ---: | ---: | ---: |
| 0..31 | 128 | 1 | 4 |
| 16..47 | 128 | 2 | 4 |
| 1..32 | 128 | 2 | 5 |
| 0,8,16,...,248 | 128 | 8 | 32 |

第二行为什么不是 8 个 sector？16–47 对应 sector 2、3、4、5，恰好四个。跨两条 line 并不意味着必须传输两条完整 line。

Lab0 保留 `global_transactions` 这个旧字段名，但它准确的含义是“按 warp 指令累计触及的 128B line 数”。另外提供 `global_sectors`。这两个都不是 profiler 测得的实际 DRAM 读写量；模型不模拟缓存、写策略和时间。

## 8. 推导 vector add 的计数

每元素两次读、一次写，每值 4 字节：

```text
requested_bytes = n * (2+1) * 4 = 12n
FLOPs = n 次加法
```

`n=1024, block=128`，有 8 个 block，每 block 4 个 warp，总共 32 个 warp。a/b/c 起点都按 128B 对齐：

```text
连续：32 warps * 3 次访存 * 1 line = 96 lines
跨步：32 warps * 3 次访存 * 8 lines = 768 lines
```

两种的有用字节都是 `12*1024=12288`，但请求触及的 line 数不同。不能把这张表的比值直接叫实测 GPU 加速比。

## 9. 一个有条件的时间下界

设有效工作来自显存，先忽略缓存、启动和同步开销。硬件每秒最多搬 `BW` 字节、算 `P` 次浮点操作：

```text
memory_time = bytes / BW
compute_time = FLOPs / P
理想时间下界 = max(memory_time, compute_time)
```

假设搬 1GB、算 10GFLOP，BW=100GB/s，P=10TFLOP/s：

```text
memory_time = 1e9 / 100e9 = 0.01 秒
compute_time = 10e9 / 10e12 = 0.001 秒
下界为 0.01 秒；在这个模型下称为 memory-bound
```

两者不能直接相加来得到 Roofline 下界；而且下界不是性能保证。很小的任务可能受 kernel 启动延迟限制。这部分对应 Lab0 Q4，完成索引练习后再做即可。

## 10. 现在做什么，哪些先跳过

进入 [Lab0](../labs/lab00/README.md)，按 Q0–Q6 做。每题只引入一项变化，先预测再运行。验收问题：

1. `n=10, block=4` 时，最后一组哪些 tid 应跳过？
2. 能否解释一个元素的两次读、一次写的地址？
3. 为什么 word 16 开始的 32 个连续值触及两条 line，但只有四个 sector？

<details>
<summary>自查要点</summary>

最后一组 bid=2，tid=0、1 处理 i=8、9；tid=2、3 跳过。读写基址分别为 0、n、2n。line 和 sector 的划分粒度不同，必须分别计算。

</details>

完成本讲核心后，可读 [第 2 讲](lec02.md) 的访存推导。它的 occupancy、profiling 与 CUDA 示例先选读。[第一讲进阶](lec01_gpu.md) 保留硬件吞吐、CUDA、float4 等材料，等补过 C++ 后再看，不计入当前完成条件。
