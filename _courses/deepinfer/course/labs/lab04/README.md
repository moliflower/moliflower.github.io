# Lab4：有限块池与共享所有权

先修：Lab3 → [第 9 讲](../../lectures/lec09.md)。约 2–3 小时，可分两次完成。只用 Python 标准库，修改 `course/labs/lab04/lab04.py`。这是块生命周期模型，没有 KV 数值、并发或 GPU。

## 1. 先跑一条可解释的失败

从仓库根目录执行：

```bash
python course/tools/grade.py lab04 --question q1 --unlock
python course/tools/grade.py lab04 --question q1 --failfast
python course/tools/grade.py lab04 --question q1 --hint 1
```

初始 NotImplementedError 表示题目尚未实现。Windows 可把 `python course/tools/grade.py` 替换成 `.\course\run.ps1`。

## 2. API 与契约

构造器已提供：`BlockPool(capacity)` 创建 refs 列表，长度为非负整数 capacity。`available()` 返回 refs=0 的数量。所有整数都排除 bool。

| 题号 | 接口 | 必须满足的行为 |
| :--- | :--- | :--- |
| Q1 | `allocate(count)` | 非负整数；按最小 ID 取空闲块，将引用数设为 1，返回新的 ID 列表；0 返回 []；不足抛 MemoryError；非法 count 抛 ValueError |
| Q2 | `retain(block_ids)` | 输入为不重复整数 ID 的列表，每块必须在范围内且 refs>0；整批验证后每块 +1，返回 None |
| Q3 | `release(block_ids)` | 与 Q2 同样验证；每块 -1；归零即可再分配，返回 None |
| Q4 | `locate(table,t,B)` | table 是非负整数 ID 列表，B 为正整数，t 为整数；要求 0≤t<len(table)*B；返回 `(table[t//B],t%B)` |

Q2/Q3 空列表有效。重复、空闲、负数、越界、bool ID 均抛 ValueError。**任何失败都不能部分改变 refs**。不能修改输入列表；返回的 ID 列表不得与内部状态共享。调用者不直接修改 refs；它公开只是为了手算与观察。

Q4 不要求 table 的 ID 连续或唯一；它不掌握池容量与真实序列长度，也不验证页中是否已经写入 token。空表没有可访问位置。

## 3. 先手算，再实现

```text
pool = BlockPool(3)
A = pool.allocate(2)      -> [0,1], refs=[1,1,0]
pool.retain([0])          -> refs=[2,1,0]
B_tail = pool.allocate(1) -> [2], refs=[2,1,1]
pool.release(A)           -> refs=[1,0,1]
pool.allocate(1)          -> [1], refs=[1,1,1]
```

最后一步复用的是物理块 1，而不是增长出新的块 3。写代码时把验证和变更分开；Q2/Q3 可以共享一个验证函数。

## 4. 调试任务与验收

构造 refs=`[1,0]`，尝试 release `[0,1]`。如果失败后得到 `[0,0]`，记录：期望、实际、哪一次写操作发生得太早，以及如何回归验证。

```bash
python course/tools/grade.py lab04
```

测试包含 150 步固定种子所有权轨迹，并核对每一块的引用数。通过以后仍需写出：共享块为何不会在第一个请求结束时释放；MemoryError 如何与 ValueError 区分；locate 为何不能取代序列长度检查。

自评四题各 5 分：测试 3 分、手算/解释 2 分，共 20 分，属于 Unit2 试运行验收，不并入 Unit1。下一步：[第 10 讲](../../lectures/lec10.md) 与 [HW5](../../hw/hw05/README.md)。
