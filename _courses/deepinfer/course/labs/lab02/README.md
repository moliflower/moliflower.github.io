# Lab2：用小函数组合 Attention

> **阅读**：[L3 数值算子](../../lectures/lec03.md)、[L5 Attention](../../lectures/lec05.md)；先完成预备矩阵练习与 Lab1。  
> **投入**：4–6 小时。必做 Q1–Q8，分三个 checkpoint。  
> **交付**：`lab02.py`、shape 表、一次调试记录、checkoff 回答。

## 0. 文件和命令

学生框架 `lab02.py` 给定八个接口；`test_lab02.py` 为公开测试；参考文件在 `lab02_solution.py`。标准库即可，不需要 GPU。测试前先预测：

```bash
python3 course/tools/grade.py lab02 --question q1 --unlock
python3 course/tools/grade.py lab02 --question q1 --failfast
python3 course/tools/grade.py lab02 --question q1 --hint 1
```

将 q1 换成 q2–q8 可逐题运行。只修改学生文件；保留测试和函数签名。新函数可以组合前题，但不要复制多份相同循环。

## 1. Review：shape 先于代码

本 Lab 中 Q 是 `(sq,d)`，K/V 都是 `(sk,d)`，d>0。非空矩阵保证为矩形；一般 ragged tensor 验证不属于此 Lab。数值有限、算术中间值可表示，只有 Q3 另允许 mask 的 -inf。`matmul/transpose` 的空输入按框架返回 `[]`，不模拟通用零维 shape 系统。所有结果是新列表，输入不变。维度/计数参数必须是整数且不能用bool代替。

| 阶段 | 输入 | 输出 |
| :--- | :--- | :--- |
| transpose | K(sk,d) | Kᵀ(d,sk) |
| matmul | Q(sq,d), Kᵀ(d,sk) | S(sq,sk) |
| scale/mask/softmax | S(sq,sk) | P(sq,sk) |
| matmul | P(sq,sk), V(sk,d) | O(sq,d) |

完整小例子：Q=`[[0],[0]]`、K=`[[1],[2]]`、V=`[[2],[6]]`。所有点积是 0。非 causal 的两行概率都是 `[0.5,0.5]`，输出 `[[4],[4]]`；causal 第一行概率为 `[1,0]`，输出变为 `[[2],[4]]`。

**预测**：只将未来 V[1] 改为 `[100]`，causal 的 O[0] 是否改变？先写原因，再运行。

## 2. Q1：矩阵乘

`matmul(a,b)`：A(m,k)@B(k,n) 返回新矩阵 C(m,n)；内维不匹配抛 `ValueError`。三层循环中 i 控制输出行、j 控制输出列、p 控制点积。每格的 total 重新清零，各输出行独立。

自测 `[[1,2,3]] @ [[1],[0],[-1]] → [[-2]]`。先预测 `(1,3)@(3,1)`，再写循环。可迁移自己的预备单元实现。

## 3. Q2：转置

`transpose(a)`：m×n 返回 n×m 的新列表。自测 `[[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]`。结果修改不得改变输入。

<details>
<summary>提示：两个循环的方向</summary>

输出第 j 行来自输入第 j 列；外层遍历输入列，内层遍历输入行。转置后元素数量不变。

</details>

## 4. Q3：逐行稳定 Softmax

`softmax_rows(x)` 每一行单独减 max、取指数、归一化。每行至少有一个有限值，其他项允许 -inf；空行、全 -inf 行、NaN 或 +inf 抛 `ValueError`。外层空列表返回 `[]`，mask 的 -inf 项输出概率为0。

自测 `[[0,0],[0,ln(2)]] → [[1/2,1/2],[1/3,2/3]]`。如果两行共同除以一个总和，形状虽对，概率含义已经错了。

## 5. Q4：Causal mask

`causal_mask(scores)` 接受方阵并返回新方阵，`j<=i` 保留，其余设为 `-math.inf`。非方阵抛 `ValueError`，`[]` 返回 `[]`。不得改原 scores。

三行允许列分别为 `[0]`、`[0,1]`、`[0,1,2]`。mask 应在 softmax 之前；否则直接清零后，每行和通常小于 1。

**Checkpoint A**：Q1–Q4 通过；手工填写第一节的 shape 表，并给一个错转置的反例。

## 6. Q5：组合单头 Attention

`attention(q,k,v,causal=True)`：先对空 Q 或空 K 返回 `[]`；否则要求 d>0、三者同 head_dim、V 非空且 K/V 同长度；causal 时 sq=sk，违反任一要求抛 `ValueError`。矩形输入只在 noncausal 模式合法；decode 的绝对位置偏移留待后续。

框架的职责顺序：

```text
检查 shape
scores = 组合 Q 与转置 K
scores 每项除以 sqrt(d)
按 causal 选择是否使用 mask
probs = 逐行归一化
返回 probs 与 V 的矩阵乘
```

测试不只检查 shape：还检查未来 V 改动不能影响过去的输出。调试时依次看 S、mask 后 S、P、O，最早出错的阶段才是修复目标。

## 7. Q6：Multi-head（独立组合）

`multi_head_attention(q,k,v,num_heads)`：先检查 num_heads 为正整数、三者 seq 相同，再对三者都为空返回 `[]`。非空输入须同 d_model>0 且能被 num_heads 整除；违反要求抛 `ValueError`。本题每头默认 causal，不包含投影 Wq/Wk/Wv/Wo。

每行 `[a,b,c,d]` 拆成两个 head 时，是 `[a,b]` 与 `[c,d]`，并不是交错的 `[a,c]` 与 `[b,d]`。每头缩放用自己的 `sqrt(d_head)`。

**能区分错误实现的例子**：Q=`[[0,0],[1,1]]`，K=`[[0,0],[1,2]]`，V=`[[2,10],[6,30]]`，heads=2。第一行输出 `[2,10]`；第二行分别为 `(2+6e)/(1+e)` 与 `(10+30e²)/(1+e²)`。直接返回 V、用 d_model 缩放、混合两个 head 都会失败。

**Checkpoint B**：Q5/Q6 通过；画出 split→单头→concat 的数据布局。

**修错练习**：有人将 Q5 中每行输出写为 `probs[i][0]*v[0][j]`，于是只有一个 token 的测试全部通过。用第一节两 token 的例子找出漏掉的求和项，再设计一个与示例不同的输入。将修复后测试写入自己的调试记录，解释它排除了哪个错误实现。

## 8. Q7：有效配对的 FLOPs

`attention_flops(b,h,sq,sk,d,causal=True)`，所有维度须为非负整数，否则 `ValueError`；causal 要求 sq=sk。每个配对的 QK 和 PV 两个点积按 4d FLOPs 计，忽略 softmax、缩放和投影。

- full：配对数 sq×sk。
- causal：配对数 `s(s+1)/2`，包含对角线。

例如 b=h=1,s=3,d=2，full=72 FLOPs，causal=48 FLOPs。先算满矩阵再 mask 的 Python 实现执行的是满矩阵工作；此函数输出的是可跳过上三角时的有效工作模型。

## 9. Q8：三阶段读写模型

`attention_bytes(...,bytes_per_elem=2,materialized=True)`：维度及 bytes_per_elem 必须是非负整数，否则 `ValueError`；所有张量按相同元素字节数计算。零字节仅是允许的数学零工作模型，不对应真实dtype。

| 数据 | 请求次数 | 元素数量 |
| :--- | ---: | ---: |
| Q/K/V 读、O 写 | 各一次 | b*h*(2sq+2sk)*d |
| S 写、S 读、P 写、P 读 | 共四次 | 4*b*h*sq*sk |

True 返回两行之和乘 bytes_per_elem；False 仅保留第一行，代表输入各读一次的理想下界。FlashAttention 的实际 K/V 重读由 tile 大小和遍历决定，不能由这个布尔值测出。

例如 b=1,h=8,sq=sk=128,d=64,2B：True=1572864B，False=524288B。一个 S 的容量是 262144B；S/P 的四次读写请求是它的四倍。

**Checkpoint C**：Q7/Q8 通过；用两个列名分别写“容量”和“请求流量”。

## 10. 出口检查

```bash
python3 course/tools/grade.py lab02
```

回答：为什么除 sqrt(d)？mask 后第 0 行为什么不会分母为零？有效 causal FLOPs 为什么不是严格一半？`materialized=False` 为什么不等于 profiler 的 HBM bytes？

**Optional**：读 FlashAttention [论文](https://arxiv.org/abs/2205.14135)，用教材中的 `(m,l,o)` 合并规则手算两个 tile。后续 GPU 实现以前述 Python 结果为 oracle。
