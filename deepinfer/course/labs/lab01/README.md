# Lab1：归约、稳定 Softmax 与归一化

> **阅读**：[L3 数值算子](../../lectures/lec03.md)；第一讲与 Lab0。需要会循环、列表、指数和平方根。  
> **投入**：3–5 小时，可分三次。必做 Q1–Q7；GPU 映射为选做。  
> **交付**：`lab01.py`、自己的调试记录与 checkoff 回答。

## 0. 文件与工作流

`lab01.py` 是待完成函数框架，`test_lab01.py` 是公开规格，`lab01_solution.py` 是参考解法。只修改学生文件，使用标准库。可以复制 [记录模板](../../WORKSHEET.md) 到自己的笔记。

**共同数值契约**：向量中的数值有限，且本题中的平方/求和中间值能被 Python float 表示；不要求实现任意大数或 NaN/Inf 输入处理。Python float 通常是64位，本 Lab 的4B流量模型假设真实算子使用float32，两者用途不同。所有函数保留输入；计数参数是整数，`True/False` 不作整数计数。

从**课程根目录**运行，即直接包含 `course` 的目录；离线包解压后进入这一层，个人主页源仓库先 `cd _courses/deepinfer`。Windows安装的解释器名若为 `python`，替换 `python3`，或用 `.\course\run.ps1 lab01 --question q1`：

```bash
python3 course/tools/grade.py lab01 --list
python3 course/tools/grade.py lab01 --question q1 --unlock
python3 course/tools/grade.py lab01 --question q1 --failfast
python3 course/tools/grade.py lab01 --question q1 --hint 1
```

每题都支持同样的四条命令，替换题号即可。`--hint 2/3` 逐层提供实现帮助。先预测再编码；通过单题后运行整个 Lab。

### 从教材走到题目

| L3阅读段落 | 本Lab题目 | 写代码之前先交给自己 |
| :--- | :--- | :--- |
| 第1–2节：树形与浮点顺序 | Q1 | 长度7的每轮列表和加法次数 |
| 第3–5节：正权重、减max、online状态 | Q2/Q3 | `[0,ln2,ln4]` 的m/s表和概率 |
| 第6节：均方与epsilon | Q4 | 平方→平均→开根→逐项weight |
| 第7节：偏差与方差 | Q5 | `[1,2,3]` 的均值和总体方差 |
| 第8节：逐次读写账本 | Q6/Q7 | n=4且含/不含weight的字节表 |

Q1/Q2是状态算法，Q3复用Q2；Q4依赖Q1。若后题失败，先检查被调用的前题，而不是直接修改期望。教材的简短softmax例子用sum帮助解释数学，学生Q1仍必须实现相邻树形，不能以这个示例解除算法约束。

## 1. Review：状态、结果与一次完整示例

归约将多个值合并为一个值。`[2,-1,4]` 成对相加：第一轮 `[1,4]`，第二轮 `[5]`。当前列表是状态，最终的 5 才是返回值。浮点加法顺序会影响舍入，因此“总和一样”只在精确实数算术中成立。

先预测下面程序打印什么：

```python
x = [2, -1, 4]
y = [x[0] + x[1], x[2]]
print(x, y)
```

<details>
<summary>检查预测</summary>

输出 `[2, -1, 4] [1, 4]`。原输入不变；下一轮可以把 y 作为新的状态。

</details>

## 2. Q1：相邻成对归约

**接口**：`sum_reduce(x) -> float`。输入有限数值列表，允许空列表。不改变输入。空输入返回 `0.0`。

**算法约束**：每轮相邻两项相加，奇数尾项保留；不能用内置 `sum`、`math.fsum` 或串行累加代替本题。框架如下，先在纸上补全索引再写入学生函数：

```text
values = 输入的浮点副本
while 当前长度大于 1:
    next_values = []
    for 每一对相邻元素:
        追加这对的和
    如果有未配对尾项: 追加尾项
    values = next_values
返回唯一元素；空输入单独处理
```

**检查点**：`[1,2,3,4,5] → [3,7,5] → [10,5] → [15]`。测试还会用 `[1e16,1,-1e16,1]` 区分相邻树归约和其他加法顺序；这不是要求更高精度，而是验证指定算法。

## 3. Q2：Online softmax 状态

**接口**：`online_softmax_state(x) -> (m,s)`。输入有限数值；空输入为 `(-math.inf,0.0)`。不修改 x。

读过一个前缀后维持：`m=max(前缀)`，`s=Σexp(v-m)`。新元素 v 到来时，先确定 `m_new=max(m,v)`；旧和换到新基准需要乘 `exp(m-m_new)`，然后加入 `exp(v-m_new)`。

完整小例子 `x=[0,ln(2)]`：

| 已处理 | m | s |
| :--- | :--- | :--- |
| 无 | -inf | 0 |
| 0 | 0 | 1 |
| ln(2) | ln(2) | 1×1/2+1=1.5 |

**修改位置**：只填此函数的状态循环。注意赋值顺序，更新 s 时需要旧 m。空状态直接进入有限首元素时，`exp(-inf)=0`。

**验收**：`[1,2,3]` 得到 `(3,exp(-2)+exp(-1)+1)`。本题一遍读取求状态，仍未生成完整概率数组。

再用 `[ln4,0,ln2]` 检查“max不变”的分支：m一直是ln4，s依次1、1.25、1.75。只测试递增输入不能检查所有状态变化；只测试相等输入又可能放过“先覆盖旧m”的错误。

## 4. Q3：用状态生成概率

**依赖**：Q2。`softmax(x) -> 新列表`，空输入返回 `[]`。用状态中的 m/s 归一化每个输入。两遍算法允许再次读取 x。

自测 `[0,ln(2)] → [1/3,2/3]`、`[1000,1000] → [0.5,0.5]`。Python `math.exp(1000)` 会抛 `OverflowError`，因此必须先减 max；不要假定所有运行环境都返回 inf。

**通过条件**：输出非负、和接近 1、平移输入不改变结果。只返回均匀概率也会使“和为 1”通过，所以还要测试非均匀输入。

**修错练习**：某实现先执行 `m=m_new`，再用 `exp(m-m_new)` 缩放 s。请用 `[0,ln(2)]` 写下它的错误 s，再只调整赋值顺序修复。记录“输入→第一个错误状态→预期状态”，不要只写“测试没通过”。

## 5. Q4：RMSNorm

**接口**：`rmsnorm(x,weight,eps=1e-6)`。eps 必须有限且为正，否则抛 `ValueError`；长度不匹配也抛 `ValueError`。先完成校验，再对两者为空返回 `[]`。用 Q1 归约，保留输入。

`mean_sq=Σx²/n`，`y_i=x_i*weight_i/sqrt(mean_sq+eps)`。对 `[3,4]`，平方和 25、均方 12.5。不要开平方后再加 eps。

**独立测试**：输入全零和非均匀 weight。忽略 eps 时，将 x 乘正数 2 应得到相同归一化结果；将 x 加 2 一般会改变结果。

## 6. Q5：LayerNorm（独立迁移）

**接口**：`layernorm(x,weight,bias,eps=1e-5)`。三列表等长，其他契约同 Q4。独立写出均值、总体方差、仿射变换，不直接粘贴 RMSNorm。

对 `[1,2,3]`，均值 2，偏差 `[-1,0,1]`，方差 `2/3`，不是样本方差 1。结果为 `偏差/sqrt(2/3+eps)*weight+bias`。

**完成条件**：常量输入归一化部分为零，最终输出等于 bias。Q4/Q5 的简单实现分别有一条和两条归约；实际 kernel 可用 Welford 等方法改变 pass 数。

## 7. Q6–Q7：明确口径的请求流量

**Q6** `softmax_traffic(n,passes=3)`：n 为非负整数，passes 只允许 1/2/3，否则 `ValueError`。按 float32、输出写一次计。x 读 passes 次，因此 `(passes+1)*4n` B。1-pass 假设整行保留在片上；2-pass 使用 online state 后再读 x。

**Q7** `rmsnorm_traffic(n,on_chip=True)`：n 为非负整数，否则 `ValueError`。只数 x/y 的 activation 请求；on-chip 是 `8n` B，两遍是 `12n` B。若 weight 也需从显存读取一次，两者都加 `4n` B。Q6 的 passes 也必须是整数，不能用2.0或True代替。

手算 n=4096：Q6 三遍 65536B、两遍 49152B、一遍 32768B；Q7 两遍 49152B、一遍 32768B。除以假设带宽只能得到模型时间下界；Python 列表、缓存和实际 GPU 时延另计。

## 8. Checkpoint 与提交

- **A**：Q1–Q3 通过；能手算 online state。
- **B**：Q4/Q5 通过；能区别均方与方差。
- **C**：Q6/Q7 通过；每个字节数都写明是否含 weight。

```bash
python3 course/tools/grade.py lab01
```

在笔记回答：树归约有多少轮和多少次加法？softmax 更新 m 后为什么要重缩放旧 s？固定 epsilon 时 RMSNorm 是否严格缩放不变？记录一个你自己设计的失败例子。

**迁移核对**：x=`[3,4]`、weight=`[1,2]`、eps=0.5，RMSNorm应为 `[3/sqrt13,8/sqrt13]`。LayerNorm对x=`[1,2,3]`、weight全1、bias全0、eps=1/3，应为 `[-1,0,1]`。这两个例子各检查了哪些步骤？输入全零虽然重要，但不能替代非均匀weight或方差分母检查。

## 9. Optional：GPU 映射

完成 C++/CUDA 桥接后，将“一行一个 block”画为线程分工图。标记归约、同步、输入驻留位置。说明行太大时需要的额外 pass。此扩展不计入本 Lab 的 Python 必做题。
