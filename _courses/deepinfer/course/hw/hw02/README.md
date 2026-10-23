# Homework2：Attention IO、分页和共享预算

> **先修**：Lab2/3、HW1、[L5 Attention](../../lectures/lec05.md)、[L6 流量分析](../../lectures/lec06.md)、[L7 KV cache](../../lectures/lec07.md)。  
> **投入**：5–7 小时。必做 Q1–Q11 与三个分析题。  
> **交付**：`hw02.py`、基于 [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md) 的 IO 箭头图、分页表与分析笔记。

## 0. 文件与依赖

框架为 `hw02.py`，测试为 `test_hw02.py`，参考为 `hw02_solution.py`。实现依赖为 Q1/Q2→Q3；Q5→Q6/Q10；Q7→Q8；Q5/Q7→Q9；Q9→Q11。Q4 输入独立给出的 AI，所以可以单独测试。复用前题，避免多处维护同一个公式。

```bash
python3 course/tools/grade.py hw02 --question q1 --unlock
python3 course/tools/grade.py hw02 --question q1 --failfast
python3 course/tools/grade.py hw02 --question q1 --hint 1
```

每题替换题号，先写预测再实现。Windows PowerShell 可将 `python3` 换为 `py -3`。数量参数为非负整数；AI 非负；page_size、硬件数值要求正；attention_ai 的分母必须正。违反这些要求抛 `ValueError`。数值保证有限，不要求检查类型；保留输入不变。每题测试只调用该题接口及已声明依赖，例如 Q7 不要求先写 Q8。

开始前，确认自己能解释 HW1 中的 K/V 系数 2、读出 Lab2 矩形 matmul 的 shape，以及 Lab3 中逻辑位置到页内偏移的映射。本作业会把这些模型组合成预算判断。

| 题号 | 修改函数 | 自查重点 |
| :--- | :--- | :--- |
| Q1 | attention_flops | 矩形 full、s=1 对角线、causal shape |
| Q2 | attention_bytes | 每张量的读写次数、dtype、零 batch |
| Q3 | attention_ai | 传递 causal/materialized 参数、零分母 |
| Q4 | bound | TFLOP/s 与 GB/s 单位、拐点平局 |
| Q5 | kv_cache_bytes | K/V、KV heads、batch、dtype 各轴 |
| Q6 | kv_cache_growth | 增量与总量的区别 |
| Q7 | num_pages | 0、page、page+1、大整数 |
| Q8 | page_waste | 恰好整页时的 0；浪费单位为 slot |
| Q9 | paged_kv_bytes | 每条序列单独保留尾页 |
| Q10 | prefix_cache_savings | 已有副本与新消费者的基线 |
| Q11 | paged_batch_bytes | 不同长度的独立向上取整 |

## 1. Q1–Q4：从数据流推导 bound

### Q1：attention_flops

接口支持矩形 `(sq,sk)`；full 配对数 sq×sk，causal 要求 sq=sk，配对数 s(s+1)/2；每配对 4d FLOPs，再乘 batch 和 heads。Q1 返回整数。s=3,d=2,b=h=1 时 causal=48。

### Q2：attention_bytes

在纸上画 `Q,K → S → P` 与 `P,V → O`。本题各张量按同一种 w 字节 dtype：Q/K/V 读一次、O 写一次，基础元素数 `b*h*(2sq+2sk)*d`；materialized=True 时加 S 写/读和 P 写/读，共 `4*b*h*sq*sk`，最后乘 w。

False 为输入各读一次的理想下界。真实 tiled attention 的 K/V 可能重复读取；此函数不模拟缓存。输出是请求流量模型，不是 S 的容量。

例子 b=1,h=8,sq=sk=128,d=64,w=2：True=1572864B，False=524288B。要求书面列出每次读取/写入，而不仅调用 Lab2 函数得到数字。

### Q3：attention_ai

复用 Q1/Q2 返回 FLOPs/Bytes，字节数为 0 时抛 `ValueError`。上例 full 的 AI 分别约 21.333 与 64 FLOP/B。causal 的有效工作模型要同时说明你是否仍保留了满矩阵的物化流量。

### Q4：bound

输入 ai、peak_tflops、bandwidth_gbps，拐点是 `peak*1e12/(bandwidth*1e9)`；ai>=ridge 返回 compute，否则 memory。等号时两条限制同时达到，本接口约定返回 compute。与 HW1 的平局标签不同不表示物理结论矛盾。

**Checkpoint A**：四题通过；能指出模型中哪些项会随 s 二次增长。

## 2. Q5–Q6：有效 KV 容量与增长

Q5 `kv_cache_bytes` 的参数与 HW1 相同，返回 `2*L*Hkv*D*S*B*w`。Q6 `kv_cache_growth` 返回每条序列新增 new_tokens 的增量：复用 Q5，以 new_tokens 代 S。

L=32,Hkv=8,D=128,w=2 时每 token=131072B；B=1，新增 100 token 增长 13107200B=12.5MiB。这里是存储增量；decode 本步读取已有 KV 的总量通常远大于此增量。

## 3. Q7–Q9：页数、尾页浪费与分配容量

Q7 `num_pages(seq_len,page_size)` 返回向上取整页数，使用整数运算；seq_len=0 返回 0。Q8 `page_waste` 返回分配 slot 数减有效 token 数。Q9 `paged_kv_bytes` 将 Q7 的分配 slot 数代入 Q5；每条序列独立分页。

完整例子 S=1000,page=16：62 个满页加一个尾页，共 63 页、1008 slot、waste=8。上述模型每 token=128KiB，因此分配容量 132120576B=126MiB，有效内容 125MiB。

边界自测：S=0、S=page、S=page+1；page=0 应抛异常。若 sequence 长度不同，分别向上取整后求和，不能先合并总长度再分配。

## 4. Q10：前缀共享省了哪几份副本

`prefix_cache_savings(...,prefix_len,batch=1,...)` 的 batch 特指**新消费者数**：已有一份相同前缀被保留，batch 条新请求避免再复制 batch 份。结果等于 Q5 中 S=prefix_len、B=batch 的有效容量。

P=512，其他参数同上，一份=64MiB；3 个新消费者避免 192MiB 额外副本。如果“总共三个请求从三份压成一份”，省的是 128MiB。必须先确定基线。这里按 token 理想共享；整页共享时 P 应先向下取整到完整页。

**Checkpoint B**：Q5–Q10 通过；能解释有效容量与分配容量。

## 5. Q11：变长批次 — 独立迁移

实现 `paged_batch_bytes(seq_lens,page_size,num_layers,num_kv_heads,head_dim,bytes_per_elem=2)`。输入是各请求的有效长度列表；每条请求独立分配页，不允许共享尾页。复用 Q9，返回所有请求合计的分配字节。保持 `seq_lens` 不变，空列表返回 0。即使没有请求，page_size 和模型参数仍须满足前述契约。

例子：`seq_lens=[1,16,17]`、page=16、L=Hkv=D=1、w=2 时，每个 slot 是 4B，总共分配 64 slot，结果为 256B。把总长度 34 一次向上取整会错误得到 192B；错误来自模型，不是浮点精度。

实现前先画三行分页表，不给出代码骨架。完成后验证两个性质：交换请求顺序不改变容量；等长列表的结果与 Q9 的 batch 版本一致。**Checkpoint C**：Q11 与 W2 通过，同时能解释这两个性质为什么必要，但还不足以保证任意实现正确。

## 6. Required：分析题

**W1**：为 s=128 与 s=4096、b=1,h=8,d=64,w=2 计算 full FLOPs、两种流量、两种 AI。在假设 P=100TFLOP/s、BW=1000GB/s 下分类。哪些数值只是下界？如 Q tile=64，按每个 Q tile 重新读取 K/V，另算 tiled 请求流量并比较。

**W2**：三条请求长 `[1,16,17]`，page=16。逐条写页数、slot、waste；与“34 token 合起来分页”比较。说明后者为什么不符合每序列独立分配的接口。

**W3**：已有一份长度 35 的前缀，page=16，两个新请求共享完整页。求实际可共享 token 数、避免的 slot 数。若要共享未满尾页，画出 append 时 copy-on-write 的状态变化。

<details>
<summary>自查锚点</summary>

W1 full FLOPs 为 33554432、34359738368。s=4096 时基础 Q/K/V/O=16777216B，三阶段=1090519040B，读一次下界 AI=2048，三阶段 AI≈31.508。W2 分别 1/1/2 页，总 64 slot，waste=30；合并分页只得 48 slot。W3 共享 32 token，两个新请求避免 64 token slot 的副本。

</details>

## 7. 验收与选做

```bash
python3 course/tools/grade.py hw02
```

checkoff：写出容量、流量、FLOPs 的单位；解释 Q2 False 的条件；解释 prefix_savings 中 batch 的含义。提交代码和推导记录，保留使用提示的题号。

验收证据：Q1–Q11 的公开测试通过；W1 列出完整 IO 账本并标明下界；W2 的逐请求分配与 Q11 对照；W3 明确共享前后的副本数。若测试失败，先看分题反馈，再运行对应题号；`未完成` 不表示正确，只表示测试未全部执行或有跳过。

**Optional**：设定可用于 KV 的独立预算，比较 page=16/128 下不同长度分布的并发容量。将权重、临时 workspace 和 KV 预算列为不同项，避免把整张 8GB 显存都当作 KV。
