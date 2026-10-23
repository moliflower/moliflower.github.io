# 第 6 讲：FlashAttention 与 Decode Attention

> **课前**：40 分钟  
> **课后**：实验 2 Q7/Q8 + 画 IO 图  
> **一句话**：FlashAttention 不是“更快的 attention 公式”，而是“**不把 S×S 矩阵写回 HBM**”的 IO 优化；decode attention 则是“**用带宽换并行度**”的 split-KV 问题。

---

## 0. 本讲要解决的问题

**先修**：Lab1 的 `(m,s)` 不变式、Lab2 的 S/P/O shape 与有效 causal 配对。必会内容是第 2.1/2.2 的状态合并、2.3 的 IO 口径和 4.2 的 split 合并；无需先实现 CUDA kernel。

1. 标准 attention 的 HBM traffic 是多少？为什么长上下文会被 IO 卡住？
2. FlashAttention 如何用 tiling + online softmax 避免 materialize `S`？
3. 它的 HBM 访问量大概降到多少？block size 怎么选？
4. FlashAttention-2/3 分别改了什么？
5. decode attention 为什么 memory-bound？split-KV / FlashDecoding 解决什么问题？

---

## 1. 标准 attention 的 IO

标准实现：

```
S = Q K^T          # 写 S: sq x sk
P = softmax(S)     # 读 S，写 P: sq x sk
O = P V            # 读 P，写 O: sq x dv
```

对 `sq = sk = s`、单头、bf16：

```
S/P 各 s^2 个元素，每个 2B
写 S + 读 S + 写 P + 读 P ≈ 4 * s^2 * 2 = 8 s^2 bytes
```

`s=4096`：

```
8 * 4096^2 ≈ 134 MB
```

如果有 8 个头、batch 1，就是 1GB 级别；而 Q/K/V/O 本身只有几十 MB。  
**中间矩阵 S/P 是 IO 的主要来源。**

### 1.1 标准 attention 的复杂度

先分别数元素容量、请求流量和计算。单头同类型张量、每个输入读一次的三阶段模型：

```
Q/K/V/O 容量：4sd 个元素
S/P 同时保留时额外容量：2s² 个元素
S/P 写读请求：4s² 个元素
三阶段请求流量：(4sd + 4s²) * 元素字节数
两个矩阵乘的稠密 FLOPs：4s²d
```

因此该流量模型是 `Θ(sd+s²)` 个元素，计算量为 `Θ(s²d)`。输入重复读取、不同中间精度和缓存命中会改变实际 HBM 字节；模型中不计 Q/K/V 投影的 `Θ(sd²)` 计算。

---

## 2. FlashAttention 的核心

### 2.1 分块

把 Q 切成 `Br` 行，把 K/V 切成 `Bc` 列：

```
for each Q block Q_i (Br x d):
    m = -inf; l = 0; O = 0
    for each K/V block K_j, V_j (Bc x d):
        S_ij = Q_i K_j^T / sqrt(d)     # Br x Bc，留在 SRAM
        m_new = max(m, rowmax(S_ij))
        l = l * exp(m - m_new) + rowsum(exp(S_ij - m_new))
        O = O * exp(m - m_new) + exp(S_ij - m_new) V_j
        m = m_new
    O = O / l
```

关键：

- `S_ij` 只在 shared memory/寄存器，**不写 HBM**；
- `m, l, O` 是 running statistics/accumulator；
- 当 `m` 变大时，旧的 `l` 和 `O` 都要乘 `exp(m_old - m_new)` 重新缩放。

### 2.2 为什么是精确的

归纳不变式：处理完前 `j` 个 K/V block 后，

```
O / l = 前 j 个 block 的精确 softmax 加权和
```

新 block 到来时，旧的 `O` 和 `l` 用同一个标量 `exp(m_old - m_new)` 缩放，比例不变，再加入新贡献。这里 O 是未归一化分子，最终除以 l。在实数算术中它与稠密 attention 相同；浮点中归约顺序不同，允许小误差，不能要求 bitwise 相同。[FlashAttention 原论文](https://arxiv.org/abs/2205.14135)

**手算两个 tile**：一行 logits 为 `[0,ln(2),ln(4)]`，V 为 `[2,6,10]`。第一 tile 处理前两个：`m=ln(2),l=1.5,O=7`。第二 tile 的最大值是 ln(4)，旧状态都乘 1/2，得到 `l=1.75,O=13.5`；最终 `O/l=54/7`。直接稠密算是 `(1*2+2*6+4*10)/(1+2+4)=54/7`。

### 2.3 IO 复杂度

FlashAttention 将 S/P tile 留在片上。在论文采用的两级存储模型和适用 SRAM 容量区间中，访问量可写为：

```
Θ(s^2 d^2 / M + s d)
```

其中 `M` 以元素数计量，论文常用区间为 `d ≤ M ≤ sd`；结果也是元素访问数。固定片上容量时仍可随 s 二次增长。Q-major 遍历中每个 Q block 扫一遍 K/V，读写约 `2sd + 2sd*ceil(s/Br)` 个元素；容量允许 Br 随 M/d 增大，便能增加复用。收益是省掉完整 S/P 的物化并改善 IO 复用，不能把剩余 HBM IO 直接写成 O(sd)。

实际解释要同时说明：消除的是哪个中间数组、tile 遍历导致哪些输入重读、M 的单位是什么。先用 [第五讲](lec05.md) 的小 shape 区分容量与流量，再用本讲的分块过程数出重读次数。

### 2.4 Block size 的选择

论文给出（SRAM 大小 `M`、head dim `d`）：

```
Bc = ceil(M / (4d))
Br = min(ceil(M / (4d)), d)
```

直觉：

- `Bc` 越大，K/V 复用越多，但 shared memory 压力越大；
- `Br` 越大，Q 复用越多，但寄存器/occupancy 压力越大；
- 要保证 `Q_i`、`K_j`、`V_j`、`O`、`m/l` 都能放下。

### 2.5 Causal 的优化

当 Br=Bc 且 tile 同起点对齐时，列 tile 编号大于行 tile 编号的块完全不需要计算。对任意 Br/Bc，用绝对 token 起点判断更可靠：若 `key_start >= query_start + query_rows`，整个 tile 都在未来。

```
for j in 0..i:
    ...
```

对角线 block 需要 mask；非对角线 block 全部有效。这样 FLOPs 约减半，IO 也减少。

---

## 3. FlashAttention-2/3

### 3.1 FA2

- 减少非 matmul FLOPs（softmax、rescaling）；
- 更好的 work partitioning：Q 在 warp 之间切分，减少跨 warp 通信；
- 支持更大的 head dim；
- 在 sequence 维拆分 thread blocks，提高低 batch/head 情形的并行度。FA2 论文主要展示 A100 上的优化，不把 Hopper 的 TMA/WGMMA 管线作为其贡献。[FA2 论文](https://arxiv.org/abs/2307.08691)

### 3.2 FA3

- Hopper 的 WGMMA + TMA；
- warp-specialization：生产者 warp 负责 TMA，消费者 warp 负责 MMA/softmax；
- ping-pong scheduling：两个 warpgroup 交替，重叠 GEMM 和 softmax；
- FP8 支持：layout conformance、incoherent processing。

上述 Hopper 异步与低精度设计对应 [FA3 论文](https://arxiv.org/abs/2407.08608)。学习时应区分算法不变式和特定硬件上的实现优化。

### 3.3 面试表述

不要只说“FA2 更快”。要能说：

> FA2 减少了非 matmul 开销、改善了 warp 间的工作划分；FA3 利用 Hopper 的 WGMMA/TMA 和 warp specialization，把 GEMM 与 softmax 重叠，并用 FP8 提高吞吐。

---

## 4. Decode Attention

### 4.1 为什么 memory-bound

Decode 时 `sq=1`：

```
scores = q(1 x d) @ K^T(d x sk)   -> 1 x sk
O      = softmax(scores) @ V(sk x d) -> 1 x d
```

FLOPs：

```
4 * sk * d
```

Bytes（bf16，只算 K/V）：

```
2 * sk * d * 2 = 4 sk d
```

AI：

```
AI ≈ 1 FLOP/Byte
```

远低于拐点，memory-bound。

### 4.2 Split-KV / FlashDecoding

一个 decode 请求只有 1 个 query，GPU 的并行度不足：一个 block 处理一个 head，SM 利用率低。  
split-KV 把 KV 切成 R 个非空段，每段一个 block 计算 partial `(o_r,m_r,l_r)`；o_r 是以 m_r 为基准的未归一化加权和：

```
m = max_r(m_r)
l = Σ_r l_r * exp(m_r-m)
o = Σ_r o_r * exp(m_r-m)
O = o/l
```

如果各段保存的是已归一化输出 `O_r=o_r/l_r`，合并分子必须是 `Σ O_r*l_r*exp(m_r-m)`，不能丢掉 l_r。空段用中性状态并跳过，避免 `-inf-(-inf)`。

收益：

- 增加并行度，打满 SM；
- 每段只读自己的 KV 区间，增加独立工作；是否增加 L2 命中由布局与复用决定；
- 代价：最后 reduction 的通信/额外 kernel。

### 4.3 Paged KV

decode 的 K/V 不连续，按 block table gather：

```
physical = block_table[logical]
addr = base + physical * block_bytes + offset
```

这会让访存不连续，所以：

- block size 要够大（16/64/128）；
- kernel 要按 block 组织，尽量让一个 warp 读连续 token；
- L2 命中率很重要。

### 4.4 GQA/MQA/MLA

- GQA/MQA 减少 KV head 数，直接降低 bytes；
- MLA 存压缩 latent，进一步降低 bytes；
- 代价：投影计算/精度。

---

## 5. 代码：FlashAttention 前向伪代码

```python
def flash_attention(Q, K, V, Br, Bc, causal=True):
    # Q: [s, d], K/V: [s, d]
    O = zeros_like(Q)
    for i in range(0, s, Br):
        Qi = Q[i:i+Br]
        rows = len(Qi)  # 最后一个 query tile 可能不足 Br 行
        m = full((rows,), -inf)
        l = zeros((rows,))
        Oi = zeros((rows, d))
        for j in range(0, s, Bc):
            if causal and j >= i + rows:
                break
            Kj = K[j:j+Bc]
            Vj = V[j:j+Bc]
            S = Qi @ Kj.T / sqrt(d)          # rows x len(Kj)
            if causal:
                S = mask(S, i, j)
            m_new = maximum(m, S.max(dim=-1))
            l = l * exp(m - m_new) + exp(S - m_new[:, None]).sum(dim=-1)
            Oi = Oi * exp(m - m_new)[:, None] + exp(S - m_new[:, None]) @ Vj
            m = m_new
        O[i:i+Br] = Oi / l[:, None]
    return O
```

这是混合数组 API 的算法伪代码：`s,d` 来自 Q.shape；`mask(S,i,j)` 按全局行列位置屏蔽未来 key；zeros/full/maximum 由数组库提供，不能原样执行。输入为有限、等长非空自注意力，Br/Bc 为正；从 key0 顺序处理使每个有效 query 的首 tile 至少含一个允许的 key。在 GPU 上需另行 mask 尾块并满足同步规则。

### TileLang 结构

```python
with T.Kernel(T.ceildiv(S, Br), heads, threads=256) as (bx, bh):
    Qs = T.alloc_shared((Br, D), "float16")
    Ks = T.alloc_shared((Bc, D), "float16")
    Vs = T.alloc_shared((Bc, D), "float16")
    acc = T.alloc_fragment((Br, D), "float32")
    m = T.alloc_fragment((Br,), "float32")
    l = T.alloc_fragment((Br,), "float32")
    for j in T.Pipelined(T.ceildiv(S, Bc), num_stages=2):
        T.copy(K[j*Bc, 0], Ks)
        T.copy(V[j*Bc, 0], Vs)
        # S = Qs @ Ks^T; online softmax; acc += P @ Vs
```

具体 API 以 TileLang 版本为准；关键是 `Br/Bc`、online softmax、causal。

---

## 6. 仓库对照

- `deepinfer/model_executor/layers/attention_backend.py`：`prefill_attn_varlen_sparse`、`decode_mla_mqa_sparse`；看 `fused_norm_rope_attn_rope_cast` 如何把 norm/rope/cast 融进 attention。
- `vendor/FlashMLA`：prefill/decode 的 kernel 实现；先读 README 和接口。
- `.../models/v4_modules/mla_attn.py`：MLA 投影；`LoraRopeCompressFlashMHAAutoRecomp`。
- `.../models/v4_modules/index_attn.py`：sparse indexer。
- `.../worker/cache_engine.py`：paged KV 的布局，和 decode attention 的地址计算对应。

---

## 7. 练习

### 基础

**预测与修错**：两个 split 的 logits 都为 0，第一段含一个 V=0，第二段含三个 V=4。若把两段输出直接平均，得到 2；真实结果是多少？

<details><summary>核对</summary>

真实结果是 `(0+4+4+4)/4=3`。两段的 l 分别为 1 和 3，合并必须按 l 加权。再用第 2.2 节最大值不同的例子检查 exp 重缩放；两类反例一起才能检验正确的 split 合并。

</details>

1. 计算标准 attention 在 `s=512/2048/8192`、bf16、单头下的 S/P HBM bytes。
2. 推导 online softmax 的 `O` 和 `l` 的 rescale 公式。
3. 画一张 FlashAttention 的 `Q/K/V/S/O` 在 HBM/SRAM 之间的数据流图。
4. 计算 decode `s=4096, d=64, bf16` 的 KV bytes 和 AI。

### 进阶

5. 用 Python 实现第 5 节的 FlashAttention 伪代码，和标准 attention 对比误差。
6. 在 CUDA/Triton 中实现一个简化版（`Br=64, Bc=64`），测 `s=512/2048` 的耗时。
7. 实现 split-KV decode：`S=4` 段，比较 `s=4096` 时单 block 和 split 的耗时。
8. 用 `ncu` 看 `dram__throughput`、`sm__pipe_tensor_cycles_active`，判断 prefill/decode 的瓶颈。

### 挑战

9. 加 causal mask，验证对角线 block 的边界；比较 mask 版本和非 mask 版本的 FLOPs。
10. 读 FlashMLA 的 decode 接口，画出 paged KV + sparse index + MLA 的数据流。

---

## 8. 面试追问

1. 标准 attention 的 IO 为什么是 `O(s^2)`？FlashAttention 省了哪几次 HBM 读写？
2. online softmax 的 `m/l/O` 分别是什么？为什么旧值要 rescale？
3. FlashAttention 是精确的还是近似的？为什么？
4. FA2/FA3 各改了什么？
5. decode attention 为什么 memory-bound？split-KV 解决什么问题？
6. paged KV 对 attention kernel 的访存有什么影响？

---

## 9. 参考文献

- Dao et al., *FlashAttention*：<https://arxiv.org/abs/2205.14135>
- Dao, *FlashAttention-2*：<https://arxiv.org/abs/2307.08691>
- Shah et al., *FlashAttention-3*：NeurIPS 2024。
- FlashMLA：<https://github.com/deepseek-ai/FlashMLA>
- NVIDIA, *From Online Softmax to FlashAttention*（博客/讲义）。

---

## 10. 下一步

- 完成 [实验 2](../labs/lab02/README.md)；
- 做 [作业 2](../hw/hw02/README.md) 的 AI/bound 计算；
- 读 [第 7 讲](lec07.md) 的 KV cache。
