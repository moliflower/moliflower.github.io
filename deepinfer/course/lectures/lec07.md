# 第 7 讲：KV Cache 基础

> **课前**：35 分钟  
> **课后**：实验 3 + 作业 2  
> **一句话**：KV cache 是“用显存换计算”；PagedAttention 是“用分页换碎片和共享”；KV 量化是“用精度换带宽”。

---

## 0. 本讲要解决的问题

**先修**：列表副本、整除/取余、Lab2 的 K/V；类与 self 可在 Lab3 第 0 节现学。必做第 1–2 节与 Lab3。prefix 共享、SWA、MLA 是后续框架阅读，不要求初学者一次实现全部机制。

1. 一条序列的 KV cache 到底占多少字节？batch、seq、层数、head 数怎么影响？
2. 为什么连续布局会碎片化？paged 布局的 block table 怎么工作？
3. prefix cache 能省多少？命中率受什么影响？
4. eviction / SWA / bounded replay 分别解决什么问题？
5. MLA 为什么能把 KV 压得更小？代价是什么？

---

## 1. KV Cache 的容量模型

### 1.1 公式

标准 MHA/GQA：

```
KV_bytes = 2 * L * H_kv * D * S * B * bytes_per_elem
```

- `2`：K 和 V；
- `L`：层数；
- `H_kv`：KV head 数（GQA 时小于 Q head）；
- `D`：head_dim；
- `S`：序列长度；
- `B`：batch；
- `bytes_per_elem`：bf16=2，FP8=1。

每 token、每层：

```
2 * H_kv * D * bytes_per_elem
```

### 1.2 例子

`L=32, H_kv=8, D=128, S=2048, B=1, bf16`：

```
2 * 32 * 8 * 128 * 2048 * 2
= 268,435,456 bytes
= 256 MiB
```

`B=32`：8 GiB。`S=8192`：1 GiB/条。  
所以长上下文/大并发下，KV cache 往往比模型权重更先耗尽显存。

### 1.3 容量规划

给定显存 `M_gpu`、模型权重 `M_weights`、激活/workspace `M_act`：

```
M_kv = M_gpu * utilization - M_weights - M_act
max_tokens_ideal = floor(max(0, M_kv) / (2 * L * H_kv * D * bytes))
```

如果所有序列长度都等于 S，可用 `floor(max_tokens_ideal/S)` 估计并发上界。仅知道平均长度 `S_avg` 时，下面只是规划估计，不是保证可容纳每一批请求的上限：

```
B_max ≈ max_tokens / S_avg
```

注意：这是**总 token 数**上限，不是请求数上限；paged 布局下，碎片和 block table 也会占空间。

---

## 2. 连续布局 vs Paged 布局

### 2.1 连续布局

```
seq 0: [block 0][block 1][block 2]...
```

问题：

- 需要按 `max_model_len` 预留，实际长度远小于最大值 -> 内部碎片；
- 多条序列长度不同 -> 外部碎片；
- 扩容要搬数据；
- 前缀共享困难。

vLLM 论文提到，传统连续布局的显存有效利用率可能只有 20%–40%。

### 2.2 Paged 布局

```
逻辑 block 0 -> 物理 block 7
逻辑 block 1 -> 物理 block 2
逻辑 block 2 -> 物理 block 9
```

- 固定 `block_size`（常见 16/64/128）；
- `block_table[logical] = physical`；
- 物理 block 可以不连续；
- 按需分配，最后一块可能浪费；
- 前缀可以共享物理 block（copy-on-write）。

### 2.3 地址计算

对 token 位置 `pos`、层 `l`、head `h`、维度 `d`：

```
logical = pos // block_size
offset  = pos % block_size
physical = block_table[logical]
addr = base
     + physical * block_bytes
     + l * layer_stride
     + h * head_stride
     + offset * token_stride
     + d * elem_bytes
```

例如每层的 K 和 V 分别有一个连续数组，单个 K 数组的布局是 `[num_blocks, block_size, H_kv, D]`，则 base 已经选定层与 K/V，不再叠加 layer_stride：

```
block_bytes = block_size * H_kv * D * bytes
token_stride = H_kv * D * bytes
head_stride = D * bytes
addr = base + physical*block_bytes + offset*token_stride + h*head_stride + d*bytes
```

**手算**：block_size=2、H_kv=2、D=3、元素2B、table=`[7,2]`。pos=3、h=1、d=2 ⇒ logical=1、offset=1、physical=2。block_bytes=24、token_stride=12、head_stride=6，地址相对 base 偏移为 `2*24+1*12+1*6+2*2=70B`。逻辑 token3 并不在物理页3。

### 2.4 Page size 的 trade-off

| 小 page（16） | 大 page（128） |
| :--- | :--- |
| 内部碎片小 | 最后一块浪费大 |
| block table 项多，管理开销大 | block table 项少 |
| 前缀共享粒度细 | 共享粒度粗 |
| gather 更碎，kernel 开销大 | gather 更连续，带宽好 |

选择要看 workload：短序列多 -> 小 page；长序列/共享前缀多 -> 大 page。

---

## 3. Prefix Cache

### 3.1 收益

如果 B 条请求共享同一个长度 P 的前缀，原来存 B 份，现在只存一份。相较独立缓存，理想节省：

```
saved = 2 * L * H_kv * D * P * (B-1) * bytes
```

例：`P=512, L=32, H_kv=8, D=128, B=2, bf16`，一份前缀占64MiB，两条共享节省一份：

```
64 MiB
```

对长 system prompt / RAG / agent 历史，收益很大。

若只共享完整页，将 P 换为 `floor(P/block_size)*block_size`。B=1 时共享节省为0；仍可能通过缓存命中节省下一次 prefill 计算，两种收益不要混算。

### 3.2 命中判断

理想情况：按 token 序列的 hash 做前缀匹配：

```
hash(prefix[0:P]) -> block list
```

工程上：

- 以 block 为粒度做 hash（block hash + 父 block hash）；
- 命中后复用物理 block，引用计数 +1；
- 写入时 copy-on-write（如果 block 被共享）；
- 淘汰时引用计数为 0 才回收。

复用还必须匹配模型权重、adapter、位置编码和其他影响 KV 的配置；token 一样但模型不同，不能直接共享结果。

### 3.3 命中率的影响

- 前缀越长、重复越多，命中率越高；
- 路由策略：如果按 prefix affinity 路由，命中率高但负载不均；DeepInfer 选择“任意 prefill 机器都能命中任何前缀”，用集中式 KV store；
- TTL：3 天 vs 7 天，对 chatbot 长尾命中率有 0.5%–1% 的影响（Q&A 数据）。

---

## 4. Eviction、SWA 与 Bounded Replay

### 4.1 Eviction

当 KV 放不下：

- LRU：淘汰最久未用的 block；
- 按序列优先级：短请求/高价值请求保留；
- 换出到 host/disk：swap，代价是 PCIe/RDMA；
- 重算：preemption 后重新 prefill，代价是算力。

### 4.2 SWA

对于本身定义为 Sliding Window Attention 的层，若只保留最近 W 个 token 的 KV：

```
KV_bytes ∝ W  而不是 S
```

该层 cache 随 W 有界；混合全局层的模型总容量仍可能随 S 增长。不能把全局 attention 的旧 KV 任意丢弃并称为等价优化；那会改变模型计算。

### 4.3 Bounded Replay

SWA 丢了旧 KV 后，如果命中前缀但 SWA 缺失，可以只重算最近一段：

```
replay 最近 128 token -> 近似恢复 SWA
```

DeepInfer 的 encoder/decoder SWA bounded replay：

- encoder：命中 block KV，但 SWA 缺失时，重算最后 128 token；
- decoder：只重算最后 128 token 的 SWA，block KV 从 encoder 最后一层投影；
- 代价是常数级，不随上下文长度增长。

### 4.4 State Cache

SWA 不放在 paged block 里，而是每个序列一个固定大小的 `StateBlock`：

```
[stored SWA region][dropped region]
```

- 以 128 token 为粒度存储；
- 环形 buffer，多留 6 个位置吸收 speculative 的越界；
- 只存前 20 层（V4.1 配置），减少 state 大小。

---

## 5. MLA：更小的 KV

MLA 存压缩 latent：

```
c_KV = x W_DKV          # [d_c]
K = c_KV W_UK, V = c_KV W_UV
```

KV cache：

```
bytes = L * (kv_lora_rank + qk_rope_head_dim) * S * B * bytes
```

没有 factor 2，因为 K/V 共享 latent。

例：`L=32, rank=512, rope=64, S=2048, B=1, FP8`：

```
32 * 576 * 2048 * 1 = 37,748,736 bytes ≈ 36 MiB
```

对比第 1.2 节 GQA bf16 的 256MiB，小约7.11倍；这里同时改变了缓存表示与 dtype。若同为 bf16，MLA 为72MiB，仅由表示变化带来的比值为256/72≈3.56。例子未计 scale、对齐及分层混合精度。

代价：

- attention 前向多了一次投影；
- latent 的精度/表达受限；
- 需要专门的 FlashMLA kernel。

---

## 6. 仓库对照

- `deepinfer/worker/cache_engine.py`：`KVCache`、`StateBlock`、`KVCacheManager`、`GPUCacheInterface`；看 `per_token_bytes` 和 `get_kvcache_layout`。
- `.../tilelang_ops/cuda/load_paged_kv_cache.py`：paged gather 的地址计算。
- `.../tilelang_ops/cuda/store_kv_cache.py`：写入。
- `.../tilelang_ops/cuda/state_cache.py`：SWA 环形 buffer。
- `.../tilelang_ops/cuda/compress.py`：压缩 KV 的更新。

---

## 7. 练习

### 基础

**预测与修错**：page_size=4，长度从4增长到5，`length//page_size` 为什么会少分配一页？

<details><summary>核对</summary>

4个 token 占1页，5个占2页；需向上取整 `(length+page_size-1)//page_size`。在 Lab3 中只在 append 前的 length 是页大小倍数时分配新页。随后独立 hand trace 非连续页表的 get(1,5)，并验证修改返回值不会改缓存。

</details>

1. 计算 `L=32, H_kv=8, D=128, S=4096, B=4, FP8` 的 KV bytes。
2. 同样配置，bf16 是多少？MLA（rank=512, rope=64, FP8）是多少？
3. `seq_len=5000, page_size=64`，需要多少 page？浪费多少 slot？
4. 推导 paged 地址计算的每一步。

### 进阶

5. 实现 `PagedKVCache`（实验 3），支持 append/get/free。
6. 写一个 prefix cache 模拟器：随机请求、共享前缀、LRU 淘汰，测命中率 vs 容量。
7. 扫 page size `16/32/64/128`，测 gather 带宽和 waste。
8. 实现 `share_prefix` + copy-on-write，验证写入不会污染共享 block。

### 挑战

9. 用 `ncu` 看 paged gather 的 sector 利用率，解释为什么 page size 影响带宽。
10. 模拟 SWA bounded replay：给定前缀命中长度，计算重算 token 数和额外 FLOPs。

---

## 8. 面试追问

1. KV cache 大小怎么算？为什么 MLA 不乘 2？
2. PagedAttention 解决什么问题？block table 怎么工作？
3. prefix cache 的命中率受什么影响？怎么做 eviction？
4. SWA 和 bounded replay 分别解决什么问题？
5. page size 怎么选？小 page 和大 page 各有什么代价？
6. 如果 KV cache 放不下，你会 swap 还是 recompute？为什么？

---

## 9. 参考文献

- Kwon et al., *PagedAttention / vLLM*：<https://arxiv.org/abs/2309.06180>
- vLLM blog：<https://blog.vllm.ai/2023/06/20/vllm.html>
- DeepSeek-V2：MLA；DeepSeek Q&A：KV cache storage / SWA。
- DeepInfer `doc/inference_deployment_qa.md`。

---

## 10. 下一步

- 做 [实验 3](../labs/lab03/README.md)；
- 做 [作业 2](../hw/hw02/README.md) 的 KV 计算；
- 读 [第 8 讲](lec08.md) 的量化。
