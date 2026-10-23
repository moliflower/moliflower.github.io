# 第 5 讲：从零实现 Attention

> **课前**：35 分钟  
> **课后**：实验 2  
> **一句话**：Attention 的计算是 `softmax(QK^T/√d)V`；工程上的全部难点，是 **shape、mask、数值稳定性和 IO**。

---

## 0. 本讲要解决的问题

**先修**：预备矩阵练习与 Lab1 stable softmax；能解释矩阵的一行/一列和点积。不需要先会 PyTorch。主线先完成第 1–3 节及 Lab2；GPU、GQA/MLA 为理解框架时的第二遍阅读。

1. 给定 `Q/K/V` 的 shape，写出 attention 的完整计算图和 FLOPs。
2. causal mask 为什么在 softmax 之前？mask 的边界怎么处理？
3. multi-head 的参数量、计算量、内存布局是什么？
4. prefill 和 decode 的 attention，分别受什么限制？AI 各是多少？
5. GQA/MQA/MLA 改变了什么？为什么能省 KV？

---

## 1. 形式化定义

给定：

```
Q: [sq, d]
K: [sk, d]
V: [sk, dv]
```

单头 attention：

```
S = Q K^T / sqrt(d)          # [sq, sk]
P = softmax_row(S)            # [sq, sk]
O = P V                       # [sq, dv]
```

- `sqrt(d)`：让点积的方差回到 1 附近，避免 softmax 饱和；
- `softmax_row`：对每一行做 softmax；
- causal：同起点自注意力中 `S[i,j] = -inf for j > i`，softmax 后为 0。Lab2 进一步限制 causal 输入 sq=sk、dv=d。

**一遍完整手算**：Q=`[[0],[0]]`，K=`[[1],[2]]`，V=`[[2],[6]]`，d=1。S 为 2×2 的零矩阵；causal 后 `[[0,-inf],[0,0]]`；P=`[[1,0],[1/2,1/2]]`；O=`[[2],[4]]`。第一行输出只有一个值，是已看到的 V 的加权和；P 的列数是 token 数，不是 head_dim。

### 1.1 为什么要除 sqrt(d)

假设 `q_i, k_j ~ N(0,1)` 独立，则：

```
q·k = Σ_{p=1}^d q_p k_p
Var(q·k) = d
std(q·k) = sqrt(d)
```

不除，`d=128` 时点积的标准差约 11.3，softmax 会非常尖；除以 `sqrt(d)` 后方差回到 1。

### 1.2 Mask 的语义

causal mask：

```
S[i,j] = -inf if j > i else S[i,j]
```

softmax 后：

```
P[i,j] = 0 for j > i
```

注意：

- mask 必须在 softmax **之前**；先 softmax 再置 0 会破坏归一化（分母仍然包含未来位置）。
- 全 -inf 行会出现无效分母，例如叠加 padding mask 后没有有效 key。Lab2 明确报错。
- decode 的 query 有绝对位置 `past+i`：应 mask `j>past+i`，不能套用 `j>i`。例如 sq=1、sk=5 的末 token 应看全部 5 个 key，而 `j>0` 只让它看第一个。一般矩形 causal 需要定义对齐；Lab2 只接受方形 causal，矩形只做 noncausal。

---

## 2. Multi-head Attention

### 2.1 定义

```
d_model = h * d_head
Q_h = Q W_h^Q,  K_h = K W_h^K,  V_h = V W_h^V
head_h = attention(Q_h, K_h, V_h)
O = concat(head_1..head_h) W^O
```

参数量（单层，忽略 bias）：

```
W^Q: d_model * d_model
W^K: d_model * d_model
W^V: d_model * d_model
W^O: d_model * d_model
total: 4 * d_model^2
```

GQA/MQA 会减少 `W^K/W^V` 的 head 数：

```
W^K: d_model * (h_kv * d_head)
W^V: d_model * (h_kv * d_head)
```

### 2.2 计算量

单头、单 batch：

```
QK^T: 2 * sq * sk * d
AV:   2 * sq * sk * dv
```

multi-head：

```
FLOPs = 2 * b * h * sq * sk * (d + dv)
```

通常 `d = dv`：

```
FLOPs = 4 * b * h * sq * sk * d
```

causal（`sq=sk=s`）的有效配对是 `1+2+...+s=s(s+1)/2`。可跳过上三角时的 FLOPs 为 `4*b*h*d*s*(s+1)/2`，长序列时才近似 full 的一半。先做完整 matmul 再 mask 的参考代码仍执行稠密工作。

### 2.3 示例

`b=1, h=8, s=128, d=64`：

```
full:   4 * 1 * 8 * 128 * 128 * 64 = 33,554,432
causal: 4 * 1 * 8 * 64 * (128*129/2) = 16,908,288
```

`b=1, h=8, s=4096`：

```
full: 4 * 8 * 4096^2 * 64 = 34.36 GFLOP
```

---

## 3. Prefill vs Decode

### 3.1 Prefill

- `sq = sk = prompt_len`，通常几百到几十万；
- QK^T 和 AV 都是大矩阵乘；
- 长 prompt 的理想 AI 较高，优化良好的实现更可能接近计算瓶颈；物化 S/P、低利用率或小尺寸会改变结论；
- KV cache 写入是 O(sk) 的带宽。

### 3.2 Decode

- `sq = 1`，`sk = 当前序列长度`；
- 每步只算一个新 token，但要看整个 KV；
- QK^T 产生 `1×sk` scores，PV 产生 `1×d` 输出；
- **memory-bound**：主要成本是把 K/V 从 HBM 读出来。

### 3.3 AI 的粗略估计

Decode、单头、单 batch、bf16：

```
FLOPs ≈ 4 * sk * d
Bytes ≈ 2 * sk * d * 2 = 4 sk d   （只算 K/V；Q/O 很小）
AI ≈ 1 FLOP/Byte
```

远低于拐点，memory-bound。

Prefill、`sq=sk=s`：

```
FLOPs ≈ 4 s^2 d
Bytes_lower = Q读 + K读 + V读 + O写 = 4*s*d*2 = 8sd
AI_upper(full) = 4s²d / 8sd = s/2
AI_upper(causal有效工作) = 2s(s+1)d / 8sd = (s+1)/4
```

这是假设 bf16 输入/输出各访问一次、不物化 S/P 的理想模型；`s=4096` 的 full AI 上界为 2048 FLOP/B，s=128 为 64。实际 tiled attention 可能反复读 K/V，AI 会下降。与匹配 dtype 的 ridge 比较只能给模型判断，最终须测量。

> 这就是为什么长上下文 prefill 要 FlashAttention，而 decode 要 KV cache/量化/split-KV。

---

## 4. 代码：从 PyTorch 到 GPU

### 4.1 PyTorch 参考

```python
import math
import torch

def attention(q, k, v, causal=True):
    # 此示例只支持同起点、等长 causal；decode 需要位置偏移。
    if causal and q.size(-2) != k.size(-2):
        raise ValueError("causal example requires equal sequence lengths")
    d = q.size(-1)
    s = q @ k.transpose(-2, -1) / math.sqrt(d)
    if causal:
        mask = torch.triu(torch.ones(s.size(-2), s.size(-1), device=q.device), diagonal=1).bool()
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return p @ v
```

### 4.2 CUDA：一个 block 一行

下面是索引算法片段，限定 sq=sk>0、d>0，并依赖向整个 block 广播结果的 `block_reduce_max/sum`；没有 host 驱动，不可直接编译。先用 Lab2 的 Python 输出作 oracle。

```cuda
__global__ void attention_row(const float* Q, const float* K, const float* V,
                              float* O, int sq, int sk, int d) {
    int row = blockIdx.x;
    const float* q = Q + row * d;
    float* o = O + row * d;

    // 1) score + max
    float m = -INFINITY;
    for (int j = threadIdx.x; j < sk; j += blockDim.x) {
        float s = 0.0f;
        for (int p = 0; p < d; ++p) s += q[p] * K[j * d + p];
        s /= sqrtf((float)d);
        if (j > row) s = -INFINITY;
        m = fmaxf(m, s);
    }
    m = block_reduce_max(m);

    // 2) exp + sum
    float sum = 0.0f;
    for (int j = threadIdx.x; j < sk; j += blockDim.x) {
        float s = 0.0f;
        for (int p = 0; p < d; ++p) s += q[p] * K[j * d + p];
        s /= sqrtf((float)d);
        if (j > row) s = -INFINITY;
        sum += expf(s - m);
    }
    sum = block_reduce_sum(sum);

    // 3) output
    for (int p = threadIdx.x; p < d; p += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < sk; ++j) {
            float s = 0.0f;
            for (int qq = 0; qq < d; ++qq) s += q[qq] * K[j * d + qq];
            s /= sqrtf((float)d);
            if (j > row) continue;
            acc += expf(s - m) / sum * V[j * d + p];
        }
        o[p] = acc;
    }
}
```

这个版本：

- 每个 block 处理一行；
- max 与 sum 各算一次 score，输出阶段又对每个输出维度重算，最多约 d+2 次，计算浪费很大；
- 没有 tiling，没有 shared memory，没有 tensor core；
- 仅用于理解 index。

真实实现：

1. Q/K/V tile 放 shared；
2. online softmax 一遍算完；
3. tensor core 做 QK^T 和 PV；
4. causal mask 在 tile 级别处理；
5. FlashAttention 不 materialize `S`。

### 4.3 TileLang 伪代码

```python
@T.prim_func
def attention(Q: T.Tensor((B, H, S, D), "float16"),
              K: T.Tensor((B, H, S, D), "float16"),
              V: T.Tensor((B, H, S, D), "float16"),
              O: T.Tensor((B, H, S, D), "float16")):
    with T.Kernel(T.ceildiv(S, BM), B * H, threads=256) as (bx, bh):
        Qs = T.alloc_shared((BM, D), "float16")
        Ks = T.alloc_shared((BN, D), "float16")
        Vs = T.alloc_shared((BN, D), "float16")
        acc = T.alloc_fragment((BM, D), "float32")
        m = T.alloc_fragment((BM,), "float32")
        l = T.alloc_fragment((BM,), "float32")
        # ... online softmax + T.gemm ...
```

具体 API 以 TileLang 版本为准；关键是理解 `BM/BN`、online softmax、causal mask。

---

## 5. GQA、MQA 与 MLA

### 5.1 GQA/MQA

- MHA：`h_q = h_kv`；
- MQA：`h_kv = 1`，所有 Q head 共享一组 K/V；
- GQA：`h_kv` 介于 1 和 `h_q` 之间。

收益：

```
KV bytes ∝ h_kv
```

`h_kv` 从 32 降到 8，KV 变为 1/4；decode 的带宽压力下降。

代价：K/V 的表达能力下降，需要用训练弥补。

### 5.2 MLA

MLA 不存完整 K/V，而存压缩 latent：

```
c_KV = x W_DKV          # [d_c]
K = c_KV W_UK, V = c_KV W_UV
```

KV cache 只存 `c_KV`（加 RoPE 部分）：

```
bytes = L * (kv_lora_rank + qk_rope_head_dim) * S * B * bytes
```

好处：KV 大幅缩小；代价：attention 前向多了一次投影，计算更复杂。

### 5.3 Sparse indexer

DeepInfer 的 V4 attention 还有 indexer：用 MQA 打分，选 top-k token，再对选中的 token 做 attention。这样 decode 只读 top-k 的 KV，进一步降低带宽；代价是索引本身的计算和准确性。

---

## 6. 仓库对照

- `deepinfer/model_executor/layers/attention_backend.py`：`prefill_attn_varlen_sparse`、`decode_mla_mqa_sparse`；注意 shape 和 `index_topk`。
- `.../models/v4_modules/mla_attn.py`：`ColumnParallelLora`、`MQAIndexer`、`LoraRopeCompressFlashMHAAutoRecomp`。
- `.../models/v4_modules/index_attn.py`：MQA indexer。
- `.../models/v4_modules/attn.py`：V4Attention 的完整路径，包含 compressed KV、SWA、YoCo、cross-layer。

先看 docstring 和 shape，不要逐行。

---

## 7. 练习

### 基础

**修错题**：对本讲零 logits 的手算例子，先 softmax 再把未来位置设为零，会得到哪一行和错误？

<details><summary>核对</summary>

第一行变成 `[0.5,0]`，和为 0.5，输出变成 1 而非 2。先 mask 再 softmax，使归一化只针对允许的位置。接着独立做 Lab2 Q6，用非相同 head 的输入检查 split/concat，不能只用 shape 测试。

</details>

1. 推导 `Var(q·k)=d`；解释为什么除 `sqrt(d)`。
2. 计算 `b=2,h=8,s=512,d=64,causal` 的 FLOPs。
3. 手写 causal mask，验证 `P` 的每一行和为 1。
4. 用 PyTorch 实现 attention，和 `F.scaled_dot_product_attention` 对比。

### 进阶

5. 实现一个 block 一行 attention（第 4.2 节），测 `s=128/512/2048` 的耗时，和 PyTorch 对比。
6. 改成每个 query tile 的 online softmax，解释 K/V 为什么仍可能被不同 query tile 重读；比较请求流量。
7. 实现 GQA：`h_q=8, h_kv=2`，验证输出 shape 和 KV bytes。
8. 用 `ncu` 看 attention kernel 的 `dram__throughput` 和 `sm__pipe_tensor_cycles_active`，判断 prefill/decode 的瓶颈。

### 挑战

9. 实现 tiled FlashAttention 前向（可参考第 6 讲），支持 causal。
10. 实现 MLA 的 latent KV 投影，比较 KV bytes 和 attention 误差。

---

## 8. 面试追问

1. 为什么 attention 要除 `sqrt(d)`？不除会怎样？
2. causal mask 为什么在 softmax 之前？
3. prefill 和 decode 的 attention 分别是什么 bound？AI 各是多少？
4. GQA/MQA/MLA 分别改变了什么？为什么能省 KV？
5. FlashAttention 省的是 FLOPs 还是 IO？
6. 如果 decode 的 attention 只占 20% 时间，你会先优化它吗？

---

## 9. 参考文献

- Vaswani et al., *Attention Is All You Need*。
- Dao et al., *FlashAttention* / *FlashAttention-2*。
- DeepSeek-V2：MLA；DeepSeek-V3：MLA + MoE。
- The Illustrated Transformer。

---

## 10. 下一步

- 做 [实验 2](../labs/lab02/README.md)；
- 做 [作业 2](../hw/hw02/README.md) 的 FLOPs/AI 部分；
- 读 [第 6 讲](lec06.md) 的 FlashAttention。
