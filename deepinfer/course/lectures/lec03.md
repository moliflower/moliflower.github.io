# 第 3 讲：Reduction、Softmax 与 RMSNorm

> **课前**：35 分钟  
> **课后**：实验 1  
> **一句话**：这三个算子看起来只有几行，但它们决定了你搬多少 HBM、用多少寄存器、以及数值会不会炸。

---

## 0. 本讲要解决的问题

**先修**：Python 函数、循环、列表；`exp` 是指数、`sqrt` 是平方根；第 1 讲的 float32=4B。主线是第 1.1、1.5、2.1–2.3、3.1 与 Lab1。CUDA/shuffle 是桥接后选读，无 GPU 也能完成必做题。

1. 一个 reduction 的**工作量**和**深度**是多少？为什么 GPU 要用树形？
2. 多 block reduction 为什么要两遍？代价是什么？
3. softmax 的 1/2/3-pass 各搬多少字节？online softmax 的更新公式怎么推？
4. RMSNorm 与 LayerNorm 的数学差异是什么？融合后省了哪些 traffic？
5. 浮点非结合性如何影响结果？什么时候必须用 deterministic reduction？

---

## 1. Reduction：工作量、深度与数值

### 1.1 三个指标

对一个长度为 `n` 的数组求和：

| 算法 | 工作量（加法次数） | 深度（依赖链） | 额外空间 |
| :--- | ---: | ---: | ---: |
| 串行 | n-1 | n-1 | O(1) |
| 相邻二叉树 | n-1 | ceil(log2 n) | O(n)，允许原地覆盖时无需另一整份数组 |
| warp shuffle | 有效归约树 n-1；实际各 lane 可能执行冗余加法 | 满 warp 为 5 轮 | 每 lane O(1) 寄存器 |
| 多 block 两遍 | (n-grid)+(grid-1)=n-1 | ceil(log2(n/grid))+ceil(log2 grid)，假设均分 | O(grid) partials |

本表 n>0，grid 为非空分组数；n=0 的返回值由接口约定。工作量决定总操作，深度决定最长依赖链，性能还受可用并行度与访存影响。

**手算**：`[1,2,3,4,5] → [3,7,5] → [10,5] → [15]`。三轮分别做 2、1、1 次加法，共 4=n-1。分两组 `[1,2,3]` 与 `[4,5]` 时第一遍做 2+1 次，第二遍 1 次，仍共 4 次，不能再额外加一次 grid-1。

### 1.2 Warp shuffle reduction

以下 CUDA 片段只在一个完整 warp 的 32 个 lane 全部参与、mask=`0xffffffff` 时使用。`i>=n` 的线程可以传 0，但不能提前退出。block 示例要求 blockDim.x 是 32 的倍数且不超过 1024；部分 warp 需另写合法的 active mask 算法。

```cuda
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val; // lane 0 得到整个 warp 的和
}
```

5 步后，lane 0 拥有 32 个 lane 的和。其他 lane 的值不保证正确（取决于实现），所以通常只在 lane 0 使用结果。

### 1.3 Block reduction

```cuda
__global__ void block_reduce(const float* x, float* out, int n) {
    __shared__ float smem[32];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float val = (i < n) ? x[i] : 0.0f;

    val = warp_reduce_sum(val);
    if ((tid & 31) == 0) smem[tid >> 5] = val;
    __syncthreads();

    if (tid < 32) {
        val = (tid < (blockDim.x + 31) / 32) ? smem[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) out[blockIdx.x] = val;
    }
}
```

要点：

- 第一层：每个 warp 内部 shuffle；
- 第二层：每个 warp 的 lane 0 写 shared；
- 第三层：第一个 warp 再 reduce 这些 partial；
- `__syncthreads()` 保证 shared 可见；
- 边界：`i >= n` 的 lane 补 0，不能读越界。

### 1.4 多 block：两遍 vs atomics

两遍：

```
kernel 1: 每个 block 写 partials[bid]
kernel 2: 一个 block reduce partials
```

traffic：

```
read x: n * 4
write partials: grid * 4
read partials: grid * 4
write final: 4
```

atomics：

```cuda
atomicAdd(out, val);
```

- 优点：一个 kernel；
- 缺点：原子冲突、结果顺序不确定、不可复现；
- 对 float，`atomicAdd` 的顺序会影响结果。

### 1.5 浮点非结合性

浮点加法不满足结合律：

```
(a + b) + c != a + (b + c)
```

例子（float32）：

```
1e8 + 1.0 - 1e8  -> 0.0（1.0 被吞掉）
```

因此：

- 串行、树形、warp shuffle、atomic 的结果可能不同；
- 如果要求 deterministic（batch invariant、可复现），必须固定 reduction 顺序，或使用更高精度（FP64/Kahan）并固定树形；
- 误差上界：树形 reduction 的误差约为 `O(log n * eps * sum|x|)`，串行约为 `O(n * eps * sum|x|)`。树形通常更准。

---

## 2. Softmax：从定义到 online 更新

### 2.1 定义与不变性

```
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

对任意常数 `c`：

```
softmax(x + c) = softmax(x)
```

证明：分子分母同乘 `exp(c)`，约掉。因此可以减最大值 `m = max(x)`：

```
y_i = exp(x_i - m) / Σ_j exp(x_j - m)
```

这样最大指数是 `exp(0)=1`，不会溢出。

### 2.2 Online 更新公式

设已经处理了前 `k` 个元素，维护：

```
m_k = max(x_1..x_k)
s_k = Σ_{j<=k} exp(x_j - m_k)
```

加入 `x_{k+1}`：

```
m' = max(m_k, x_{k+1})
s' = s_k * exp(m_k - m') + exp(x_{k+1} - m')
```

因为旧的 `s_k` 是以 `m_k` 为基准的，换成新基准 `m'` 要乘 `exp(m_k - m')`。

最终：

```
y_i = exp(x_i - m_n) / s_n
```

注意：这个公式只维护 `(m, s)`，输出仍然需要第二遍（或把输出累加器一起 rescale）。FlashAttention 把 `O` 也做同样的 rescale：

```
o' = o * exp(m - m') + exp(x - m') * V
```

这里 o 是**未归一化加权和**，最终还要返回 `o/s`；如果把 o 误认为已经归一化输出，更新式会错。手算 logits `[0,ln(2)]`、V=`[2,6]`：第一步 `(m,s,o)=(0,1,2)`；第二步 `(ln(2),1.5,7)`；最终输出 `7/1.5=14/3`。

### 2.3 1/2/3-pass 的 traffic

对长度 `n`、float32：

| 实现 | 读 x | 写 y | 额外 | 总字节 |
| :--- | ---: | ---: | :--- | ---: |
| 3-pass | 3 | 1 | max、sum | 16n |
| 2-pass | 2 | 1 | max+sum 融合 | 12n |
| 1-pass（x 在 on-chip） | 1 | 1 | 需要存 x | 8n |

如果 `n=4096`：

- 3-pass：65536B = 64KiB；
- 2-pass：49152B = 48KiB；
- 1-pass：32768B = 32KiB。

在假设 2×10¹² B/s 下，每少读一遍省 16384B，对应 8.192ns/行的带宽模型差值。它不含 launch、归约、exp 与缓存；不能把这个差值当成实际 kernel 加速。

### 2.4 代码：数值稳定的 softmax

以下是算法片段，不可直接编译：`block_reduce_max/sum` 需实现为**向所有线程广播**的归约，并正确同步，不能直接用只给 lane0 结果的前节函数替代。

```cuda
__global__ void softmax_row(const float* x, float* y, int n) {
    int row = blockIdx.x;
    const float* xr = x + row * n;
    float* yr = y + row * n;

    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, xr[i]);
    m = block_reduce_max(m);

    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += expf(xr[i] - m);
    s = block_reduce_sum(s);

    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = expf(xr[i] - m) / s;
}
```

优化：

- 如果 `n <= 1024`，一行一个 block；每行数据可以留在寄存器/shared，减少 global 往返；
- 用 `__expf`（fast math）会损失精度，采样场景要评估；
- 对 vocab=128K，一行很大，必须分块/多 pass；
- 采样时通常只需要 top-k/top-p，可以避免 full softmax（见 `fused_sample.py`）。

### 2.5 边界

- 所有元素 `-inf`：分母为 0，通常的概率定义无效；选择报错或明确的 mask 语义，不能无理由设成均匀分布。
- 存在 `+inf`：`inf-inf` 产生 NaN；需要约定如何处理一个或多个 +inf。本课 Lab1 限有限输入；Lab2 允许 -inf mask，但对 NaN、+inf、全 -inf 行报错。
- 全 0：`m=0`，`s=n`，均匀分布。
- 空输入：未定义；框架通常返回空。

---

## 3. RMSNorm 与 LayerNorm

### 3.1 定义

LayerNorm：

```
μ = mean(x)
σ² = mean((x - μ)²)
y_i = (x_i - μ) / sqrt(σ² + ε) * w_i + b_i
```

RMSNorm：

```
r = sqrt(mean(x_i²) + ε)
y_i = x_i / r * w_i
```

差异：

- LayerNorm 需要 mean 和 variance 两个 reduction；
- RMSNorm 只需要 mean-square 一个 reduction，没有 bias；
- 忽略 ε 时，RMSNorm 对正比例缩放不变：`RMSNorm(a*x)=RMSNorm(x)`（a>0）。固定 ε 时仅近似成立；对整体平移通常不具备不变性。是否减均值影响的是平移，不是正比例缩放。

### 3.2 数值细节

- 用 FP32 累加 `x_i²`，即使输入是 BF16/FP16；
- `rsqrt` 可以比 `1/sqrt` 快，但精度略低；
- `ε` 的位置：`sqrt(mean + ε)` vs `sqrt(mean) + ε`，不同框架可能不同，读代码要确认；
- 如果 `mean(x²)` 很小，`ε` 防止除零；如果很大，`ε` 可忽略；
- 对 `x` 做 residual add 时，注意 `x + residual` 的溢出/精度。

### 3.3 融合的收益

一个 decoder layer 里常见：

```
residual add -> RMSNorm -> QKV projection
```

分开写：

1. residual add：读 x、读 residual、写 x（12n bytes）；
2. RMSNorm：读 x、写 y（8n）；
3. QKV：读 y。

融合后：

1. 读 x、读 residual，算 sum；
2. 在寄存器/shared 里算 RMSNorm；
3. 直接写 FP8 量化后的 y，或直接喂给 GEMM。

少一次 HBM 往返，就是融合的收益。DeepInfer 的 `fused_transform.py` 把 norm + RoPE + Hadamard + cast + KV store 融在一起，是同一思路的极致版。

### 3.4 仓库对照

- `deepinfer/model_executor/layers/tilelang_ops/cuda/norm.py`：`norm_tl` 的参数 `has_residual`、`do_fp8_cast`、`return_origin_out`、`shared_sf_block_m`；看它如何在一次读入中完成多种变换。
- `.../tilelang_ops/fused_transform.py`：`fused_norm_rope_hadamard_cast_store_kvcache` 的参数列表，理解融合了哪些步骤。
- `.../layers/layernorm.py`、`.../layers/rotary_embedding.py`：上层封装。
- `.../layers/sampler.py`：softmax 在采样中的位置；`fused_sample.py` 如何避免 full softmax。

---

## 4. 性能分析

### 4.1 Reduction 是 memory-bound

对一个 `n` 元素的 reduction：

```
Bytes = n * 4（读）+ 少量写
FLOPs = n
AI = 1/4 = 0.25 FLOP/Byte
```

远低于拐点，memory-bound。优化点：

- 减少读次数（一次读入，寄存器/shared 内完成）；
- 合并多个 reduction；直接用 `E[x²]-E[x]²` 会在均值大、方差小时发生消减，稳定单遍方差可用 Welford，不能仅因 pass 少就替换数值算法；
- 对多行做 batched reduction，提高并行度；
- 避免 atomic 热点。

### 4.2 Softmax 的 pass 数

如 2.3 所述，3-pass -> 2-pass -> 1-pass 的收益来自少读 x。对长 vocab，1-pass 需要把 x 放在 shared/寄存器；如果放不下，2-pass 是折中。

### 4.3 RMSNorm 的 occupancy

一个 block 处理一行，`blockDim=256/512`；如果 hidden=4096，每个 thread 处理 8–16 个元素，寄存器压力小；如果 hidden=8192，每个 thread 处理 16–32 个，注意 register spill。

---

## 5. 练习

### 基础

**修错题**：更新 online state 时先写 `m=m_new`，随后执行 `s=s*exp(m-m_new)+exp(v-m_new)`。输入 `[0,ln(2)]` 得到的 s 为什么会是 2？

<details><summary>核对</summary>

旧 m 被覆盖，使旧 s 的缩放因子错误地变成 1。应使用旧 m 更新 s，然后再赋值 m。正确 s=1.5。本题对应 Lab1 Q2，之后独立完成 Q5 LayerNorm，不以背出更新式代替不变式解释。

</details>

1. 证明 `softmax(x+c)=softmax(x)`。
2. 推导 online softmax 的 `s'` 更新公式。
3. 计算 4096 维 float32 的 3-pass/2-pass/1-pass traffic 和 2TB/s 下的时间差。
4. 写一个 warp reduce sum，验证 32 个随机数的和与 CPU 结果在 1e-5 内。

### 进阶

5. 实现 block reduce + grid two-pass，测 `n=1e6/1e7/1e8` 的带宽。
6. 实现 row-wise softmax（一行一个 block），对比 3-pass 和 2-pass 的耗时。
7. 实现 RMSNorm + residual + FP8 cast 的融合 kernel（可用 TileLang），对比分开写的 traffic。
8. 构造一个浮点非结合性例子，展示串行和树形结果不同；说明如何保证 deterministic。

### 挑战

9. 实现 Kahan/Neumaier 补偿求和，比较误差和吞吐。
10. 实现 n 很大时的两遍 online softmax，解释完整概率输出为什么仍需要重读 x 或保存足够信息；再比较只输出加权和的 `(m,s,o)` 算法。

---

## 6. 面试追问

1. reduction 的工作量和深度分别是什么？为什么 GPU 用树形？
2. 多 block reduction 为什么需要两遍？atomics 有什么问题？
3. softmax 为什么要减 max？online softmax 的 rescale 怎么推？
4. LayerNorm 和 RMSNorm 的数学差异？为什么 RMSNorm 更快？
5. 浮点非结合性对推理有什么影响？什么是 batch invariant？
6. 一个 4096 维 softmax，3-pass 和 2-pass 差多少字节？在 2TB/s 下差多少时间？

---

## 7. 参考文献

- NVIDIA, *Online normalizer calculation for softmax*（Milakov & Gimelshein）。
- LayerNorm: Ba et al., 2016；RMSNorm: Zhang & Sennrich, 2019。
- FlashAttention, §3：online softmax 与 IO complexity。
- PMPP, chapter 10（reduction）。

---

## 8. 下一步

- 做 [实验 1](../labs/lab01/README.md)；
- 做 [作业 1](../hw/hw01/README.md) 的量化与 KV 部分；
- 读 [第 4 讲](lec04.md) 的 GEMM 与 tiling。
