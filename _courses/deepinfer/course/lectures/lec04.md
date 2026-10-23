# 第 4 讲：Tiling、GEMM 与 Tensor Core

> **课前**：40 分钟  
> **课后**：项目 1 Phase 2  
> **一句话**：GEMM 的优化不是“把循环写快”，而是把 HBM 的重复读取搬到 shared memory/寄存器，再喂给 tensor core。

---

## 0. 本讲要解决的问题

**先修**：矩阵乘的 `(M,K)@(K,N)` shape、三层循环、FLOP/Byte 与带宽单位。必做先手算第 1–2 节，再完成 Project1 的显式 tile 搬运与计数；Tensor Core 指令、CUDA 和 TileLang 为桥接后选读。

1. 一个 `n×n×n` 的 GEMM，FLOPs、Bytes、AI 各是多少？在你的卡上是 memory-bound 还是 compute-bound？
2. 朴素 GEMM 为什么慢？tiling 为什么能把 traffic 降到 `1/tile`？
3. shared memory tile、register tile、warp tile 各自负责什么？
4. tensor core 的 mma 形状、fragment 布局、FP8/FP4 的 scale 怎么工作？
5. 怎么用 `ncu` 判断 GEMM 是“算力没打满”还是“带宽没打满”？

---

## 1. GEMM 的数学与代价

### 1.1 定义

```
C[m,n] = A[m,k] @ B[k,n]
C[i,j] = Σ_{p=0}^{k-1} A[i,p] * B[p,j]
```

FLOPs：

```
2 * m * n * k   （每个输出 k 次乘加，乘加=2 FLOPs）
```

方阵 `n×n×n`：

```
FLOPs = 2 n^3
```

### 1.2 最小 traffic 与 AI

读 A、B，写 C：

```
Bytes = (m*k + k*n + m*n) * bytes_per_elem
```

方阵、float32：

```
Bytes ≈ 12 n^2
AI = 2n^3 / 12n^2 = n / 6
```

| n | 理想 AI (FP32) | 假设标量 FP32 ridge=13 FLOP/B 时的 roofline 上界 |
| ---: | ---: | :--- |
| 128 | 21.3 | 理想点落在算力平台；小任务未必能达到 |
| 1024 | 170.7 | 理想点落在算力平台 |
| 4096 | 682.7 | 理想点落在算力平台 |

> 这里把 A/B 各读一次是理想流量下界。实际算法可能重读输入，故实际 AI 更低；短任务还可能受 launch/并行度限制。FP32 CUDA-core ridge 不能拿来判断 FP16/BF16 Tensor Core kernel，必须匹配峰值的 dtype 与计算单元。

### 1.3 朴素 GEMM 的重复读取

```cuda
// 计数用伪代码；假设 C 已清零，实际 kernel 通常先在寄存器累计。
for (int i = 0; i < n; ++i)
  for (int j = 0; j < n; ++j)
    for (int p = 0; p < n; ++p)
      C[i][j] += A[i][p] * B[p][j];
```

每个 `(i,j)` 读 A 的一行和 B 的一列。A 被读 `n` 次，B 被读 `n` 次：

```
global reads ≈ 2 n^3 个元素
```

`n=1024` 时约 2.147G 次输入元素读取 = 8.590GB（8GiB），而 A/B/C 容量合计为 12MiB。这是算法发出的标量读取量；缓存可以复用数据，实际 HBM 流量不会自动等于 8GiB。

---

## 2. Tiling：把重复读取搬到片上

### 2.1 Block tile

把 C 切成 `BM × BN` 的块。一个 CTA 负责一个块：

```
for k0 in 0..K step BK:
    load A[i0:i0+BM, k0:k0+BK] -> shared
    load B[k0:k0+BK, j0:j0+BN] -> shared
    compute C_tile += A_tile @ B_tile
```

global traffic（每个 CTA）：

```
A: BM * K
B: K * BN
C: BM * BN
```

总 CTA 数 `(M/BM)*(N/BN)`：

```
global elements ≈ (M/BM)*(N/BN)*(BM*K + K*BN) + M*N
                = M*N*K*(1/BN + 1/BM) + M*N
```

方阵 `BM=BN=BT`：

```
≈ 2 n^3 / BT + n^2
```

相比朴素 `2n^3`，降到 `1/BT`。

**完整手算**：n=3、tile=2，C 有 2×2 个输出 tile。每个 A 元素被两个列 tile 读取，每个 B 元素被两个行 tile 读取，因此输入共 `2*9*2=36` 次，C 写 9 次，共 45 元素=180B；朴素为 `2*27+9=63` 元素=252B。尾块只统计有效元素。

Project1 的精确方阵模型是 `elements=2*n²*ceil(n/tile)+n²`，而非把 n/tile 当小数。上面的 `2n³/BT+n²` 适用于可整除情形。若实际 kernel 读取 padding，须把 padding 另计。

### 2.2 三层 tiling

| 层级 | 负责 | 存储 | 典型大小 |
| :--- | :--- | :--- | :--- |
| CTA / block tile | 一个 block 的 C 块 | shared memory | 128×128 / 128×256 |
| warp tile | 一个 warp 的 C 块 | registers | 64×64 |
| thread tile | 一个 thread 的 C 元素 | registers | 8×8 |

数据流：

```
HBM -> shared memory -> registers -> tensor core
```

每往上一层，带宽更高、容量更小。

### 2.3 双缓冲 / pipeline

```
load tile k+1 -> compute tile k
```

用 `cp.async`（Ampere）或 TMA（Hopper+）把下一块数据提前搬到 shared，和当前块的计算重叠。CUTLASS/DeepGEMM 的核心就是这条 pipeline。

### 2.4 Bank conflict 与 swizzle

shared memory tile 的行 stride 若等于 32 个 word，按列访问会 32-way conflict。解决：

- padding：行 stride 33；
- XOR swizzle：`col ^ (row & mask)`；
- TMA swizzle：硬件支持的 128B/64B/32B swizzle 模式。

### 2.5 Wave quantization

一个 SM 同时跑若干 CTA。若 grid 的 CTA 数不是 `SMs * CTAs_per_SM` 的整数倍，最后一波只有部分 CTA，利用率下降。

例：108 SMs，每 SM 2 个 CTA，共 216 个并发 CTA；grid=1000 -> 4 整波 + 136 个尾 CTA（63% 尾波）。对大 GEMM，尾波占比小；对小 GEMM，尾波可能是主要开销。

---

## 3. Tensor Core

### 3.1 MMA 指令

Tensor core 一条指令做一个小矩阵乘：

```
D = A @ B + C
```

Ampere 的 `mma.sync.aligned.m16n8k16`：

- A: 16×16（FP16/BF16）
- B: 16×8
- C/D: 16×8（FP32 累加）

Hopper 的 `wgmma` 以 warpgroup（4 warps）为单位，支持更大的 tile；Blackwell 的 `tcgen05` 进一步把 MMA 和 TMA 结合。

### 3.2 Fragment 布局

Tensor core 的操作数是分布在寄存器里的 **fragment**，不是普通矩阵。一个 warp 的 32 个 lane 各持有 fragment 的一部分；布局由 PTX 文档规定。CUTLASS 用 `Layout` 类型在编译期描述这些映射。

如果布局错了，结果会错；如果 layout 不匹配，就需要 `ldmatrix`/`movmatrix` 重排，增加开销。

### 3.3 精度

| 输入 | 累加 | 特点 |
| :--- | :--- | :--- |
| FP16 | FP32 | 精度较高，范围小 |
| BF16 | FP32 | 范围大，精度较低 |
| FP8 E4M3 | FP32 | 需要 scale，范围/精度更小 |
| FP4 E2M1 | FP32 | 需要 block scale，精度最低 |

关键点：

- **累加用 FP32**，低精度输入不等于低精度累加；
- **scale**：FP8/FP4 需要 per-tensor/per-token/per-group scale；
- **outlier**：FP8 的动态范围比 FP16 小，per-group scale 是折中；
- **rounding**：FP8 cast 的舍入模式会影响误差。

DeepInfer 的 Q&A：

- 主 KV：per-16 e4m3；
- indexer KV：per-32 ue8m0 MXFP4；
- MoE 路由专家权重：per-32 ue8m0 MXFP4。

---

## 4. 代码：一个可读的 tiled GEMM

```cuda
#define BM 16
#define BN 16
#define BK 16

__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BN + threadIdx.x;
    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // load A tile: BM x BK
        As[threadIdx.y][threadIdx.x] = (row < M && k0 + threadIdx.x < K)
            ? A[row * K + k0 + threadIdx.x] : 0.0f;
        // load B tile: BK x BN
        Bs[threadIdx.y][threadIdx.x] = (k0 + threadIdx.y < K && col < N)
            ? B[(k0 + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();

        for (int p = 0; p < BK; ++p)
            acc += As[threadIdx.y][p] * Bs[p][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
```

启动配置为 `block=dim3(16,16)`、`grid=dim3((N+15)/16,(M+15)/16)`，共 256 个线程。这里 BM=BN=BK=16，每线程搬 A/B 各一个元素。尾块无效位置填零，所有线程都经过两次 barrier；不能提前 return 跳过同步。这个 kernel 片段需配合 host 驱动编译，host 基础见 C++/CUDA 桥接单元。

这是一个教学版本：

- 每个 thread 只算 1 个输出，寄存器复用少；
- 没有 register blocking；
- 没有 vectorized load / cp.async；
- 没有 tensor core。

真实高性能 GEMM 会：

1. 每个 thread 算 `TM×TN` 个输出（register tile）；
2. 用 `float4`/`cp.async` 搬 shared；
3. 用 `ldmatrix` 把 shared 搬到 fragment；
4. 用 `mma`/`wgmma` 做矩阵乘；
5. 用 double buffer 重叠 load/compute；
6. 用 swizzle 避免 bank conflict；
7. 用 split-K 处理小 M/N、大 K。

### TileLang 版本

以下是 API 结构示意，需要 `tilelang` 与 `tilelang.language as T`、匹配版本及 GPU host 调用；不能作为当前 Python 作业的可运行答案。尾块的编译器 mask 行为需以使用的版本验证。

```python
@tilelang.jit(out_idx=[2])
def matmul(M, N, K, BM=128, BN=128, BK=32):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"),
             B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=256) as (bx, by):
            As = T.alloc_shared((BM, BK), "float16")
            Bs = T.alloc_shared((BK, BN), "float16")
            Cl = T.alloc_fragment((BM, BN), "float32")
            T.clear(Cl)
            for k0 in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
                T.copy(A[by * BM, k0 * BK], As)
                T.copy(B[k0 * BK, bx * BN], Bs)
                T.gemm(As, Bs, Cl)
            T.copy(Cl, C[by * BM, bx * BN])
    return main
```

TileLang 帮你做 layout、pipeline、tensor core 映射；你要选的是 `BM/BN/BK`、`num_stages`、threads、swizzle。

---

## 5. 性能分析

### 5.1 指标

| 指标 | 含义 | 目标 |
| :--- | :--- | :--- |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | tensor core 利用率 | 越高越好 |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 总利用率 | 接近 100% |
| `dram__throughput...` | HBM 利用率 | 大 GEMM 通常不高 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | shared bank conflict | 接近 0 |
| `smsp__average_warps_issue_stalled_*` | stall 原因 | 定位 pipeline/依赖 |

### 5.2 效率

```
TFLOP/s = 2*M*N*K / (time_seconds * 1e12)
% peak = TFLOPS / peak_TFLOPS
```

对 `M=N=K=4096`，FP16 tensor peak 312 TFLOPS（A100）：

```
FLOPs = 2 * 4096^3 = 137.4 GFLOP
若 10 ms -> 13.7 TFLOPS -> 4.4% peak（很低）
若 0.5 ms -> 274.9 TFLOPS -> 88% peak（接近 cuBLAS）
```

### 5.3 常见瓶颈

- tile 太小：shared 复用不足，global traffic 高；
- tile 太大：shared/寄存器压力大，occupancy 低；
- BK 太小：pipeline 深度不够，tensor core 等数据；
- bank conflict：shared load 串行；
- 尾波：grid 不是并发 CTA 的整数倍；
- split-K 过度：额外 reduction 开销。

---

## 6. 正确性与边界

- **边界**：M/N/K 不是 tile 的倍数时，load 要 mask，store 要 mask；否则越界。
- **累加精度**：FP16 输入用 FP32 累加；如果累加也用 FP16，误差会很大。
- **NaN/Inf**：FP8 cast 可能把大值 clamp 到 amax；检查 scale。
- **对齐**：tensor core 对 shared layout 有对齐要求；`ldmatrix` 需要 16B 对齐。
- **确定性**：GEMM 的归约顺序由 tile 决定；split-K 会改变顺序，可能影响结果。

---

## 7. 仓库对照

- `deepinfer/model_executor/models/v4_modules/mlp.py`：`DeepseekMoE`、`_grouped_gemm`、`mega_moe_forward`；看它如何调用 DeepGEMM。
- `deepinfer/model_executor/layers/linear.py`：`ReplicatedLinear` 的 FP8/FP16 路径。
- `vendor/DeepGEMM`（需初始化 submodule）：grouped GEMM、mega MoE、FP8/FP4。
- `deepinfer/model_executor/layers/tilelang_ops/cuda/quant.py`：scale 布局。

读的时候问：这个 GEMM 的 M/N/K 是多少？FP8 还是 BF16？scale 粒度？为什么用 grouped/contiguous layout？

---

## 8. 练习

### 基础

**预测—解释—修错**：有人用 `2*n**3/tile+n**2` 给 n=3、tile=2 预测 36 个元素，测试却计到 45。先不用代码解释差的 9 个元素来自哪里，再把公式改成对任意正整数 tile 成立的版本。

<details><summary>核对</summary>

有两个输出 tile 行/列，不能把它们当成 1.5 个；A/B 各被读 2 次。改成 `2*n*n*((n+tile-1)//tile)+n*n`。随后做 Project1 的尾块测试，确认是数据复用带来计数下降，而非只换循环顺序。

</details>

1. 计算 `n=512/1024/4096` 的 FLOPs、最小 Bytes、AI；判断你的卡上是否 compute-bound。
2. 推导 tiled GEMM 的 global traffic 公式，验证 `tile=32` 时约降到 `1/32`。
3. 用 `matmul_traffic(n, tile)` 画 `n=1024` 时 traffic 随 tile 的曲线。

### 进阶

4. 实现第 4 节的 tiled GEMM，测 `n=512/1024/2048` 的 TFLOPS，和 PyTorch `matmul` 对比。
5. 加 register blocking（每个 thread 算 4×4 个输出），再测。
6. 用 `ncu` 看 tensor core 利用率、shared bank conflict、occupancy；解释每个数字。
7. 用 TileLang 写一个 `BM=BN=128, BK=32` 的 GEMM，扫 `num_stages=1/2/3`。

### 挑战

8. 实现 split-K GEMM，比较 `K` 很大、`M/N` 很小时的性能。
9. 把 FP16 输入换成 FP8（per-tensor scale），比较 TFLOPS 和误差。
10. 读 DeepGEMM 的 grouped GEMM 接口，画一张 token permutation → grouped GEMM → combine 的数据流。

---

## 9. 面试追问

1. 朴素 GEMM 为什么慢？tiling 的收益来自哪里？
2. block tile / warp tile / thread tile 各自负责什么？
3. tensor core 的 mma 形状是什么？FP16 输入为什么要 FP32 累加？
4. FP8/FP4 的 scale 粒度怎么选？outlier 怎么处理？
5. 什么是 wave quantization？小 GEMM 为什么效率低？
6. `sm__pipe_tensor_cycles_active` 只有 20%，你会先查什么？

---

## 10. 参考文献

- CUTLASS docs：Efficient GEMM、CuTe layout。
- NVIDIA, *CUDA C++ Programming Guide*：WMMA / tensor core。
- DeepGEMM：<https://github.com/deepseek-ai/DeepGEMM>
- PMPP, chapter 7；siboehm, *How to Optimize a CUDA Matmul Kernel*。

---

## 11. 下一步

- 做 [项目 1](../projects/project1/README.md) 的 Phase 2；
- 用 `matmul_traffic` 填报告表；
- 读 [第 5 讲](lec05.md) 的 attention。
