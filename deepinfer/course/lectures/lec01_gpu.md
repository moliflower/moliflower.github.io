# 第 1 讲进阶：硬件吞吐与 CUDA 阅读

> **轨道**：选读，不属于第一讲的入门验收。  
> **前置**：完成 [第一讲核心](lec01.md) 和 Lab0，之后再补 C/C++ 指针与 CUDA 基础。  
> **一句话**：GPU 不是“更快的 CPU”，而是一台**吞吐机器**；写 kernel 的第一件事，是决定“一个 thread 做什么、一个 block 做什么、数据放在哪一层”。

---

## 0. 本讲要解决的问题

**本页是后置硬件资料，不是第一讲第二页必做题。** 当前自学主线在 [L1](lec01.md)、[L2核心](lec02.md)、[L3](lec03.md)、[L4](lec04.md) 已完整展开，Lab0–3与Project1均可只用标准库Python完成。下面出现的CUDA编译、设备规格和kernel片段要等C++/CUDA桥接准备好再实践；没有host驱动的片段不是可直接运行脚本。复习只需把thread/block、word/byte、请求/设备流量三个对应关系连起来。

从课程材料运行 `python course/...` 时，当前目录必须直接包含 `course`；离线包进入解压根目录，个人主页源仓库先 `cd _courses/deepinfer`。上游DeepInfer的实现源码属于额外阅读，不在课程离线包中；不要为做Python必修题安装整个推理引擎。

给定一个 kernel，你要能回答四个问题：

1. 它受**带宽**限制，还是受**算力**限制？
2. 需要多少 thread / block？每个 thread 处理多少元素？
3. 每个 warp 的一次访存会拆成多少个 **sector / cache line**？
4. 理论最短时间是多少？实测差多少？差在哪？

这一讲建立回答这四个问题所需的最小模型。前置：C 语言、指针、矩阵乘的基本定义。

---

## 1. 形式化执行模型

### 1.1 线程层级

一次 kernel launch 定义为：

```
grid  G = (Gx, Gy, Gz)
block B = (Bx, By, Bz)
warp  W = 32 threads
```

一个 thread 的全局索引（1D 情况）：

```
i = blockIdx.x * blockDim.x + threadIdx.x
```

约束：

- `Bx * By * Bz <= 1024`（大多数 NVIDIA GPU）；
- 一个 block 被调度到一个 SM 上，block 内 thread 可以共享 shared memory；
- 不同block之间**不能**用 `__syncthreads()` 同步。global memory和atomic本身也不构成全grid barrier；跨block协作需要正确协议、受支持的协作同步或多个kernel阶段，不能用自旋等待假定所有block都已驻留。

### 1.2 SIMT 与分支发散

入门时可以用一个 warp 共同发射指令、各 lane 有 active mask 的模型理解执行。Volta 以后的 independent thread scheduling 允许维护每线程执行状态，不能把“所有线程永远锁步”当作同步保证。若 lane 走不同分支：

```
if (lane < 16) { A; } else { B; }
```

硬件用 active mask 串行执行 A、B 两条路径，未激活的 lane 空转。代价近似：

```
cost = cost(A) + cost(B)
```

而不是 `max(cost(A), cost(B))`。所以 kernel 应尽量让同一 warp 走同一路径；无法避免时，用 predication（短分支）或重排数据。

### 1.3 Grid-stride loop

当 `n` 很大、grid 不需要覆盖全部元素时：

```cuda
for (int i = blockIdx.x * blockDim.x + threadIdx.x;
     i < n;
     i += blockDim.x * gridDim.x) {
    c[i] = a[i] + b[i];
}
```

优点：

- grid 大小可以与问题规模解耦（按 SM 数量配置）；
- 每个 thread 处理多个元素，摊薄 launch 和 index 开销；
- 适合“元素数远大于 thread 数”的 memory-bound kernel。

缺点：每个 thread 的循环次数可能不同，尾部有负载不均；对 compute-bound kernel 不如精确 tiling。

---

## 2. 硬件结构：SM、warp 与内存层次

### 2.1 一个 SM 里有什么

以 NVIDIA Ampere/Hopper 级别的 SM 为例（具体数字查规格表）：

| 资源 | 典型值 | 影响 |
| :--- | ---: | :--- |
| warp scheduler | 4 | 每 cycle 可发射的 warp 数 |
| 32-bit 寄存器 | 65536 / SM | 限制 occupancy |
| shared memory + L1 | 128–256 KB / SM | 可配置划分 |
| max warps / SM | 48–64 | 限制并发 warp 数 |
| max threads / block | 1024 | 限制 block 大小 |
| tensor core | 每 SM 多个 | 矩阵乘吞吐 |

一个 SM 同时驻留多个 block；每个 block 拆成若干 warp。warp scheduler 在每个 cycle 从“可发射”的 warp 中选一个发射指令。**warp 级并行**是 GPU 隐藏延迟的核心。

### 2.2 内存层次

| 层级 | 容量 | 延迟（量级） | 带宽（量级） | 作用 |
| :--- | ---: | ---: | ---: | :--- |
| 寄存器 | 每 thread 几十–几百 | 1 cycle | 极高 | 线程私有 |
| shared memory / L1 | 每 SM 百 KB | ~20–30 cycle | ~10–20 TB/s | block 内共享 |
| L2 | 几十 MB | ~200 cycle | ~几 TB/s | 全 GPU 共享 |
| HBM | 几十 GB | ~400–800 cycle | ~1–8 TB/s | 全局内存 |

数量级关系：**每往下一层，带宽降一个数量级，容量升一个数量级**。优化的本质是“把重复访问留在更上层”。

### 2.3 延迟隐藏与 Little's Law

要打满带宽，必须有足够多的 outstanding memory request：

```
in_flight_bytes = bandwidth × latency
```

例：`BW = 2 TB/s`，`latency = 500 ns`：

```
in_flight = 2e12 × 500e-9 = 1e6 bytes = 1 MB
```

也就是说，任意时刻要有约 1MB 的请求在飞。一个 warp 一次 load 128B，需要约 8192 个并发 warp 级请求；这就是 occupancy 和 ILP（instruction-level parallelism）重要的原因。

> occupancy 不是目的，而是“达到足够 in-flight 请求”的手段。高 occupancy 但每个 warp 都在等同一条依赖链，仍然慢。

---

## 3. 案例：vector add 的完整分析

### 3.1 数学与流量

```
c[i] = a[i] + b[i],  i = 0..n-1
```

float32：

```
Bytes = 3 * n * 4 = 12n
FLOPs = n          （一次加法）
AI    = n / 12n = 1/12 ≈ 0.083 FLOP/Byte
```

AI 极低；数据足够大、工作集从显存读取时，Roofline 预测它受带宽限制。小输入可能主要受 launch 延迟影响，缓存内工作集也不应直接套 HBM 带宽。

### 3.2 Roofline

```
P <= min(peak_FLOPs, bandwidth × AI)
T >= max(Bytes / bandwidth, FLOPs / peak_FLOPs)
```

以 A100 为例（数量级）：

- HBM ≈ 1.5–2 TB/s；
- FP32 ≈ 19.5 TFLOP/s；
- FP16 tensor ≈ 312 TFLOP/s。

FP32 的 ridge point：

```
AI* = 19.5e12 / 1.5e12 ≈ 13 FLOP/Byte
```

vector add 的 AI ≈ 0.083，远小于 13，memory-bound。

`n = 2^28`（约 2.68 亿）：

```
Bytes = 12 * 2.68e8 ≈ 3.2 GB
T_mem ≈ 3.2e9 / 1.5e12 ≈ 2.1 ms
T_compute ≈ 2.68e8 / 19.5e12 ≈ 0.014 ms
```

理想时间下界约 2.1 ms；实测若 4 ms，请求字节数除以时间得到约 50% 的标称带宽，可进一步检查 coalescing、对齐和缓存行为。

### 3.3 Naive CUDA

```cuda
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

问题：

- 每个 thread 只做 1 次加法，launch/index 开销占比高；
- 没有利用 vectorized load；
- `n` 很大时 grid 很大，尾部 block 可能很少。

### 3.4 Vectorized + grid-stride

```cuda
__global__ void vector_add_vec4(const float4* __restrict__ a,
                                const float4* __restrict__ b,
                                float4* __restrict__ c,
                                int n4) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n4;
         i += blockDim.x * gridDim.x) {
        float4 x = a[i];
        float4 y = b[i];
        float4 z;
        z.x = x.x + y.x; z.y = x.y + y.y;
        z.z = x.z + y.z; z.w = x.w + y.w;
        c[i] = z;
    }
}
```

要点：

- `float4` 让每个 thread 一次搬 16B，减少指令数；
- `__restrict__` 告诉编译器 `a/b/c` 不重叠，允许更好的调度；
- 要求指针 16B 对齐，`n` 是 4 的倍数；尾部用 scalar kernel 处理；
- grid 可设为 `SMs * blocks_per_SM`，不随 `n` 线性增长。

### 3.5 性能测量

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);
for (int i = 0; i < 10; ++i) launch();      // warmup
cudaEventRecord(start);
for (int i = 0; i < 100; ++i) launch();
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms; cudaEventElapsedTime(&ms, start, stop);
double bw = 3.0 * n * sizeof(float) * 100 / (ms * 1e-3) / 1e9;
```

报告：`GB/s`、`% peak`、median/p90，不要只报最好的一次。

---

## 4. 正确性与边界

- **n = 0**：grid 可能为 0，kernel 不 launch；host 端直接返回。
- **n 不是 block 的倍数**：最后 block 要 `if (i < n)`；vectorized 版本要处理 `n % 4`。
- **对齐**：`float4` 要求 16B 对齐；`cudaMalloc` 通常 256B 对齐，但切片/偏移可能破坏对齐。
- **别名**：若 `c` 与 `a` 重叠，`__restrict__` 是 UB；不要给可能有 alias 的指针加。
- **整数溢出**：`n > 2^31` 时 `int` 溢出；用 `int64_t`/`size_t`，并注意 grid 维度上限。
- **NaN/Inf**：vector add 会传播 NaN；如果要检测，单独写 kernel。
- **确定性**：vector add 是确定的；reduction 的加法顺序会影响浮点结果，见第 3 讲。

---

## 5. 仓库对照

- `deepinfer/model_executor/layers/tilelang_ops/cuda/norm.py`：看 `threads`、`blk_d`、`vec_size`、`blk_m` 怎么选；这就是“让每个 thread 读连续 vector”的工程化版本。
- `.../cuda/gather.py`：index 型 kernel，访存不连续时怎么处理。
- `.../cuda/store_kv_cache.py`：block table / slot mapping 下的写入。
- `.../cuda/state_cache.py`：circular buffer 的 `%` 运算。

读的时候只问三个问题：一个 CTA 处理多少行？一个 thread 处理多少元素？尾部怎么处理？

---

## 6. 练习

### 基础

1. 对 `n=1024, block=128` 的 vector add，计算 grid、warp 数、每个 warp 的 load/store transaction 数。
2. 把 naive kernel 改成 grid-stride，grid 固定为 `SM_count * 8`，测不同 `n` 的带宽。
3. 用 `float4` 重写，处理 `n % 4 != 0`，对比 naive 的带宽。

### 进阶

4. 写一个 bandwidth test：只读、只写、copy、triad（`a[i] = b[i] + scalar * c[i]`），画 `n` 从 1MB 到 1GB 的带宽曲线，解释小尺寸为什么低。
5. 用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 计算你的 kernel 的理论 occupancy，和 `ncu` 的 `sm__warps_active.avg.pct_of_peak_sustained_active` 对比。
6. 把 `float4` 改成 `float2`/`float`，在相同 `n` 下比较；解释为什么不是“越宽越好”。

### 挑战

7. 写一个 vector add 的 CUDA graph 版本，比较 launch 开销。
8. 用 `__ldg` / `__ldcs` / `__stcs` 等 cache hint，观察不同数据规模下的带宽变化。

---

## 7. 面试追问

1. SIMT 和 SIMD 的区别是什么？warp divergence 的代价怎么估算？
2. 为什么 vector add 是 memory-bound？AI 是多少？拐点怎么算？
3. occupancy 受哪些资源限制？高 occupancy 一定快吗？
4. 什么是 Little's Law？它和“要多少并发 warp”有什么关系？
5. `float4` 一定比 `float` 快吗？在什么情况下不一定？
6. 如何测一个 kernel 的有效带宽？warmup、cache flush、median 为什么重要？

---

## 8. 参考文献

- NVIDIA, *CUDA C++ Programming Guide*：Programming Model、Memory Hierarchy。
- NVIDIA, *CUDA C++ Best Practices Guide*：Memory Optimizations、Execution Configuration。
- PMPP, chapters 1–3。
- Williams et al., *Roofline: An Insightful Visual Performance Model*。

---

## 9. 下一步

- 做 [实验 0](../labs/lab00/README.md)；
- 读 [第 2 讲](lec02.md) 的 coalescing 与 bank conflict；
- 在 `notes/lec01.md` 写出第 6 节练习 1、2 的计算过程。
