# 第 8 讲：量化（FP8 / FP4）

> **课前**：35 分钟  
> **课后**：作业 1 Q6–Q8 + 算一次 KV 量化收益  
> **一句话**：低精度能减少存储与搬运，也可能提高硬件算力；是否加速取决于瓶颈、硬件支持和转换成本，scale 粒度则影响误差与元数据开销。

---

## 0. 本讲要解决的问题

**先修**：二进制位、绝对值、列表切片、整数除法；理解 `round` 的最近偶数舍入。必做先区分 FP 编码与 INT 均匀网格，再完成 HW1 Q6–Q8。FP8 intrinsic 和 GEMM 指令为选读。

1. FP32/FP16/BF16/FP8/FP4 的位布局、动态范围、精度各是什么？
2. 对称量化/非对称量化、per-tensor/per-token/per-group 的公式和误差？
3. 为什么 outlier 会毁掉 per-tensor？per-group 怎么缓解？
4. FP8/FP4 GEMM 的 scale 怎么乘？ue8m0 是什么？
5. KV cache 量化能省多少？精度代价怎么评估？

---

## 1. 数值格式

### 1.1 浮点数的组成

规格化的非零二进制浮点数（零、次正规数与特殊值另有编码）：

```
value = (-1)^sign * 1.mantissa * 2^(exponent - bias)
```

- 指数位决定**动态范围**；
- 尾数位决定**精度**；
- 符号位 1 bit。

| 格式 | 总位宽 | sign | exponent | mantissa | 动态范围（约） | 精度（约） |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| FP32 | 32 | 1 | 8 | 23 | 1e-38 – 3e38 | 1e-7 |
| FP16 | 16 | 1 | 5 | 10 | 6e-5 – 6e4 | 1e-3 |
| BF16 | 16 | 1 | 8 | 7 | 1e-38 – 3e38 | 1e-2 |
| FP8 E4M3（最大值448的变体） | 8 | 1 | 4 | 3 | ~1.56e-2 – 448 | ~0.1 |
| FP8 E5M2 | 8 | 1 | 5 | 2 | ~6e-5 – 57344 | ~0.25 |
| FP4 E2M1 | 4 | 1 | 2 | 1 | 1 – 6 | ~0.5 |

> 此表范围下限统一列最小规格化正数；若支持次正规数，还能表示更小的值，例如 E4M3 可到2⁻⁹、E2M1可到0.5。FP8 两种编码及特殊值的区别见 [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)；HW1 不要求实现这些浮点格式。

### 1.2 为什么 BF16 比 FP16 范围大

BF16 的指数位和 FP32 一样（8 bit），尾数只有 7 bit；FP16 指数 5 bit、尾数 10 bit。  
所以 BF16 的溢出风险较低，但相同量级附近间距较大。它在许多训练设置中无需 FP16 常用的 loss scaling；不代表任何计算都不会溢出。

### 1.3 MX 格式

MXFP8/MXFP4：

```
每 32 个元素共享一个 ue8m0 scale
```

`ue8m0`：8-bit 无符号指数，只有指数位，没有尾数/符号；普通有限编码表示 `2^(e-127)`，保留编码需依格式规范处理。  
好处：scale 本身很小（1B/32 元素），且乘 scale 可以用位移/指数运算。

---

## 2. 量化公式

### 2.1 均匀整数对称量化（INT，供 HW1 实现）

以下 q 是有符号整数编码，使用均匀网格。FP8/FP4 的编码则有指数和尾数，间距随指数改变；本节的 `round(x/scale)` 不能当作 FP8 cast。FP 格式概念与整数练习分别掌握。

```
qmax = 2^(bits-1) - 1
scale = amax / qmax
q = clamp(round(x / scale), -qmax, qmax)
x_hat = q * scale
```

- `amax = max(|x|)`；全零组需要单独约定正 scale，例如 HW1 使用1，编码全部为0；
- `round` 可以是 round-to-nearest-even；
- 对称量化的零点为 0，适合权重/KV（分布近似对称）。

### 2.2 非对称量化

```
scale = (max - min) / (2^bits - 1)
zero_point = round(-min / scale)
q = clamp(round(x / scale) + zero_point, 0, 2^bits - 1)
x_hat = (q - zero_point) * scale
```

适合非对称分布（如 ReLU 后的 activation）。

上述简式要求 min/max 覆盖零且不同，才能使 zero_point 位于编码区间。一般实现需先令 min=min(0,min(x))、max=max(0,max(x))，并约定常量/全零输入；本课不把这个边界未完善的片段作为 HW1 的必做接口。

### 2.3 粒度

| 粒度 | scale 数量 | 优点 | 缺点 |
| :--- | ---: | :--- | :--- |
| per-tensor | 1 | 简单 | outlier 毁掉整个 tensor |
| per-token | B*S | 对 activation 稳 | scale 数量多 |
| per-channel | C | 对权重稳 | 需要对应布局 |
| per-group (16/32/128) | N/G | 兼顾精度/开销 | scale 存储/读取开销 |

DeepSeek 的 Q&A：

- 主 KV：per-16 e4m3；
- indexer KV：per-32 ue8m0 MXFP4；
- MoE 路由专家权重：per-32 ue8m0 MXFP4。

---

## 3. 量化误差

### 3.1 最大绝对误差

均匀整数网格、round-to-nearest，且 x 位于可表示区间内、没有 clipping 时：

```
| x - x_hat | <= scale / 2
```

相对误差：

```
| x - x_hat | / |x| <= scale / (2|x|)
```

靠近 0 的元素相对误差大；靠近 amax 的元素相对误差小。

### 3.2 Outlier 的影响

假设一个 tensor 里 99% 的元素在 `[-1,1]`，一个 outlier 是 100：

```
amax = 100
scale = 100/127 ≈ 0.787
1.0 -> q=1 -> x_hat≈0.787，误差 0.213（21%）
```

per-group 把 outlier 隔离在一个 group 里，其他 group 的 scale 仍然小，误差显著降低。

### 3.3 误差上界（per-group）

对 group `g`：

```
| x - x_hat | <= amax_g / (2 * qmax)
```

细分一个原组时，各子组的 amax 不会增大，因此这项最坏误差上界不会增大；单个数值的实际舍入误差不保证单调降低。scale 数量/存储会增加。

### 3.4 量化误差 vs 模型精度

- 权重量化：误差会累积到 logits；
- KV 量化：影响 attention 的 softmax，长上下文更敏感；
- activation 量化：受 outlier 影响最大；
- 需要 end-to-end 评估（PPL、任务指标），不能只看单算子误差。

---

## 4. FP8/FP4 GEMM 的 scale

### 4.1 Per-tensor scale

```
C = (A * sA) @ (B * sB) = (A @ B) * (sA * sB)
```

所以 GEMM 可以用整数/FP8 输入，最后乘 `sA*sB`。

### 4.2 Per-group scale

对 K 维分组：

```
A[m, k] = A_q[m, k] * sA[m, k//G]
B[k, n] = B_q[k, n] * sB[k//G, n]
C[m, n] = Σ_k A_q * B_q * sA * sB
```

如果 `sA` 沿 K 变化，不能简单提到 GEMM 外面；需要在累加时按 group 乘 scale，或者用“scale 预乘到 A/B”再 GEMM。

DeepGEMM 的做法：

- 把 scale 转换成特定 layout（如 TMA-aligned col-major）；
- 在 epilogue 或 mainloop 中按 group 乘；
- 用 tensor core 的 FP8/FP4 指令；
- 对 FP4 使用 packed 格式（两个 4-bit 一个 byte）。

### 4.3 累加精度

- FP8/FP4 输入，FP32 累加；
- 如果累加也用 FP16，误差会很大；
- 对长 K（如 7168），FP32 累加是必须的。

---

## 5. KV Cache 量化

### 5.1 收益

bf16 -> FP8：

```
数据 payload bytes 减半（scale/对齐另计）
```

bf16 -> FP4：

```
KV bytes 减到 1/4（加 scale 开销）
```

例：`L=32, H_kv=8, D=128, S=4096, B=1`：

- bf16：512 MiB；
- FP8：256 MiB；
- FP4：128 MiB（+ scale）。

**含元数据的手算**：MXFP4 的32个数占16B payload，加1B scale，共17B；BF16 同组为64B。压缩比是 `64/17≈3.76`，不是4。若尾组也分配满32个槽或 scale 有对齐，实际比例还会变小。

### 5.2 精度风险

- KV 的 outlier 比权重更难处理；
- attention 的 softmax 对 K 的误差敏感；
- 长上下文时，误差会累积；
- 需要 per-16/per-32 的 scale，甚至混合精度（关键层 bf16）。

### 5.3 评估方法

- 单算子：cosine similarity、max abs error；
- 端到端：PPL、任务准确率、长上下文 retrieval；
- 消融：只量化 K、只量化 V、不同 group size。

---

## 6. 代码：per-group 量化

下面是可在标准库 Python 运行的**均匀 INT**小例子，说明 group 遍历；并非 FP8 cast。约定有限数值、group_size为正整数、bits为2–8整数；简例只覆盖这些有效输入，完整参数校验与反量化长度检查按 HW1 规格实现。

```python
def quantize_per_group(x, group_size=32, bits=8):
    qmax = 2 ** (bits - 1) - 1
    q = []
    scales = []
    for g in range(0, len(x), group_size):
        chunk = x[g:g+group_size]
        amax = max(abs(v) for v in chunk)
        scale = amax / qmax if amax else 1.0
        scales.append(scale)
        q.extend(max(-qmax, min(qmax, round(v / scale))) for v in chunk)
    return q, scales

def dequantize_per_group(q, scales, group_size=32):
    out = []
    for i, v in enumerate(q):
        out.append(v * scales[i // group_size])
    return out
```

误差：

```python
x = [-1.0, 0.0, 1.0, 100.0]
q, scales = quantize_per_group(x, group_size=2)
x_hat = dequantize_per_group(q, scales, group_size=2)
assert len(x) == len(x_hat)
err = max((abs(a - b) for a, b in zip(x, x_hat)), default=0.0)
print(q, scales, x_hat, err)
```

预测第一组 q=`[-127,0]`、scale=1/127，第二组 q=`[1,127]`、scale=100/127；重建第三个数约0.7874，最大误差约0.2126。改变 group_size 时量化与反量化必须用相同分组。

### FP8 cast 的注意点

- E4M3 的 max 是 448；超过要 clamp；
- 次正规数/NaN/Inf 的编码；
- `round` 的模式（nearest-even）；
- scale 的 layout（row-major/col-major/packed）；
- 使用 `__nv_cvt_float_to_fp8` 时显式选择格式与饱和策略，并对照 [CUDA FP8 转换文档](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html) 的最近偶数舍入契约；FP4 对应 [CUDA FP4 转换文档](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP4__MISC.html)。

---

## 7. 仓库对照

- `deepinfer/model_executor/layers/tilelang_ops/common.py`：FP4/FP8 常数、`customized_finfo_amax`。
- `.../tilelang_ops/cuda/quant.py`：`ue8m0_exp_scale`、scale 计算。
- `.../tilelang_ops/cuda/norm.py`：`do_fp8_cast` 分支，norm + quant 融合。
- `.../tilelang_ops/cuda/fused_transform.py`：norm + rope + cast + KV store。
- `doc/inference_deployment_qa.md`：Low-precision computation。

---

## 8. 练习

### 基础

**修错题**：所有输入为0时直接写 `scale=amax/qmax`，再执行 `round(x/scale)` 会怎样？将零组编码为零、scale设为1，为什么仍能完全重建？

<details><summary>核对</summary>

scale为0导致除零；约定scale=1后，q仍全0，反量化0×1仍为0。该约定保证实现合法而不改变结果。随后独立测试尾组（长度不是G倍数）、全零组与含outlier组，对齐 HW1 的空输入和长度契约。

</details>

1. 对 `x=[-1,0,1]`，8-bit 对称量化，求 scale、q、dequant、误差。
2. 对 `x=[-1,0,1,100]`，per-tensor vs per-group（G=2）的误差各是多少？
3. 计算 bf16/FP8/FP4 的 KV bytes（给定 L/H/D/S/B）。
4. 解释 ue8m0 的表示范围；为什么它适合做 block scale？

### 进阶

5. 实现 per-tensor/per-token/per-group 量化，测随机正态和长尾分布的误差。
6. 实现 FP8 E4M3 的 cast（可以先用 Python 模拟），和 PyTorch 的 FP8 对比。
7. 在 GEMM 中加 per-tensor scale，验证 `(A*sA)@(B*sB)` 和 `(A@B)*sA*sB` 在浮点容差内相等；sA/sB 是标量，不能写成矩阵乘 `A@sA`。
8. 用 `ncu` 看 FP8 GEMM 的 tensor core 利用率和带宽，和 BF16 对比。

### 挑战

9. 实现 per-group FP8 GEMM 的 scale 乘法，比较“预乘 scale”和“epilogue 乘”的误差/性能。
10. 模拟 KV 量化：只量化 K/V，测 attention 输出误差和 PPL 变化。

---

## 9. 面试追问

1. FP8 E4M3 和 E5M2 有什么区别？分别适合什么？
2. 为什么 per-tensor 对 outlier 敏感？per-group 怎么缓解？
3. 对称量化的最大绝对误差是多少？相对误差呢？
4. FP8/FP4 GEMM 的 scale 怎么乘？为什么不能简单提到 GEMM 外面？
5. KV cache 量化能省多少？精度风险在哪？
6. 如果量化后 PPL 变差，你会先查什么？

---

## 10. 参考文献

- NVIDIA, *FP8 Formats for Deep Learning*；Hopper/Blackwell 文档。
- DeepSeek-V3：FP8 训练/推理。
- Micikevicius et al., *FP8 Formats for Deep Learning*。
- DeepInfer Q&A：Low-precision computation。

---

## 11. 下一步

- 完成 [作业 1](../hw/hw01/README.md) 的 Q6–Q8；
- 读 [项目 1](../projects/project1/README.md) 的 roofline 部分；
- 进入 Unit 2：PagedAttention、continuous batching、sampling。
