# Quiz 2: Attention, KV Cache, Quantization

> **时间**：10 分钟  
> **规则**：闭卷，不看笔记，不用 AI。  
> **目的**：检查 Unit 1 后半段的核心数字。

---

## Q1（2 分）

`batch=2, heads=8, seq=512, head_dim=64`。只计 QK 与 PV 的乘加（一次乘加计 2 FLOPs），含对角线、真正跳过上三角，causal attention 的 FLOPs 是多少？如果先算完整矩阵再 mask，计算量是否相同？

## Q2（2 分）

32 层、8 个 KV head、head_dim=128、seq_len=4096、batch=4、FP8（1 byte）的 KV cache 多大？

## Q3（2 分）

`seq_len=5000, page_size=64`：

- 需要多少个 page？
- 最后一个 page 浪费多少 slot？

## Q4（2 分）

`batch=1, heads=8, seq=512, head_dim=64, bf16` 的 attention：

- 三阶段 materialized 算法的请求流量多少 bytes？约定 Q/K/V/O 各访问一次，S 写/读、P 写/读各一次。
- 不落地 S/P、Q/K/V/O 各访问一次的理想流量下界多少 bytes？它不等于真实 FlashAttention 的设备流量。
- 差多少？

## Q5（2 分）

MLA 的 KV cache 公式为什么不乘 2？它存的是什么？

---

<details>
<summary>答案（做完再看）</summary>

### A1

```
有效配对 = 512 * 513 / 2 = 131,328
FLOPs = 4 * 2 * 8 * 131,328 * 64
      = 537,919,488
```

先做稠密矩阵乘再 mask 的实现仍执行 `4*b*h*s*s*d = 1,073,741,824` FLOPs；“除以二”只是在长序列时忽略对角线的近似，不能当本题精确答案。

### A2

```
2 * 32 * 8 * 128 * 4096 * 4 * 1
= 1,073,741,824 bytes = 1 GiB
```

### A3

- pages = ceil(5000 / 64) = 79；
- allocated = 79 * 64 = 5056；
- waste = 56 slots。

### A4

- Q/K/V/O = `8 * (2*512 + 2*512) * 64 = 1,048,576` 元素；
- 一个 S 或 P = `8 * 512 * 512 = 2,097,152` 元素；
- materialized 请求流量 = `(1,048,576 + 4*2,097,152) * 2 = 18,874,368` bytes；
- 理想下界 = `1,048,576 * 2 = 2,097,152` bytes；
- 差 `16,777,216` bytes = 16 MiB。容量是要存多少，流量是读写搬了多少；不能只加一次 score 容量代替四次中间读写。

### A5

MLA 只存一份压缩 latent（`kv_lora_rank`）加上 RoPE 部分（`qk_rope_head_dim`）；K 和 V 共享同一份 latent，所以没有 K/V 两份，不乘 2。

</details>

---

## 评分

- 8–10 分：可做 Unit1 综合验收；Unit2 仍为待发布计划。
- 5–7 分：回看 Lecture 6/7/8 的对应公式。
- 0–4 分：先重做 Lab 2/3 和 HW 2。
