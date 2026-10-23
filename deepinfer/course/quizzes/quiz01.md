# Quiz 1: GPU Model, Roofline, Softmax, Indexing

> **时间**：10 分钟  
> **规则**：闭卷，不看笔记，不用 AI。  
> **目的**：检查你是否真的理解了 Week 1–2 的核心数字，而不是只会跑测试。

---

## Q1（2 分）

一个 warp 的 32 个 thread 读取 32 个连续的 `int32`，起始地址是 word 16（不是 0）。  
本题采用 Lab0 的简化模型：以 128 字节（32 个 word）的 line 为一次统计单位。触及几条 line？如果改用 32B sector 统计，又触及几个？这里不把 line 计数称为真实 GPU 的 transaction 数。

## Q2（2 分）

`vector_add` 处理 `n = 1,000,000` 个 float32：

- 搬多少字节？
- 做多少 FLOPs？
- 如果带宽 500GB/s、算力 10TFLOP/s，它是 memory-bound 还是 compute-bound？预测时间是多少？

## Q3（2 分）

一个 4096 维 float32 的 softmax：

- 3-pass 搬多少字节？
- 2-pass 搬多少字节？
- 采用 Lab1 的 activation-only 口径，分别为 `16n` 与 `12n` 字节；在 2TB/s 带宽下，两者的带宽时间下界差多少？这能直接当实测提速吗？

## Q4（2 分）

`n=10, block=4`，需要几个 block？最后一个 block 的 tid=0..3 分别处理哪些 i，哪些应该跳过？

## Q5（2 分）

函数应该返回一个新列表，不改变输入：

```python
def scale(values):
    result = values
    for i in range(len(result)):
        result[i] = result[i] * 2
    return result
```

输入 `a=[2,3]`，调用后 `a` 是什么？指出契约被破坏的原因，并说明如何修复。先手算，不运行。

---

<details>
<summary>答案（做完再看）</summary>

### A1

2 条 line。word 地址 16..47 跨 line 0（0..31）和 line 1（32..63）。对应 byte 64..191，共 4 个 32B sector。两种数字统计的单位不同。

### A2

- Bytes = `3 * n * 4 = 12,000,000` 字节 ≈ 12MB。
- FLOPs = `n = 1,000,000`（一次加法）。
- memory time = `12e6 / 500e9 = 24e-6 s = 24us`。
- compute time = `1e6 / 10e12 = 0.1e-6 s = 0.1us`。
- 在此理想模型下 memory-bound，时间下界 24us；忽略 launch、缓存和同步等成本。

### A3

- 3-pass = `16 * 4096 = 65,536` 字节 = 64 KiB。
- 2-pass = `12 * 4096 = 49,152` 字节 = 48 KiB。
- 差 16,384 字节；在 2TB/s 下 `16384 / 2e12 = 8.192e-9 s = 8.192ns`。
- 这是所选流量模型的带宽时间下界差；小向量还受启动开销、缓存等影响，不能作为实测提速。

### A4

需要 3 个 block。最后一个 bid=2，i 为 8、9、10、11；只处理 8、9，跳过 10、11。guard 是 `i<n`。

### A5

调用后 `a` 是 `[4,6]`，因为 `result` 与 `values` 指向同一个列表。可先建新列表，再逐项追加翻倍后的元素；或先复制这个一维数值列表，再改副本。

</details>

---

## 评分

- 8–10 分：Week1–2 核心已掌握，可以继续下一阶段。
- 5–7 分：回看 Lecture 2/3 的对应小节，再做一遍 Lab。
- 0–4 分：不要赶进度；先把 Lab 0/1 的 checkoff 问题讲一遍。
