# 15-Week Calendar

> 每周 12–15 小时。**Lecture 在前，Lab 在中，Homework/Project 在后。**  
> 标注“待发布”的材料尚无完整代码与验收，不承诺按日期自动补充。已发布材料按先修推进，日历日期仅为示例。

---

## Unit 1: GPU Kernels（Week 1–4）

**零基础入口**：[Python 起步 0–3](foundations/README.md)，先学习变量、函数、条件、循环和列表，再进入调试预备。

**入门前置**：[预备 0–3](prep/README.md)，练运行、调试、矩阵与测试，不要求 C++。正式第 1 周以第一讲核心和 Lab0 为主。CUDA/occupancy/profiling 等为选读，后续材料继续审校。

**目标**：建立 GPU 心智模型，能写 memory-bound kernel，能算 roofline，能解释 attention/KV 的代价。

| Week | Lecture | Lab | Homework | Project | 本周产出 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | [L1 Python 执行模型](lectures/lec01.md) · [L2 访存推导](lectures/lec02.md) | [Lab0 Q0–Q6](labs/lab00/README.md) | Lab0 Q4 基础 Roofline | [P1](projects/project1/README.md) 只预览 | 调试记录 + Lab0 |
| **2** | [L3 Reduction & softmax](lectures/lec03.md) · [L4 Tiling & GEMM](lectures/lec04.md) | [Lab1](labs/lab01/README.md) | [Quiz1](quizzes/quiz01.md) | P1 Phase1 | 数值算子 |
| **3** | [L5 Attention](lectures/lec05.md) · [L6 FlashAttention](lectures/lec06.md) | [Lab2](labs/lab02/README.md) | Lab2 Q7/Q8 代价练习 | P1 Phase2 | Attention 参考实现 |
| **4** | [L7 KV cache](lectures/lec07.md) · [L8 Quantization](lectures/lec08.md) | [Lab3](labs/lab03/README.md) | [HW1](hw/hw01/README.md) · [HW2](hw/hw02/README.md) · [Quiz2](quizzes/quiz02.md) | P1 报告 | 可分多周完成，先修齐全再做 |

### Week 1 详细

- **Lecture 1**：Python 手算、编号映射、尾部判断、word 地址、line/sector、时间下界。
- **Lecture 2**：只先读访存推导；shared、occupancy 和 CUDA 练习可后置。
- **Lab0**：Q0 运行示例，Q1/Q2 填索引，Q3 统计，Q4 模型，Q5 独立算子，Q6 修错。
- **HW**：本周不做尚未学过的 KV、attention、量化题，HW1/2 移到第 4 周。
- **Project1**：只看目标，不要求实现 reduction/softmax 等未学内容。
- **Checkoff**：解释尾部、一次读写的地址、line/sector 的区别。

### Week 2 详细

- **Lecture 3**：tree reduction、warp shuffle、online softmax、RMSNorm/LayerNorm、traffic。
- **Lecture 4**：GEMM 的 FLOPs/AI、tiling、register blocking、tensor core 简介。
- **Lab 1**：sum_reduce、online softmax、softmax、rmsnorm、layernorm、traffic 计算。
- **练习**：Lab1 数值与 traffic 题，Quiz1 不提前考 KV；HW2 后置。
- **Project 1**：完成 Phase 1，开始 Phase 2 的 `matmul_tiled`/`matmul_traffic`。
- **Checkoff**：为什么 softmax 要减 max？tiled matmul 的 traffic 为什么降到 1/tile？

### Week 3 详细

- **Lecture 5**：attention 的数学、Q/K/V、multi-head、causal mask。
- **Lecture 6**：FlashAttention 的 online softmax、tiling、IO complexity；decode attention 为什么 memory-bound。
- **Lab 2**：实现 matmul/transpose/softmax/causal mask/attention/multi-head，算 FLOPs 和 bytes。
- **代价练习**：先做 Lab2 Q7/Q8；完整 HW2 在 KV 讲完后做。
- **Project 1**：Phase 2 完成，开始写 report 的 roofline 表。
- **Checkoff**：prefill 和 decode 的 attention 分别是什么 bound？为什么？

### Week 4 详细

- **Lecture 7**：KV cache 基础：为什么 decode 需要 cache、K/V 布局、容量计算；PagedAttention 留到 Week 5。
- **Lecture 8**：FP8/FP4、per-tensor/per-token/per-group、scale 布局、误差。
- **Lab 3**：实现连续布局 + paged 布局的 KV cache（append/get/free），算 page waste。
- **HW1/2**：先修齐备后做 Roofline、KV、量化与 attention 代价，可把本阶段拆成多周。
- **Project 1**：完成 report，做 5 分钟讲解；可选 GPU 移植。
- **Checkoff**：decode 为什么需要 KV cache？为什么长上下文会把 KV cache 变成瓶颈？

---

## Unit 2: Inference Engine（Week 5–8）

**目标**：理解并实现一个 mini inference engine 的核心：KV cache、batching、chunked prefill、sampling。

| Week | Lecture | Lab | Homework | Project | 本周产出 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **5** | [L9 共享块池](lectures/lec09.md) · [L10 前缀缓存](lectures/lec10.md) | [Lab4](labs/lab04/README.md) | [HW5](hw/hw05/README.md) | [Project2 Phase1](projects/project2/README.md) | 有限块池 + FIFO 请求轨迹 |
| **6** | [L11 Continuous batching](lectures/lec11.md) · [L12 Chunked prefill](lectures/lec12.md) | [Lab5](labs/lab05/README.md) | [HW6](hw/hw06/README.md) | [P2 Phase2](projects/project2_scheduler/README.md) | 调度模拟器 + 18 组实验 |
| **7** | L13 Sampling · L14 Speculative decoding（待发布） | Lab 6: Spec decode（待发布） | HW 7: Sampling（待发布） | P2 checkpoint 2 | 采样 + 投机解码 |
| **8** | L15 Engine architecture · L16 Async/CUDA Graph（待发布） | Lab 7: Engine trace（待发布） | HW 8: Engine（待发布） | **P2 due** | Mini Engine + benchmark |

### Week 5–6 重点

- KV cache 的 allocate/free/evict；
- continuous batching vs static batching；
- chunked prefill 的 chunk size 和 TTFT/TPOT trade-off；
- prefix cache 的命中率和 eviction；
- 用离散事件模拟器测 throughput/latency。

### Week 7–8 重点

- sampling：top-k/top-p/temperature/repetition penalty；
- speculative decoding：draft/target、acceptance、verification；
- 异步调度：schedule/forward/sample/post-process 的重叠；
- CUDA Graph：capture/replay、batch size 对齐；
- 把 mini engine 的吞吐和 HF `generate` 对比。

---

## Unit 3: MoE, Distributed, and Performance（Week 9–12）

**目标**：理解大模型推理的规模化：MoE/EP、通信、profiling、论文与开源。

| Week | Lecture | Lab | Homework | Project | 本周产出 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **9** | L17 MoE · L18 EP/EPLB（待发布） | Lab 8: Top-k gate（待发布） | HW 9: MoE（待发布） | Project 3: EPLB Simulator 发布 | MoE 参考实现 |
| **10** | L19 Parallelism · L20 Communication（待发布） | Lab 9: All-reduce（待发布） | HW 10: Communication（待发布） | P3 checkpoint 1 | 通信模拟 |
| **11** | L21 Profiling · L22 Benchmarking（待发布） | Lab 10: Profile（待发布） | HW 11: Benchmark（待发布） | P3 checkpoint 2 | Profile 报告 |
| **12** | L23 Paper: DeepSeek-V3 · L24 Paper: vLLM（待发布） | Lab 11: Paper reproduction / PR（待发布） | HW 12: Paper notes（待发布） | **P3 due** | 1 个 PR / issue |

### Week 9–10 重点

- MoE top-k gate、token permutation、grouped GEMM；
- EPLB 的 logical/physical expert、冗余专家、greedy 重排；
- DP/TP/PP/EP/CP 的区别；
- all-reduce/all-gather/all-to-all 的通信量；
- 画一张 64 卡 Wide-EP 的 decode 数据流。

### Week 11–12 重点

- nsys/ncu 的读数：DRAM throughput、occupancy、stall；
- 读 DeepSeek-V3、vLLM PagedAttention、Sarathi-Serve；
- 给 DeepInfer/vLLM/SGLang 提一个文档/测试/small fix；
- 把 P3 的报告写成博客。

---

## Unit 4: Final Project & Interview（Week 13–15）

**目标**：把前 12 周的能力组合成一个可展示的作品，并准备面试。

| Week | 内容 | 产出 |
| :--- | :--- | :--- |
| **13** | 选 final project；写 proposal；补 Unit 1–3 的漏洞 | 1 页 proposal |
| **14** | 实现 + benchmark；模拟面试（项目 + 八股 + 手撕） | 可运行 demo + 面试记录 |
| **15** | 最终演示；写博客；更新简历 | 15 分钟讲解 + 博客 + 简历 bullet |

### Final Project 选题（三选一）

1. **Mini inference engine**：paged KV + continuous batching + prefix cache + speculative decoding，用 Qwen2.5-0.5B 跑通。
2. **Kernel Zoo GPU 版**：把 Lab/Project 1 的 4 个 kernel 移植到 TileLang/CUDA，做 benchmark 和 roofline 对比。
3. **EPLB Simulator**：合成/真实 trace，比较 random/greedy/DeepInfer 策略，写报告。

### 面试准备

- 高频题清单见 [`../doc/study_plan_zh.md`](../doc/study_plan_zh.md) 第 7.3 节；
- 每周 3 道手撕（reduction/softmax/GEMM/attention/top-k/LRU）；
- 每两周一次模拟面试；
- 把 Project 1/2/3 写成 STAR 故事，带数字。

---

## 每周检查表

每周日晚上问自己：

- [ ] 本周 Lab 的测试全绿了吗？
- [ ] 我能不看笔记讲清本周的 3 个 checkoff 问题吗？
- [ ] 我能解释一个循环不变式，并给错误实现构造一个反例吗？
- [ ] 我记录了哪些实验数字？是否标明单位、输入、模型假设和复现命令？
- [ ] 我下周的第一个任务是什么？

如果连续两周有三个“否”，把节奏降到慢速版（30 周），不要硬赶。
