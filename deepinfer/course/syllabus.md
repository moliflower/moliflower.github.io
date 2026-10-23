# Syllabus

> **课程**：DeepInfer 101: From Kernels to Inference Systems  
> **版本**：2026-10-22  
> **节奏**：15 周课程蓝图，12–15 小时/周；起步另计，按掌握度延长。当前可执行材料到 Unit1。  
> **入口**：零基础先做 [Python 起步](foundations/README.md)；会函数、循环、列表后做调试预备。不要求先学 C++ 或 CUDA。  
> **硬件**：Unit 1 不需要 GPU；GPU 轨道可选

---

## 1. 课程描述

这门课讲两件事：

1. **算子**：GPU 上的 vector add、reduction、softmax、RMSNorm、GEMM、attention 怎么写、怎么快、怎么分析。
2. **推理系统**：KV cache、PagedAttention、continuous batching、chunked prefill、MoE/EP、投机解码、异步调度。

课程以 DeepInfer（DeepSeek 的开源推理框架）为“工业级参考实现”，但**不要求你运行完整 DeepInfer**。做法是：

- 用纯 Python 的 SIMT 模拟器复现机制；
- 用 roofline 复现性能模型；
- 用仓库里的真实代码做对照阅读；
- 用 Project 把多个机制组合成一个可运行的小系统。

> 这门课不是“CUDA API 大全”，也不是“DeepInfer 源码逐行讲解”。它是一条**从写对到写快、从算子到系统**的路径。

---

## 2. 学习成果

入门预备单元的目标：读 traceback、构造最小复现、检查列表副本与矩阵索引、编写浮点断言。完成后进入第一讲与 Lab0 的纯 Python 核心。

**分层约定**：Python 手算与实现为核心；形式化推导逐步深入；CUDA/TileLang、occupancy、profiling 和工业源码是进阶轨道。后续章节仍在按这一入口标准审校。

学完 Unit 1（Week 1–4），你应该能：

- 解释 grid/block/warp/thread、coalescing、shared memory、bank conflict、occupancy；
- 独立写出 vector add、tree reduction、softmax、RMSNorm、naive/tiled matmul；
- 用 roofline 判断 memory-bound / compute-bound，并估算理论时间；
- 计算 KV cache 大小，解释 MLA 为什么省显存；
- 解释 per-tensor/per-token/per-group 量化的差别和误差来源。

学完 Unit 2（Week 5–8），你应该能：

- 解释 PagedAttention、block table、prefix cache、continuous batching、chunked prefill；
- 写一个 mini inference engine（paged KV + batching + sampling）；
- 解释 TTFT/TPOT/goodput 和调度 trade-off。

学完 Unit 3（Week 9–12），你应该能：

- 解释 MoE top-k gate、grouped GEMM、EP、EPLB；
- 解释投机解码的接受/拒绝、变长验证；
- 解释 DP/TP/PP/EP/CP 和 all-reduce/all-to-all；
- 用 nsys/ncu 定位一个 kernel 的瓶颈。

---

## 3. 课程形式

### Lecture（30–45 分钟）

每节课有：

- **Before class**：推荐的视频/阅读（不要求全看，按你的基础选）；
- **Notes**：核心概念、公式、ASCII 图；
- **仓库对照**：去 DeepInfer 里看哪几个文件；
- **Checkpoint**：3 个问题，做完 Lab 后回答。

### Lab（按题推进，可分多次完成）

- Guided，测试驱动；
- 每个 Lab 有 `README.md`、starter code、`test_*.py`；
- 目标不是“跑通”，而是“能解释每个数字”。

### Homework（3–5 小时）

- 比 Lab 难，偏数学 + 小函数；
- 有 checkoff 问题；
- 允许讨论，但必须自己写。

### Project（2–3 周）

- 分 phase，有 checkpoint；
- 有测试、报告、5 分钟讲解；
- Project 1 是 Kernel Zoo；Project 2 是 Mini Engine；Project 3 是 EPLB Simulator。

### Checkoff（10 分钟）

每个 Lab/Homework/Project 做完后，回答 3 个问题，写在 `notes/` 里。要求：

- 不看笔记，能讲 2–3 分钟；
- 能举一个数字例子；
- 能回答“如果换成 X 会怎样”。

> 代码测试通过后，还需要独立解释。对照 [掌握度 rubric](ASSESSMENT.md)，用一个新输入检查理解。

---

## 4. 评分（自评）

| 项目 | 分数 | 依据 |
| :--- | ---: | :--- |
| Lab0–3 | 40 | 每个 10 分：代码、解释、调试、迁移 |
| HW1–2 | 20 | 每个 10 分：代码、推导、反例、checkoff |
| Project1 | 30 | 项目内部 100 分 rubric 乘 0.3 |
| Quiz1–2 | 10 | 两份原始成绩合计乘 0.5 |
| 总分 | 100 | 只计算已发布的 Unit1 |

详细规则见 [掌握度与评分](ASSESSMENT.md)。Python 起步与调试预备不计分；Unit2–4 是后续计划，不进入当前分母。自动测试与浏览器勾选都不能代替书面推导和口头解释。

---

## 5. 迟交与节奏

这是 self-paced 课程，没有真正的 deadline。建议：

- 每周日晚上交本周 Lab/Homework；
- Project 每个 checkpoint 给自己一个日期；
- 如果落后超过 1 周，优先补 Lab，不要跳着做 Project。

**慢速版**：把每个 Week 拆成两周，总时长 30 周。不要为了赶进度跳过 Lab。

---

## 6. 协作

| 类型 | 规则 |
| :--- | :--- |
| Lecture / 笔记 | 随便分享 |
| Lab | 可以讨论、互相看代码；但每个人要自己敲一遍 |
| Homework | 可以讨论思路；代码自己写；checkoff 自己答 |
| Project | 可以找伙伴 pair；但每个人要能讲清全部代码 |
| Checkoff | 必须独立回答，不能看答案 |

**不要做的事**：

- 不要把别人的 Lab/Homework 代码直接复制；
- 不要在看不懂的情况下让 AI 生成整段实现；
- 不要在 Project 里用别人的 kernel 冒充自己的。

> 参考解用来核对思路。把独立完成和参考后修复的部分记清楚，复习时就知道该练哪里。

---

## 7. AI 使用政策

AI 是很好的 tutor，但很容易变成“代写”。建议：

**可以用 AI：**

- 解释一段代码/论文；
- 帮你画调用图、列大纲；
- 在你写完 kernel 后 review；
- 生成测试用例的边界条件；
- 解释报错。

**不要用 AI：**

- 直接生成 Lab/Homework/Project 的完整实现；
- 在你还没尝试之前问答案；
- 让 AI 替你写 checkoff 回答；
- 把 AI 生成的代码贴进项目而不理解。

**一个简单规则**：如果你不能在 5 分钟内给一个朋友讲清这段代码，就不要提交它。

---

## 8. 提交

每个 assignment 做完后：

```bash
cd /path/to/DeepInfer
python course/tools/grade.py lab00
```

然后在 `notes/` 写 checkoff，commit：

```bash
git add course/labs/lab00/lab00.py notes/lab00.md
git commit -m "lab00: vector add + coalescing"
```

建议开一个学习分支：

```bash
git switch -c study/deepinfer-101
```

不要把答案提交到主分支，也不要把 `notes/` 放进公开仓库（如果你不想公开）。

---

## 9. 求助

- 先看 Lab 的 `README.md` 和测试的 assertion message；
- 再读仓库对应文件（每节课有“仓库对照”）；
- 还不行，把问题写清楚：**你期望什么、实际什么、最小复现是什么**；
- 问 AI 时，附上代码、报错、你试过的方案。

> “我的代码不 work”不是问题描述；“我希望 `vector_add([1,2],[3,4])` 返回 `[4,6]`，实际返回 `[4,6,0]`，因为我把 grid 算成了 `n/block+1`”才是。

---

## 10. 教材与资源

### 主教材

- [教材索引](TEXTBOOK.md) 中的 Python 起步、调试预备与 lecture notes；
- DeepInfer 仓库的 `doc/inference_deployment_qa.md`；
- `doc/study_plan_zh.md` 的长期路线图。

### 参考书

- *Programming Massively Parallel Processors*（PMPP），第 1–6 章；
- *Computer Architecture: A Quantitative Approach*，GPU 章节；
- *CUDA C++ Programming Guide*。

### 论文（按课程进度）

- FlashAttention / FlashAttention-2；
- PagedAttention / vLLM；
- Orca、Sarathi-Serve；
- DeepSeek-V2（MLA）、DeepSeek-V3（MoE/FP8）；
- DeepEP / DeepGEMM / FlashMLA 的 README；
- DistServe / Splitwise / Mooncake（P/D 分离）。

### 视频

网站视频页按主题提供链接，当前新手入口不依赖外部视频。原则：

- 先看 1.5× 的概览，再回到文字；
- 不要刷完 40 小时视频才开始写代码；
- 一个主题最多看 2 个来源，避免“视频收集癖”。

---

## 11. 这门课不做什么

- 不教你 CUDA 的所有 API；
- 不逐行讲 DeepInfer 的 5 万行代码；
- 不要求你跑通 V4.1-Flash（你的机器跑不动）；
- 不承诺 15 周后你能通过所有面试；它只是把你的基础打牢。

---

## 12. 一句话

**先写对，再写快，再讲清楚。**  
规格明确、测试可解释、实验可复现，项目才有可展示的证据。
