# DeepInfer 101: From Kernels to Inference Systems

> **从算子到推理系统** · self-paced 课程 · 15 周 · 每周 12–15 小时  
> 灵感来自 UC Berkeley CS61A：**lecture → lab → homework → project → checkoff**。  
> 长期路线图见 [`../doc/study_plan_zh.md`](../doc/study_plan_zh.md)；这里是它的“课程执行版”。  
> **本地课程网站**：[`site/index.html`](site/index.html)（双击打开，或运行 `site/open.bat`）。网站包含**排版好的讲义/实验/作业页面**和**精选 B 站视频页**。  
> **零基础入口**：先做 [Python 起步](foundations/README.md)，再做 [调试预备](prep/README.md)。已会函数、列表与循环可直接进入预备课。GPU 扩展不计入入门验收。
> **教材主线**：[教材索引与术语表](TEXTBOOK.md) · **验收**：[掌握度与评分](ASSESSMENT.md)。
> **本次教材重整**：不新增周次；起步、预备与第 1–6 周重新梳理。先读 [自学路线与诊断入口](LEARNING_PATH.md)，再按“问题 → 图解 → 推导 → 手算 → 最小实现 → 练习 → 自测”推进。已有题号、接口、评分与学习标记保留。

---

## Start Here（先获得一个确定能运行的起点）

1. 读 [Python 起步](foundations/README.md)，运行 `python course/foundations/demo.py`；Windows 可用 `.\course\run.ps1 --demo`。完成起步 Q0–Q4。
2. 做 [`预备 0–3`](prep/README.md)，不要求 C++ 或 GPU。
3. 读 [`第一讲核心`](lectures/lec01.md)，先填编号与地址表。
4. 做 [`Lab0 Q0–Q6`](labs/lab00/README.md)，用 `--question q1 --failfast` 按题运行。
5. 记录一次调试过程，再做出口检查，不急着安装 CUDA。

以上命令从能够直接看到 `course` 文件夹的课程根目录运行。ZIP 使用解压根目录；个人主页源码仓库先执行 `cd _courses/deepinfer`。若环境只有 `python3`，统一替换命令名。

> 不需要 GPU。Unit 1 的核心轨道是纯 Python；GPU 是可选加速轨道。

---

## 这门课怎么运作

参考 CS61A 的分层练习与项目设计，本课使用“阅读 → 预测 → 实现 → 测试 → 解释”的循环。它是自学课程，不代表 Berkeley 官方课程或质量认证：

| 环节 | 形式 | 目的 |
| :--- | :--- | :--- |
| **Lecture** | 分段阅读、手算、预测、推导 | 将公式中的量连接到代码变量 |
| **Lab** | 示例 → 填空 → 独立实现 → 修错，可分多次 | 把概念变成能跑的代码 |
| **Homework** | 3–5 小时，偏数学 + 小函数 | 补上“为什么”和边界情况 |
| **Project** | 2–3 周，分 phase + checkpoint | 把多个概念组合成一个作品 |
| **Checkoff** | 3 个问题，写在 `notes/` | 防止“看了等于会了” |
| **Quiz** | 10 分钟，闭卷，每周/每两周 | 检查核心数字和直觉 |
| **Autograder** | `python course/tools/grade.py` | 本地即时反馈，不依赖平台 |

### 每周节奏

```
周一–周二：看 lecture + 读材料（1.5h）
周三–周四：做 lab（2h）
周五–周日：做 homework / project（4–6h）
周日晚上：写 notes + checkoff（1h）
```

**输入 : 输出 = 1 : 2**。看 1 小时视频，至少写 2 小时代码。

---

## 15 周日历

| 周 | Lecture | Lab | Homework | Project |
| :--- | :--- | :--- | :--- | :--- |
| 起步 | [表达式、函数、条件、列表与循环](foundations/README.md) | foundations Q0–Q4 | 独立变式与解释 | 不计入 15 周 |
| 预备 | [运行、调试、矩阵、测试](prep/README.md) | prep Q0–Q3 | 不加额外作业 | 不计入 15 周 |
| 1 | [L1 Python 执行模型](lectures/lec01.md) · [L2 访存推导](lectures/lec02.md)（CUDA 等选读） | [Lab0 Q0–Q6](labs/lab00/README.md) | Lab0 Q4 基础 Roofline | [P1](projects/project1/README.md) 只预览 |
| 2 | [L3 Reduction & softmax](lectures/lec03.md) · [L4 Tiling & GEMM](lectures/lec04.md) | [Lab1](labs/lab01/README.md) | [Quiz1](quizzes/quiz01.md) | P1 Phase1，复用已学内容 |
| 3 | [L5 Attention](lectures/lec05.md) · [L6 FlashAttention](lectures/lec06.md) | [Lab2](labs/lab02/README.md) | Lab2 Q7/Q8 代价练习 | P1 Phase2 |
| 4 | [L7 KV cache](lectures/lec07.md) · [L8 Quantization](lectures/lec08.md) | [Lab3](labs/lab03/README.md) | [HW1](hw/hw01/README.md) · [HW2](hw/hw02/README.md) · [Quiz2](quizzes/quiz02.md)，可延长节奏 | P1 报告 |
| 5 | [L9 共享块池](lectures/lec09.md) · [L10 前缀缓存](lectures/lec10.md) | [Lab4 有限块池](labs/lab04/README.md) | [HW5 前缀与容量](hw/hw05/README.md) | [Project2 Phase1](projects/project2/README.md)：准入与释放 |
| 6 | [L11 Continuous batching](lectures/lec11.md) · [L12 Chunked prefill](lectures/lec12.md) | [Lab5 调度器](labs/lab05/README.md) | [HW6 延迟与 goodput](hw/hw06/README.md) | [P2 Phase2](projects/project2_scheduler/README.md)：轨迹与实验 |
| 7 | L13 Sampling · L14 Speculative decoding | Lab 6: Spec decode（待发布） | HW 7: Sampling（待发布） | P2 checkpoint 2 |
| 8 | L15 Engine architecture · L16 Async/CUDA Graph | Lab 7: Engine trace（待发布） | HW 8: Engine（待发布） | **P2 due** |
| 9 | L17 MoE · L18 EP/EPLB | Lab 8: Top-k gate（待发布） | HW 9: MoE（待发布） | Project 3: EPLB Simulator 发布 |
| 10 | L19 Parallelism · L20 Communication | Lab 9: All-reduce（待发布） | HW 10: Communication（待发布） | P3 checkpoint 1 |
| 11 | L21 Profiling · L22 Benchmarking | Lab 10: Profile（待发布） | HW 11: Benchmark（待发布） | P3 checkpoint 2 |
| 12 | L23 Paper: DeepSeek-V3 · L24 Paper: vLLM | Lab 11: Paper reproduction / PR（待发布） | HW 12: Paper notes（待发布） | **P3 due** |
| 13 | L25 Review · L26 Advanced topics | Lab 12: Final project setup（待发布） | — | Final project proposal |
| 14 | L27 Final workshop · L28 Mock interview | — | — | Final project |
| 15 | L29 Presentations · L30 Wrap-up | — | — | **Final due** |

> **发布状态（2026-10-24）**：Python 起步、预备、Unit1，以及 Unit2 第 5–6 周已发布。第 6 周新增 L11/L12、Lab5、HW6、Project2 Phase2 与 18 组模拟实验；第 7–15 周及 Project2 后续阶段待发布。15 周是蓝图；零基础起步另计，第 4 周内容建议拆成多周。每批验收与 Unit1 的 100 分分开。

---

## 评分（自评，100 分）

| 项目 | 分数 | 依据 |
| :--- | ---: | :--- |
| Lab0–3 | 40 | 每个 10 分：代码、解释、调试、迁移 |
| HW1–2 | 20 | 每个 10 分：代码、推导、反例、checkoff |
| Project1 | 30 | 项目内部 100 分 rubric 乘 0.3 |
| Quiz1–2 | 10 | 两份原始成绩合计乘 0.5 |
| 总分 | 100 | 只计算已发布的 Unit1 |

详细规则见 [掌握度与评分](ASSESSMENT.md)。Python 起步与调试预备不计分；Unit2 第 5、6 周分别验收，不进入 Unit1 分母。自动测试与浏览器勾选都不能代替书面推导和口头解释。

---

## 目录

```
course/
├── README.md                 # 本文件：课程首页
├── syllabus.md               # 课程政策、AI 使用、checkoff
├── calendar.md               # 15 周详细日历
├── site/                     # 本地 HTML 课程网站（含渲染讲义 notes/、视频页）
├── foundations/              # 零基础 Python：四节正文 + Q0–Q4
├── TEXTBOOK.md               # 先修、练习对应和术语表
├── ASSESSMENT.md             # 只对已发布 Unit1 评分
├── run.ps1                   # Windows 自动寻找可用 Python
├── prep/                     # 预备 0–3：运行、调试、列表/矩阵、测试
├── lectures/
│   ├── lec01.md              # Python 执行与 GPU 编号模型
│   ├── lec02.md              # Coalescing & occupancy
│   ├── lec03.md              # Reduction & softmax
│   ├── lec04.md              # Tiling & GEMM
│   ├── lec05.md              # Attention from scratch
│   ├── lec06.md              # FlashAttention & decode
│   ├── lec07.md              # KV cache basics
│   └── lec08.md              # FP8/FP4 quantization
├── quizzes/
│   ├── quiz01.md             # Week 1–2 闭卷小测
│   └── quiz02.md             # Week 3–4 闭卷小测
├── labs/
│   ├── lab00/                # SIMT 模拟器 + vector add
│   ├── lab01/                # softmax + RMSNorm + traffic
│   ├── lab02/                # attention from scratch
│   └── lab03/                # continuous + paged KV cache
├── hw/
│   ├── hw01/                 # roofline + KV cache + quant
│   └── hw02/                 # attention & KV cache math
├── projects/
│   └── project1/             # Kernel Zoo
└── tools/
    ├── grade.py              # 本地 autograder
    └── build_notes.py        # 把 Markdown 渲染成 site/notes/*.html
```

---

## 环境

### 核心轨道（必需）

- Python 3.10+（仓库推荐 3.12）
- 标准库即可：`unittest`、`math`
- 不需要 GPU、不需要 PyTorch、不需要安装任何包

```bash
cd /path/to/DeepInfer
python course/tools/grade.py --list
python course/tools/grade.py lab00
```

> 参考答案在 `*_solution.py`；`python course/tools/grade.py --solution` 会跑参考答案。**先自己做，再对答案。**

### GPU 轨道（可选）

- WSL2 Ubuntu 或 Linux
- Python 3.12
- PyTorch（CUDA 12.8+ wheel）
- TileLang 0.1.15
- 一张 NVIDIA GPU（RTX 5060 8GB 足够做小 kernel）

```bash
# 在 WSL 里
python -m venv ~/venvs/deepinfer
source ~/venvs/deepinfer/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install tilelang==0.1.15 numpy
```

> 不要用仓库 `requirements-nvidia.txt` 里的 `torch==2.13.0+cu132`，除非你的驱动支持 CUDA 13。

---

## 课程哲学

1. **规格与测试相互核对**：先读输入/输出契约，再用例子核对测试。测试可能有遗漏，通过不等于覆盖所有情况。
2. **先正确，再快**：每个 kernel 先写 naive 版本，测正确性，再算 roofline。
3. **每次只优化一个瓶颈**：coalescing → shared memory → tiling → pipeline → occupancy。
4. **checkoff 比分数重要**：能讲清楚，才算真的会。
5. **项目要有“可展示性”**：README、测试、benchmark 表、报告，缺一不可。

---

## 和 DeepInfer 仓库的关系

这门课不要求你跑完整 DeepInfer（你的 8GB 显卡也跑不动 V4.1-Flash）。课程的做法是：

- 用纯 Python/SIMT 模拟器复现**机制**；
- 用小的 shape 复现**性能模型**；
- 用仓库里的真实代码做**对照阅读**（每节课有“仓库对照”小节）；
- 等你有了 GPU/云环境，再把 Project 1 的 kernel 移植到 TileLang。

仓库里的对应文件：

- `deepinfer/model_executor/layers/tilelang_ops/cuda/norm.py` → Lab 1 / Project 1
- `deepinfer/model_executor/layers/tilelang_ops/cuda/gather.py` → Lab 0
- `deepinfer/worker/cache_engine.py` → HW 1 / Unit 2
- `csrc/engine/scheduler.hpp` → Unit 2
- `examples/benchmark.py` → Unit 3

---

## 常见问题

**Q：C++ 基础少，Python 能看懂但调试困难，能学吗？**  
A：完全零基础从 Python 起步开始；会简单函数但调试困难从预备单元开始。第一讲和 Lab0 核心不要求 C++；先用 Python 练定位错误，再进入必要 C++ 和 GPU 桥接。后续材料中的 CUDA、profiling 先选读。

**Q：每天只有 1–2 小时，15 周够吗？**  
A：标准节奏 12–15h/周。如果只有 8h/周，用 30 周：每个 Week 拆成两周，Lab 和 HW 不变。

**Q：为什么不直接读 DeepInfer 源码？**  
A：源码是“答案”，课程是“题目”。先做题，再读答案，效率高得多。每节课的“仓库对照”会告诉你去读哪几段。

**Q：AI 能用吗？**  
A：能当 tutor，不能当代写。详见 [`syllabus.md`](syllabus.md)。

**Q：学完能做什么？**  
A：你能独立写出 vector add/reduction/softmax/RMSNorm/GEMM 的参考实现，能算 roofline，能解释 KV cache 和 attention 的 FLOPs；下一步（Unit 2）你会写一个 mini inference engine。

---

## 下一步

- 想先看全貌：读 [`../doc/study_plan_zh.md`](../doc/study_plan_zh.md) 的“一页纸版本”。
- 想直接开始：打开 [Python 起步](foundations/README.md)。
- 想要更硬的 GPU 内容：等 Unit 2，或现在就看 [`lectures/lec04.md`](lectures/lec04.md) 的 GEMM 部分。
