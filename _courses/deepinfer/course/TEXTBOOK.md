# 教材：从零基础到可验证的推理算子

这份索引把课程正文、练习与验收串在一起。正文就在仓库内，可以离线读；外部课程、论文和博客用于完成本章之后的扩展。遇到陌生概念，先回到表格中的先修，避免在多个链接之间来回跳。

本课程借鉴 [CS61A Lab1](https://cs61a.org/fa26/labs/lab01/) 的预测与编程练习、[HW1](https://cs61a.org/fa26/hw/hw01/) 的函数规格、[Hog](https://cs61a.org/fa26/projects/hog/) 的分阶段项目，以及 [Composing Programs](https://composingprograms.com/3ed/designing-functions/) 的函数设计方法。题目和 Infra 路线为本地课程设计，未获得 Berkeley 认证；“对齐”指可检查的教学设计，不表示已达到一门成熟大学课程的完整性。

## 1. 两个入口

- **没有写过 Python**：从 [Python 起步](foundations/README.md) 开始，学四节并做 Q0–Q4。
- **会写简单函数，但调试吃力**：直接做 [调试预备](prep/README.md)。如果不能解释 `return` 与 `print` 或列表别名，先补起步课。

第一遍不要求学 CUDA、C++、Tensor Core 或装完整 DeepInfer。Python 核心题能先练算法、数据布局、数值稳定性与成本分析。

## 2. 每一章学完要留下什么

| 章节 | 必要先修 | 核心练习 | 离开本章的证据 |
| :--- | :--- | :--- | :--- |
| [Python 起步 0–3](foundations/README.md) | 整数四则运算 | foundations Q0–Q4 | 追踪变量；独立写列表循环；解释边界 |
| [调试预备 0–3](prep/README.md) | 函数、条件、循环、列表 | prep Q0–Q3 | traceback → 最小反例 → 修复 → 新测试 |
| [L1 执行与地址](lectures/lec01.md) | prep | Lab0 Q0–Q6 | 列出尾组索引，区分 word 与 byte |
| [L2 访存](lectures/lec02.md) | L1 | Lab0 访存计数；GPU 部分选读 | 区分模型 line 与硬件 sector，说明假设 |
| [L3 数值算子](lectures/lec03.md) | 一维循环、平方、平均值 | Lab1 | 极端 logits 下 softmax 稳定；解释 eps |
| [L4 矩阵乘与 tiling](lectures/lec04.md) | prep 矩阵乘、Lab0 单位 | Project1 Phase2（先完成 Phase1） | 手算一个非整除 tile，核对访问计数 |
| [L5 Attention](lectures/lec05.md) | L3、矩阵乘与转置 | Lab2 Q1–Q6 | 手算一个输出行，说明 mask 和归一化 |
| [L6 Attention 的流量](lectures/lec06.md) | L5、L4 成本模型 | Lab2 Q7–Q8；HW2 在 L7 后 | 分清数学有效工作、实现 FLOPs、容量和流量 |
| [L7 KV cache](lectures/lec07.md) | 列表副本、整数分组、L5 | Lab3；HW2 | append/get/free 不破坏所有权，解释尾页 |
| [L8 量化](lectures/lec08.md) | L3、误差与字节单位 | HW1；Quiz2 | 解释 scale、舍入误差与元数据成本 |
| [Kernel Zoo](projects/project1/README.md) | Lab0/1、L4 | Q1–Q9、实验与报告 | 正确性、流量计数、可复现实验、局限性 |
| [L9 共享块池](lectures/lec09.md) | L7、Lab3、HW2 | Lab4 | 失败无部分修改、引用数与所有者一致 |
| [L10 前缀缓存](lectures/lec10.md) | L9、Lab4 | HW5 | 区分公共前缀、完整块命中、共享容量与淘汰 |
| [Mini Engine Phase1](projects/project2/README.md) | Lab4、HW5 | Q1/Q2、轨迹与报告 | 容量守恒、FIFO 等待、释放重试与独立快照 |
| [L11 Continuous batching](lectures/lec11.md) | Phase1、L9/L10 | Lab5 Q1 | 迭代边界、并发数与 token 预算不能混淆 |
| [L12 Chunked prefill](lectures/lec12.md) | L11 | Lab5 Q2、HW6 | 首 token、请求等权 TPOT、SLO 与统计窗口 |
| [Mini Engine Phase2](projects/project2_scheduler/README.md) | Lab5、HW6 | Q1/Q2、18 组实验与报告 | 精确时间轨迹、独立快照、模型局限 |

“会讲一个结论”还不够。每章至少写一个正常输入、一个边界输入，并说明测试期望从哪里来。可复用 [作业记录模板](WORKSHEET.md)。

## 3. 读公式前先认清量

| 术语 | 本课程含义 | 容易混淆的地方 |
| :--- | :--- | :--- |
| shape | 各维度长度，如矩阵 `(rows, cols)` | 长度为 3 的列表合法索引是 0、1、2 |
| element / word / byte | 元素是一个数；Lab0 的 word 是一个 4B 槽；byte 是字节 | “地址 + 1”究竟前进一个字节还是一个槽 |
| 容量 | 同时保留数据所需的空间 | 读同一数据三次，容量不变，流量增加 |
| 请求流量 | 按给定算法统计的读写字节数 | 缓存、复用与合并会影响设备实际传输 |
| FLOPs | 浮点运算次数；本课程一次乘加计 2 | FLOP/s 是每秒速度，不是总次数 |
| GB / GiB | GB=10^9 B，GiB=2^30 B | 带宽常用十进制，容量题明确标单位 |
| 不变式 | 每轮循环前后维持的关系 | 变量值可以变化，关系不能被破坏 |
| 误差容限 | 允许实际结果偏离期望的范围 | 先检查 shape、长度与有限性，再比较误差 |
| Roofline | 给定流量和峰值模型的性能上界/时间下界 | Python wall time 不能直接当 GPU kernel 时延 |
| checkpoint | 某阶段可检查的产物 | 打开过页面或勾选完成，不等于代码已验收 |

## 4. 从课程作品走向 28 届 Infra 准备

先把“读过 DeepInfer”变成具体证据：能重现一个失败、解释一个数据布局、从假设推导成本、复现实验并说出结论的适用范围。Kernel Zoo 报告是第一份这样的作品。

本仓库的 15 周是课程蓝图，零基础起步与补课时间另计。当前可运行材料覆盖 Unit1 和 Unit2 第 5–6 周；采样、通信与负载均衡、后续 C++/GPU 桥接仍待发布。逐迭代调度目前仅为固定 tick 模型，不含真实设备或 KV 集成。未发布部分没有被现有测试验证，也不计入成绩。长期准备还需要数据结构、Linux/C++、体系结构、并发和网络，见 [长期路线](../doc/study_plan_zh.md)。

每完成一个阶段保留四样材料：自己的实现、能说明问题的测试、可重跑的实验命令、含局限性的解释。招聘时间按目标公司的实际公告确认；这里不把“28 届”推断成固定报名日期。
