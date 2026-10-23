# Start Here

这个仓库现在有三份为你准备的学习材料：

## 1. 课程：DeepInfer 101

👉 **[`course/README.md`](course/README.md)**

CS61A 风格的 15 周课程：

- Lecture + Lab + Homework + Project + Checkoff
- 当前入口：Python 起步 → 调试预备 → 第一讲核心 → Lab0；Unit1 与 Unit2 第 5–6 周已发布
- 不需要 GPU：核心轨道是纯 Python
- 本地 autograder：`python course/tools/grade.py`
- 本地课程网站：[`course/site/index.html`](course/site/index.html)

**5 分钟开始：**

```bash
# 在 DeepInfer 根目录；如果 WSL 只有 python3，统一替换命令名
python course/foundations/demo.py
python course/tools/grade.py foundations --question q0
```

Windows 若 `python` 命令不可用，运行 `.\course\run.ps1 --demo` 和 `.\course\run.ps1 foundations --question q0`。启动器自动寻找可用 Python，不安装依赖。

然后打开 [Python 起步](course/site/notes/foundations.html)，按起步 0–3 学习。教材正文和作业对应见 [教材导航](course/site/notes/textbook.html)；课程评分只计算已发布的 Unit1，见 [验收标准](course/site/notes/assessment.html)。

## 2. 长期路线图

👉 **[`doc/study_plan_zh.md`](doc/study_plan_zh.md)**

46 周的学习路线：地基 → 算子 → 引擎 → 项目/面试。  
课程是这份路线图的“执行版”；路线图解释“为什么学这些”。

## 3. 本地课程网站

👉 **[`course/site/index.html`](course/site/index.html)**

CS61A 风格的本地网站：周历、课程大纲、教材、资源、视频、Infra 秋招。  
讲义/实验/作业已经渲染成排版好的网页；视频页按主题精选了 B 站链接和“看哪几段”。  
双击打开，或运行 `course/site/open.bat`；不需要服务器，不需要联网。

## 建议顺序

1. 先做 [Python 起步](course/site/notes/foundations.html) 的 Q0–Q4，再做 [`预备单元`](course/site/notes/prep.html)：能从失败测试定位错误再进入下一步。
2. 读 [`第一讲核心`](course/site/notes/lec01.html)：用 Python、编号表和地址推导，不要求 CUDA。
3. 做 [`Lab0`](course/site/notes/lab00.html)：Q0 示例 → Q1/Q2 索引 → Q3 组合 → Q4 模型 → Q5 独立算子 → Q6 修错。
4. 每题运行 `python course/tools/grade.py lab00 --question q1 --failfast`；卡住时用 `--hint 1`。
5. 记录一次失败的输入、期望、实际和修复依据；讲不清的点标记“需要复习”。

---

> 说明：完整 DeepInfer 需要 Linux + GCC 13 + CUDA 13 + SM100，当前笔记本（RTX 5060 8GB / SM120）跑不动完整模型。课程用“纯 Python 模拟器 + 小 shape + 仓库对照阅读”来学机制；等有 GPU/云环境，再把 Project 1 的 kernel 移植到 TileLang。
