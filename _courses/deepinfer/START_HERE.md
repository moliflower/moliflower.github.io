# Start Here

这里是 DeepInfer 101 的自学材料。首次使用请先打开 [自学路线](course/site/notes/learning-path.html)，再按自己的基础进入相应章节。

## 1. 课程：DeepInfer 101

👉 **[`course/README.md`](course/README.md)**

CS61A 风格的 15 周课程：

- Lecture + Lab + Homework + Project + Checkoff
- 当前入口：Python 起步 → 调试预备 → 第一讲核心 → Lab0；Unit1 与 Unit2 第 5–6 周已发布
- 不需要 GPU：核心轨道是纯 Python
- 本地 autograder：`python course/tools/grade.py`
- 本地课程网站：[`index.html`](index.html)

**5 分钟开始：**

先确认当前目录能直接看到 course 文件夹。下载 ZIP 后进入解压根目录即可；如果拿到的是个人主页源码仓库，先执行 `cd _courses/deepinfer`。后面的 `python course/...` 都从这个“课程根目录”运行。

```bash
python course/foundations/demo.py
python course/tools/grade.py foundations --question q0
```

Windows 若 `python` 命令不可用，运行 `.\course\run.ps1 --demo` 和 `.\course\run.ps1 foundations --question q0`。启动器自动寻找可用 Python，不安装依赖。

系统只有 python3 时统一替换 python。然后打开 [Python 起步](course/site/notes/foundations.html)，按起步 0–3 学习。教材正文和作业对应见 [教材导航](course/site/notes/textbook.html)。Unit1 的 100 分、第 5 周和第 6 周各自的 100 分分别记录，见 [验收标准](course/site/notes/assessment.html)。

## 2. 长期路线图

👉 **[`doc/study_plan_zh.md`](doc/study_plan_zh.md)**

46 周的学习路线：地基 → 算子 → 引擎 → 项目/面试。  
课程是这份路线图的“执行版”；路线图解释“为什么学这些”。

## 3. 本地课程网站

👉 **[`index.html`](index.html)**

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

> 说明：本课的必修部分是纯 Python 教学模型，不要求运行完整上游模型。图解、容量模型和虚拟 tick 帮助理解机制，不代表已经测得 GPU 性能。GPU 扩展等具备设备与相应编程基础后再做。
