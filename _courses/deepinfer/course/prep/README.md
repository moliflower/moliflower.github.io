# 预备单元：把“看得懂”变成“能定位错误”

> **适合你**：能写简单 Python 函数，遇到错误不容易排查；C++ 基础少。  
> **需要**：Python 3.10+；不需要 C++、CUDA、GPU 或第三方包。  
> **节奏**：四次短练习，可分几天完成；不计入正式 15 周。

如果还没有写过 Python，先完成 [Python 起步](../foundations/README.md) 的四节课和 Q0–Q4。能解释 `return`、`if`、`for`，并独立构造新列表后再进入这里；本单元开始练调试，不再把这些语法当成新概念。

## 1. 本单元的产物

你将修复 `prep.py` 中三个位置的错误、一个测试辅助函数，并独立写一个向量加法。文件里的错误是故意留下的，不是安装失败。

完成后，你应能说清楚一次调试的五件事：输入、期望、实际、最小复现、修复依据。不会要求你记住 traceback 的每一行。

## 2. 按顺序学习

| 课 | 先学什么 | 动手任务 | 检查命令的题号 |
| :--- | :--- | :--- | :--- |
| [预备 0：运行和报错](00-run.md) | 路径、运行文件、追踪变量 | 运行已经完成的示例 | q0 |
| [预备 1：定位逻辑错误](01-debug.md) | 最小复现、循环不变式 | 修复最大值函数 | q1 |
| [预备 2：列表和矩阵](02-matrix.md) | 副本、二维索引、累计值 | 修复复制与矩阵乘法 | q2 |
| [预备 3：让测试帮助你](03-tests.md) | 断言、误差、边界、自测 | 修复比较函数，独立写向量加法 | q3 |

所有下面这样的命令都在**课程根目录**运行，即能直接看到 `course` 文件夹的位置。离线ZIP解压后进入这一层；若从个人主页源码仓库开始，先 `cd _courses/deepinfer`。不要已经进入 `course` 后又输入以 `course/` 开头的路径。若环境只提供 `python3`，将命令中的 `python` 替换为 `python3`；Windows启动器也可用 `.\course\run.ps1 prep --question q0`。

```bash
python course/prep/demo.py
python course/tools/grade.py prep --question q0
```

第一条最后显示 `result = 5`；第二条显示 `PASS`。先获得一个确定能运行的起点，再处理故意留下的错误。

## 3. 卡住时怎样得到帮助

```bash
python course/tools/grade.py prep --question q1 --failfast
python course/tools/grade.py prep --question q1 --hint 1
python course/tools/grade.py prep --question q1 --hint 2
python course/tools/grade.py prep --question q1 --hint 3
```

每次只显示所选的一层提示。先尝试自己的解释，再逐层打开；不要求一次独立完成。`--failfast` 让当前题在第一次失败时停止，避免被后面的错误淹没。

只编辑 `prep.py`。`demo.py` 是现成示例，`test_prep.py` 是公开规格，`prep_solution.py` 是对照用的完整解法。参考解法测试会在临时副本中运行，不覆盖你的文件。

## 4. 记录一次修复

在自己的笔记中写下下面五行即可；不强制使用 Git，不需要新建固定路径的文件。

```text
输入：
期望：
实际：
我定位到的原因：
修复后新增的测试：
```

## 5. 出口检查

四节阅读分别只对应q0、q1、q2、q3；q2包含复制矩阵与矩阵乘法两项，q3包含比较函数与独立向量加法两项。先修复当前小题再整份运行，避免把其他故意保留的错误误认成一次修复引起的回归。

```bash
python course/tools/grade.py prep
```

- 能修好全负数的最大值，而不是给某个测试特判。
- 能解释复制矩阵时为什么要复制每一行。
- 能手算矩阵的一格，并在代码中找到对应的累计过程。
- 能让测试既接受微小浮点误差，也拒绝明显错误。

有一项讲不清，就回看对应一课，状态记为“需要复习”，已完成的部分仍然有效。准备好后，进入 [第 1 讲](../lectures/lec01.md)，再做 [Lab0](../labs/lab00/README.md)。

## 6. C++ 在哪里开始？

这条入门路径先用 Python 建立算法和调试能力。之后的 C++ 桥接按“函数与编译 → vector → 引用 → 指针与连续内存”展开，再进入 GPU。桥接单元尚未发布；现在不用为了本单元去安装 CUDA 或补完整套 C++。
