# Python 起步：从第一行代码到可测试的函数

**入口：从未写过程序也可以。** 只需要会编辑并保存文本、做整数四则运算。本单元不要求矩阵、GPU、C++ 或任何第三方包。建议先预留 6–12 小时，分成多次；时间是预算，完成条件是能自己解释并修改程序。

## 1. 第一次运行

用编辑器打开**课程根目录**，再打开终端。课程根目录的判断标准是能直接看到 `course` 文件夹：离线ZIP解压后进入这一层；若打开的是 `moliflower.github.io` 源码仓库，先运行 `cd _courses/deepinfer`。下文所有 `python course/...` 都从这里运行，不从 `course` 文件夹内部运行。终端用来输入命令，`.py` 文件用来保存Python代码；不要把终端命令写进 `.py` 文件。

```bash
python --version
python course/foundations/demo.py
python course/tools/grade.py foundations --question q0
```

需要 Python 3.10 或更新版本。若 Windows 的 `python` 打开商店、显示“系统无法访问此文件”或找不到命令，可在 PowerShell 从课程根目录运行本课程的启动器：

```powershell
.\course\run.ps1 --demo
.\course\run.ps1 foundations --question q0
```

启动器依次探测可运行的 Python，也会尝试当前用户 Codex 配套的 Python；不会安装包或修改系统设置。如果均不存在，按 [Python 官方 Windows 使用说明](https://docs.python.org/3/using/windows.html) 安装 Python 后重开终端。WSL/Linux 中若只有 `python3`，将全文命令名换成 `python3`。后文的 `python course/tools/grade.py 参数` 在 Windows 也可以统一换成 `.\course\run.ps1 参数`。

示例前 3 行应为 `count = 3`、`bytes = 6`、`returned = 7`，随后打印三轮累计过程。Q0 应通过；后续题初次运行出现 `NotImplementedError` 表示练习尚未填写。先看错误末尾的题号，不要修改测试。

## 2. 阅读与练习按同一顺序走

| 阅读 | 新概念 | 本次练习 | 独立完成的证据 |
| :--- | :--- | :--- | :--- |
| [起步 0：表达式与函数](00-expressions.md) | 值、名字、赋值、调用、返回值 | Q0 已完成示例；Q1 字节计数 | 能区分 `print` 与 `return` |
| [起步 1：条件与整数](01-control.md) | 比较、布尔值、缩进、整除、余数 | Q2 最少分组数 | 解释为什么整除时不加一 |
| [起步 2：列表与循环](02-sequences.md) | 索引、遍历、append、新列表 | Q3 长度截断 | 输出变化不会改坏输入 |
| [起步 3：组合与边界](03-composition.md) | 累计状态、循环不变式、测试 | Q4 连续存储边界 | 换一组长度仍能自己推导 |

## 3. 题目规格

只修改 [foundations.py](foundations.py) 的 TODO。函数签名与已提供的参数检查保留。所有输入均为普通 Python 整数或整数列表；字符串、小数和其他类型不在本单元输入域，不要求你写通用类型检查。负数或不允许的零按以下规定抛 `ValueError`，检查代码已提供。

| 题 | 函数 | 结果与边界 | 依赖 |
| :--- | :--- | :--- | :--- |
| Q0 | `add_example(left, right)` | 已完成；`(2, 5)` 返回 `7` | 无 |
| Q1 | `bytes_for_elements(count, element_bytes)` | `count>=0`、每元素字节数 `>0`；返回整数；0 个元素占 0 字节 | 起步 0 |
| Q2 | `ceil_groups(count, group_size)` | `count>=0`、组容量 `>0`；整数运算，0 个元素需要 0 组 | 起步 1 |
| Q3 | `clamp_lengths(lengths, limit)` | 非负长度与上限；返回新列表，每项为原值与上限较小者；空输入返回新 `[]` | 起步 2 |
| Q4 | `prefix_offsets(lengths)` | 非负长度；返回所有起点及最终终点；`[]` 返回 `[0]`，零长度合法 | 起步 3；无其他函数依赖 |

Q1/Q2 是局部填空，Q3 给出结果容器，Q4 独立组织函数体。每题测试可单独运行，不会因其他题还没写而失败。

```bash
python course/tools/grade.py foundations --list
python course/tools/grade.py foundations --question q1 --unlock
python course/tools/grade.py foundations --question q1 --failfast
python course/tools/grade.py foundations --question q1 --hint 1
```

`--unlock` 是先预测再编码的练习，不会锁住代码。需要帮助时依次看 `--hint 1`、`2`、`3`。公开测试在 [test_foundations.py](test_foundations.py)，参考解在 `foundations_solution.py`；先写自己的解释再对照。

## 4. 出口检查

先按“阅读一小节→纸上预测→修改对应函数→单题测试”循环推进；不要第一天把整个课程评分器的未实现题都当作安装失败。Q0是已完成示例，不是独立编码成绩；Q1–Q4逐步减少骨架。每次通过后，换一个输入自己解释，才能判断是否理解而非只记住测试答案。

```bash
python course/tools/grade.py foundations
```

测试通过之后，关闭答案，用不曾练过的输入回答：

1. `ceil_groups(13, 4)` 是多少？如果换成 12，哪一步改变？
2. `clamp_lengths([1, 7, 4], 4)` 返回什么？原列表应当是什么？
3. `prefix_offsets([1, 4, 0, 2])` 的每个数字代表什么？为什么结果比输入多一项？

再用自己的话说出 `return`、`if`、`for` 各自的职责，并记录一个你亲自修复的失败用例。讲不清的题回读对应章节，不清零已有进度。完成后进入 [调试预备单元](../prep/README.md)，再学第一讲和 Lab0。
