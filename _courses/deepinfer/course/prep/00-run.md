# 预备 0：运行程序，读懂第一条反馈

> **目标**：运行一个正确示例，追踪三轮循环，知道报错从哪里开始看。  
> **预计投入**：20–40 分钟，含动手；不是只读一遍。

## 1. 确认自己在哪里

用编辑器打开课程文件夹，再打开终端。终端所在目录应当能直接看到 `course`；离线ZIP解压后进入这一层。若打开的是个人主页源码仓库，先 `cd _courses/deepinfer`。Windows PowerShell可运行 `Get-Location` 和 `Get-ChildItem` 查看当前位置与文件；WSL/Linux可运行 `pwd` 和 `ls`。这是下文“课程根目录”的唯一含义。

检查 Python：

```bash
python --version
```

需要 3.10 或更新版本。如果命令不存在而你在 WSL，用 `python3 --version`；后面的命令也统一使用 `python3`。不要因为命令名不同就重装整个仓库的依赖。

## 2. 先预测，再运行

打开 `demo.py`，先看这段已完成的算法：

```python
total = 0
for value in [2, -1, 4]:
    total += value
```

在纸上填写每轮结束后的 `total`。随后从课程根目录运行：

```bash
python course/prep/demo.py
```

期望输出：

```text
index value before after
0 2 0 2
1 -1 2 1
2 4 1 5
result = 5
```

`before` 是本轮加法前的累计值，`after` 是加法后的值。调试不是打印越多越好，而是打印能验证自己判断的变量。

## 3. 看 traceback 的顺序

接下来运行一个尚未修复的题目：

```bash
python course/tools/grade.py prep --question q1 --failfast
```

这次会失败。关键内容类似：

```text
FAIL: test_empty (...TestMaximum.test_empty)
...
AssertionError: ValueError not raised
```

具体测试的先后以输出为准。按这个顺序读：

1. **最下面的异常类型和说明**：这里表示规格要求抛异常，但函数没有抛。
2. **失败测试名**：`test_empty` 提示你先看空列表。
3. **对应测试中的输入和期望**：打开 `test_prep.py`，只看这个方法。
4. **自己的函数**：去 `prep.py` 找 `maximum`，不要修改测试来变绿。

| 反馈 | 含义 | 第一件事 |
| :--- | :--- | :--- |
| SyntaxError | 语法还没能被 Python 解析 | 看提示行及前一行的括号、冒号、缩进 |
| IndexError | 某次索引超出范围 | 对照列表长度和具体索引 |
| AssertionError | 实际结果不符合测试要求 | 把输入、实际、期望写到一起 |
| NotImplementedError | 课程有意留出的待实现处 | 看该题说明，不需要重装环境 |

失败测试是定位信息，不是学习能力评分。

## 4. 第一个完成检查

```bash
python course/tools/grade.py prep --question q0
```

Q0 只检查已经完成的 demo，应该通过。这里暂时不运行整个预备单元，因为后面的题仍然有意保留错误。

## 5. 自己解释

这节对应prep q0，只验证现成运行示例；q1的故意错误用于认识反馈，修复安排在下一节。保留一张三轮状态表以及一次能定位到函数的失败消息，作为进入调试的证据。

不看输出，解释输入 `[2, -1, 4]` 的第二轮为什么 `before=2, after=1`。再预测空列表时结果与 trace 是什么。

<details>
<summary>检查预测后再看</summary>

第二轮把 -1 加到之前的 2 上。空列表不进入循环，返回累计值 0 和空记录列表；这是这个求和示例的约定。最大值的空输入约定不同，下一课会单独规定。

</details>

下一步：[预备 1：定位逻辑错误](01-debug.md)。
