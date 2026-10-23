# 预备 3：让测试帮助你，而不是只看红绿

> **目标**：写出既不误报、也不漏报的检查；独立完成一个小函数。  
> **修改**：`prep.py` 中的 `assert_vector_close`、`vector_add`。  
> **预计投入**：45–75 分钟。

## 1. 一个测试回答三个问题

```python
actual = vector_add([2, -1], [3, 4])
expected = [5, 3]
assert actual == expected, f"实际 {actual}，期望 {expected}"
```

输入是什么、期望是什么、实际是什么。断言成立时继续执行，不成立时抛 `AssertionError`。课程用 `unittest` 把多个这样的检查组织在一起。

## 2. 浮点数为什么不能总用等号

先运行：

```python
print(0.1 + 0.2)
print((0.1 + 0.2) == 0.3)
```

Python 的浮点数通常使用二进制双精度；0.1 不能被有限位二进制精确表示。因此结果接近 0.3，但直接比较可能失败。这与代码索引错误是两类不同的问题。

本题只处理有限数值，规定：

```text
abs(actual[i] - expected[i]) <= atol
```

`atol` 是允许的绝对误差。我们用固定的小容差学习比较方法；真实 GPU 算子还会按数据类型、规模选择相对误差和绝对误差，不是把容差放大到测试能过。

## 3. 修复比较函数，别让它失去检查能力

`assert_vector_close` 已提供长度检查，但逐项用了精确相等。修改这一处，使它同时满足：

| actual | expected | 应当 |
| :--- | :--- | :--- |
| [0.1+0.2] | [0.3] | 通过 |
| [0.4] | [0.3] | 抛 AssertionError |
| [1,2] | [1] | 抛 AssertionError |
| [] | [] | 通过 |

不能直接删掉断言，或无条件返回：那会漏掉真正的错误。`zip` 会在最短列表处停止，因此比较前的长度检查也不能删。

<details>
<summary>局部提示</summary>

只在对应元素差的绝对值小于或等于 `atol` 时通过。保留位置 `i`、实际值和期望值，方便失败时定位。

</details>

## 4. 独立实现 vector_add

这次没有完整算法骨架。规格如下：

- 返回一个新列表，元素为对应的 `a[i]+b[i]`。
- 不修改两个输入。
- 两个列表长度不同，抛 `ValueError`，不能静默截短。
- 两个列表为空，返回 `[]`。

先在纸上写步骤，再写代码。不要求使用列表推导式，用普通循环完全可以。

<details>
<summary>提示 1：先处理什么？</summary>

先检查长度，再创建输出列表。这样不匹配输入会明确失败，而不是在循环中偶然越界。

</details>

<details>
<summary>提示 2：循环的职责</summary>

每一轮取出相同位置的一对元素，把它们的和追加到输出；输入本身不需要赋新值。

</details>

## 5. 一次只检查一题

```bash
python course/tools/grade.py prep --question q3 --failfast
```

输出包含失败测试名。若你只想运行其中一个测试，先在终端进入 `course/prep`，再运行：

```bash
python -m unittest -v test_prep.TestTesting.test_vector_mismatch
```

`test_prep` 是文件名去掉 `.py`，`TestTesting` 是测试组，最后一段是具体测试方法。这就是为什么命令需要在对应目录运行；统一的 `grade.py` 会替你选择目录。

## 6. 加一个你自己的测试

在自己的练习文件里写一个新的断言，例如含负数的输入，或修改返回列表后确认输入没变。先手算期望，再运行；不要用待测函数自己生成自己的答案。

全单元回归：

```bash
python course/tools/grade.py prep
```

将一次失败与修复写成五行记录。如果只剩浮点比较错误，就复习本课；如果不清楚为什么修改副本影响原矩阵，回到预备 2，不必全部重学。

下一步：[第 1 讲：从 Python 建立 GPU 执行模型](../lectures/lec01.md)。
