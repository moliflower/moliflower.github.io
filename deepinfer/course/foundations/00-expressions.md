# 起步 0：让一个函数把结果交回来

**目标：**能追踪赋值，调用一个函数，并通过 Q1。先会整数加法和乘法即可。

## 1. 值、表达式、名字

`3` 是一个整数值。`3 * 2` 是表达式，Python 求值得到 `6`。`count = 3` 把名字 `count` 绑定到值 `3`。随后写 `count * 2`，Python 先查到 `count` 的值，再做乘法。

```python
count = 3
other = count
count = count + 1
print(count, other)
```

先预测最后一行，再放进自己的临时 `.py` 文件运行。第三行先计算等号右侧，再更新左侧；`other` 保存的是先前的整数值，不会跟着 `count` 自动变化。

<details>
<summary>预测核对</summary>

输出 `4 3`。运行第三行前，两个名字都表示 3；运行后 `count` 表示 4，`other` 仍表示 3。整数的这个例子不能直接推广为“所有赋值都会复制数据”，列表会在起步 2 讨论。
</details>

## 2. 函数是可复用的计算步骤

```python
def add_example(left, right):
    total = left + right
    return total

answer = add_example(2, 5)
print(answer)
```

`def` 定义一个函数。圆括号里的 `left`、`right` 是参数名字。冒号后面缩进四个空格的行属于函数体。定义时只记住这些步骤，调用 `add_example(2, 5)` 时才执行。

| 时刻 | 本次调用里的状态 | 含义 |
| :--- | :--- | :--- |
| 刚进入函数 | `left=2, right=5` | 实参绑定到参数 |
| 加法后 | `total=7` | 得到局部结果 |
| `return total` | 返回 `7` | 当前调用结束 |
| 回到外面 | `answer=7` | 调用表达式的值交给调用者 |

`return` 让调用者得到一个值；`print` 把内容显示到终端。只写 `print(total)` 而没有 `return`，调用者拿到的会是 `None`，即“没有返回具体结果”。这是初学者最常见的测试失败之一。

## 3. 从例子迁移到 Q1

假设每个元素占 2 字节，5 个元素就是 `2+2+2+2+2=10` 字节。一般写成“元素个数 × 每元素字节数”。这里统计数据本身的存储量，不是 Python 列表对象的实际内存占用。

打开 `foundations.py`。三引号包住的是说明文字；`#` 后面是注释，都用于帮助人阅读。`raise NotImplementedError(...)` 是有意留下的占位；在 Q1 中把这个占位换成自己的 `return`，保留已经写好的参数检查。

```bash
python course/tools/grade.py foundations --question q1 --failfast
```

动手前先写三个期望：`(5, 2)`、`(0, 4)`、`(3, 8)`。解释最后结果的单位为什么是字节。再运行测试；测试的是返回值，终端看见 `10` 并不等于函数返回了 `10`。

## 4. 理解检查

有人写了 `return count + element_bytes`，输入 `(2, 2)` 恰好通过。给一个能区分加法和乘法的输入，并写出两种结果。然后进入 [起步 1](01-control.md)。
