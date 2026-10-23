# HW5：前缀命中、共享容量与淘汰

先修：[第 10 讲](../../lectures/lec10.md)、Lab4、HW2。保留原 15 周蓝图中的 HW5 编号，HW3/4 没有被隐含发布。预计 2–3 小时。

修改 `course/hw/hw05/hw05.py`，只用标准库。所有数量、token ID 和时间戳都是非负整数且排除 bool；块大小必须为正整数。输入不变；返回新列表；规定的非法输入统一抛 ValueError。

## Q1：完整前缀匹配

`prefix_hit_tokens(cached, query, block_size)`：cached/query 为 token ID 列表。比较从位置 0 起的最长公共前缀，只返回其中完整块的 token 数。空列表合法。即使另一边为空，也必须验证全部 token 和 block_size。

预测：`[1,2,3,4,5]` 对 `[1,2,3,9]`，B=2，返回 2。为什么不是 3，也不是 4？给一个“后面相同但前面不同”的反例。

## Q2：只保留一份共享前缀

`shared_prefix_slots(lengths,prefix_tokens,block_size)`：lengths 为每条请求长度列表，prefix_tokens 必须是块大小的倍数，并且不超过任一长度。前缀只存一次，每条私有后缀独立取整。没有请求时返回 0，但仍验证公共参数。

手算 `[5,8,9]`、P=4、B=4：不共享 28 槽，共享 20 槽。交付一张每条请求的尾部长度、页数、浪费槽位表。再写出一条全部长度等于 P 的例子和一条 P=0 的例子。

这里不计额外的常驻前缀副本，也不把长度总和一次性取整。用整数运算，测试会包含超过 float 精确整数范围的长度。

## Q3：可回收块的顺序

`eviction_order(refcounts,last_used)`：两个等长列表，位置就是物理 ID。返回 refs=0 的 ID，按 `(last_used[ID], ID)` 升序排列。空列表返回 []。不修改任何引用数；真正回收由调用者完成。

手算 `[1,0,0,0]` / `[0,9,3,3]` 应返回 `[2,3,1]`。说明为什么块 0 不能排在第一位。

## 运行与交付

```bash
python course/tools/grade.py hw05 --question q1 --unlock
python course/tools/grade.py hw05 --question q1 --failfast
python course/tools/grade.py hw05 --question q1 --hint 1
python course/tools/grade.py hw05
```

将实现、三个手算、一个失败后修复的记录保存为自己的作业。每题 10 分：代码 5、推导 3、反例 2，共 30 分，只用于本次 Unit2 批次验收。出口问题：相同的请求命中率为什么可能有不同的 token 命中率？为什么节省槽位不能直接称为推理加速？
