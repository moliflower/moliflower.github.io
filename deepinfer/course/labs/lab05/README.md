# Lab5：Decode 优先与分块预填充

先修 [L11](../../lectures/lec11.md)、[L12](../../lectures/lec12.md)。约 2 小时；只写两个纯函数，不执行模型、不计时、不修改请求。主文件 [lab05.py](lab05.py)，公开测试 [test_lab05.py](test_lab05.py)。

## 1. 共享输入契约

active 必须为 list，每项是 tuple 或 list，恰含 `(id,prefill_remaining,output_remaining)`。ID 为非空字符串且不重复；prefill_remaining 为非负整数；output_remaining 为正整数。整数不接受 bool、float。token_budget、chunk_size 为正整数。非法输入统一 ValueError；即使预算已耗尽，也要检查整个列表。空 active 合法，但仍要检查参数。

请求顺序就是同阶段 FIFO 顺序。任何函数都不能修改输入；返回独立的新列表。output_remaining 是还没产生的输出数，最后一个 prefill chunk 会产生首 token；已经完成的请求不应出现在 active。

## 2. Q1：先找 decode 请求

`decode_first(active,token_budget)` 返回所有 prefill_remaining==0 的 ID，按输入顺序截取至预算。每条 decode 在一轮只花 1 个 token 预算。

```python
decode_first([("P", 5, 3), ("D", 0, 2), ("E", 0, 1)], 1)
# 预测结果：["D"]
```

实现后，用预算 2、空列表、全 prefill 列表再检查。不要因为队首 P 还没准备好就阻塞 D。

## 3. Q2：把剩余预算给 prefill

`plan_step(active,token_budget,chunk_size)` 返回 `(id,phase,count)` 的 tuple 列表。先输出 Q1 选中的 decode，phase="decode"、count=1；剩余预算按原顺序给 prefill 请求，每条 `min(prefill_remaining,chunk_size,剩余预算)`。phase="prefill"；count>0 才加入计划。

每请求每轮最多出现一次；不回头填满第二个 chunk；不在本轮把 prefill 请求变成 decode。这个函数只决策，不减任何计数。

输入 `[("A",5,3),("B",0,2),("C",2,1)]`、预算 4、chunk=2，预期：

```text
[("B","decode",1),("A","prefill",2),("C","prefill",1)]
```

只有 `[("A",9,3)]`、预算 8、chunk=2 时也只分配 2；这是明确的每请求 chunk 上限，不要“优化”为 8。

## 4. 测试与调试

从解压后含 course 文件夹的课程根目录执行。个人主页仓库内则先进入 `_courses/deepinfer`。网站上的完成勾选不会执行 Python。

```bash
python course/tools/grade.py lab05 --question q1 --unlock
python course/tools/grade.py lab05 --question q1 --failfast
python course/tools/grade.py lab05 --question q2 --hint 1
python course/tools/grade.py lab05
```

常见错误：先处理 prefill；预算不足一个 chunk 就不处理尾部；提前停止验证；修改 active；同请求被选两次。每个错误都用不超过三条请求的反例定位。

<details>
<summary>做完预测后再看：为什么决策不应直接修改 active？</summary>

引擎还需要用原来的轮开始状态解释计划，并统一在轮末提交变化。若策略一边遍历一边修改，就可能把刚完成 prefill 的请求再次当成 decode，或者污染历史快照。纯函数让策略测试不依赖执行器。
</details>

## 5. 交付与自评

本批 Lab5 20 分：Q1 5、Q2 7；手算与预算解释 4；一次调试和新反例 4。保留代码、三行计划表、一次失败原因。然后把自己的 plan_step 迁移到 [Project2 Phase2](../../projects/project2_scheduler/README.md)，不要直接拷参考答案。
