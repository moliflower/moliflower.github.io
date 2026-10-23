# Project2：Mini Engine · Phase1 请求生命周期

本页是 Phase1：有限容量下的 FIFO 准入、等待、结束和释放。约 3–4 小时。先修 Lab4、HW5；纯 Python，无模型权重或 GPU。[Phase2 逐迭代调度](../project2_scheduler/README.md) 已单独发布，保持本阶段代码和通过条件不变。Sampling、真实设备 benchmark 与模型执行集成仍待后续发布。

## 1. 把输入事件变成可观察的状态

修改 `course/projects/project2/project2.py`。本阶段按已知请求长度预留整条请求的页数，不存真实 KV；Lab4 的物理 ID 与共享引用暂不接入。这样先验证请求生命周期，下阶段再逐 token 增长。

始终维护：`sum(active.values()) + free_blocks == capacity`。active 是 ID→预留页数的字典，waiting 按提交顺序排列。请求不得同时出现在 active 和 waiting。

## 2. Q1：严格 FIFO 准入

`admit_fifo(waiting,free_blocks,block_size)` 返回 `(accepted_ids, remaining_free_blocks)`。

- waiting 为 `(非空字符串 ID, 正整数 token 长度)` 的列表，ID 不重复。
- free_blocks 是非负整数；block_size 是正整数；整数排除 bool。
- 先验证整个队列，包括当前无法接纳的尾部。非法值抛 ValueError。
- 每请求需要 `(length+B-1)//B` 页。从队首开始，遇到第一个放不下的就停止；不能跳过它接纳小请求。
- 返回独立的新 ID 列表，不修改输入。空队列合法。

手算：队列 A=5 tokens、B=1 token，块大小 4。空闲 1 块时一个也不接纳；空闲 2 块时只接纳 A；空闲 3 块时接纳 A、B。

## 3. Q2：重放事件

`replay(events,capacity,block_size)` 返回每个事件处理完后的独立快照列表。capacity/block_size 是正整数；events 是事件列表，每项为 tuple 或 list：

| 事件 | 行为 |
| :--- | :--- |
| `("submit",id,length)` | ID 为非空字符串且整个重放中不能重复；长度为正整数；单请求所需页数不能超过池容量，否则 ValueError；合法请求加入队尾 |
| `("finish",id)` | 只允许结束当前活动请求；释放全部预留页数；未知、等待中、已结束 ID 均 ValueError |
| 其他事件名或错误长度 | ValueError |

每个事件后都调用 Q1 重试 waiting，再记录：`{"active": {id: pages}, "waiting": [ids], "free_blocks": integer}`。快照中的字典、列表不能被后续事件修改。函数保留输入；失败抛异常，不返回部分结果，没有外部状态需要回滚。空事件列表返回 []。

## 4. 完整追踪

容量 3 块、块大小 4：

| 事件 | active | waiting | free |
| :--- | :--- | :--- | :--- |
| submit A,5 | A:2 | 空 | 1 |
| submit B,5 | A:2 | B | 1 |
| submit C,1 | A:2 | B,C | 1 |
| finish A | B:2,C:1 | 空 | 0 |
| finish B | C:1 | 空 | 2 |
| finish C | 空 | 空 | 3 |

第三行 C 明明放得下却等待，这是本阶段刻意定义的严格 FIFO 行为；它暴露队首阻塞的代价，不是要求你偷偷修改策略。下一阶段才比较不同调度规则。

## 5. 测试、报告与出口

```bash
python course/tools/grade.py project2 --question q1 --unlock
python course/tools/grade.py project2 --question q1 --failfast
python course/tools/grade.py project2 --question q2 --hint 1
python course/tools/grade.py project2
```

复制 [报告模板](report_template.md) 写一页报告。交付代码、上表的实际快照、一次调试记录，以及一个自己设计的队首阻塞案例。阶段自评 50 分：Q1 10、Q2 20、守恒/快照解释 10、报告与反例 10，不并入 Unit1 分母。

必须讲清：什么时候释放容量；为什么每条请求独立向上取整；为何历史快照要复制；为什么这里没有测到 TTFT/TPOT。完成后按 [L11](../../lectures/lec11.md) → [L12](../../lectures/lec12.md) → [Lab5](../../labs/lab05/README.md) → [Phase2](../project2_scheduler/README.md) 继续；Phase2 暂不接入页池，不能把并发数当显存容量。
