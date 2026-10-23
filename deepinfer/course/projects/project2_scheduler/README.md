# Project2 Phase2：逐迭代调度与时间轨迹

第 6 周 Checkpoint 1，约 3–4 小时。先修 [Phase1](../project2/README.md)、[Lab5](../../labs/lab05/README.md)、[HW6](../../hw/hw06/README.md)。本阶段单独放在 project2_scheduler，保留 Phase1 文件、评分和学习记录，不覆盖已写的代码。

这是一台**固定 tick 的调度模拟器**：没有模型、GPU、采样、真实 KV 或内存抢占。max_active 是请求数上限，不是显存约束；第 5 周的页池没有接入本阶段。目标是把时间顺序、首 token 和请求生命周期做正确，而不是宣称实现完整推理引擎。

## 1. Q1：迁移调度策略

在 [project2_scheduler.py](project2_scheduler.py) 的 `plan_step(active,token_budget,chunk_size)` 迁移自己的 Lab5 Q2。契约完全一致，包括验证所有记录、decode 优先、每请求一轮一个 chunk、无输入修改。为使每份练习可以独立下载和评分，这里保留明确的迁移接口，而不依赖其他目录里的未完成 starter。

## 2. Q2：模拟到所有请求结束

`simulate(requests,max_active,token_budget,chunk_size)` 返回 `{"trace":快照列表,"requests":按ID索引的结果}`。

requests 必须是 list，每项 tuple/list 恰含 `(id,arrival,prompt_tokens,output_tokens)`。ID 为非空且全局唯一的字符串；arrival 为非负整数；prompt_tokens/output_tokens/max_active/token_budget/chunk_size 为正整数；所有整数排除 bool。额外要求 token_budget >= max_active，确保每轮能覆盖所有活动 decode。提前验证全部输入，非法 ValueError；空输入返回空 trace 和空 requests，但仍验证公共参数。禁止修改输入。

到达时间可以乱序。内部按 `(arrival,原输入位置)` 排序，同时间稳定；等待队列严格 FIFO。无活动与等待请求时直接跳到下次到达，不输出空闲轮快照。

## 3. 一轮的固定顺序

在 t 加入所有 arrival<=t 的新请求，补满 max_active，取轮开始状态调用一次 Q1。执行 plan 后在 t+1 提交：

- prefill 减本次处理数；如果刚变成 0，则产生第一个输出，output_remaining 减 1。
- decode 每条产生一个输出，output_remaining 减 1。
- 所有输出时间都记录 t+1；每请求本轮最多产生一个输出。
- output_remaining==0 的请求结束并移出；等待者直到下一轮才能使用该位置。

不要把 prompt token 数算成输出 token 数；总处理工作是 `sum(prompt_tokens + output_tokens - 1)`。最后一个输出生成后不必再模拟一次没有用户输出的 decode。

每个快照的字段精确定义：

| 字段 | 类型与时点 |
| :--- | :--- |
| start / end | 本轮开始 t、结束 t+1 |
| admitted | 本轮准入 ID，FIFO 顺序 |
| plan | Q1 返回的 tuple 列表，decode 在前 |
| emitted / finished | 本轮产出/结束的 ID，保持 plan 顺序 |
| active | 轮末 ID → (剩余prompt,剩余输出) 的字典，仅未完成者 |
| waiting | 轮末已经到达但未准入的 ID，FIFO 顺序，不含未来请求 |

results[ID] 恰含 `{"arrival":原到达时间,"token_times":[各输出完成时间]}`。历史快照、结果列表、输入之间不得共享可变容器；后续步骤或修改一张快照不能污染别的快照。

## 4. 完整手算基准

输入 A=(0,3,2)，B=(1,1,1)，C=(1,1,2)，分别表示 arrival/prompt/output；max_active=2、budget=3、chunk=2。

| 轮 | 准入 | 计划 | 输出 | 结束 | 轮末 waiting |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0→1 | A | A prefill 2 | 无 | 无 | 空 |
| 1→2 | B | A prefill 1，B prefill 1 | A,B | B | C |
| 2→3 | C | A decode 1，C prefill 1 | A,C | A | 空 |
| 3→4 | 无 | C decode 1 | C | C | 空 |

A 的输出时间 [2,3]，B=[2]，C=[3,4]；TTFT 分别为 2、1、2。B 的 TPOT 无定义，A/C 为 1 tick。公开测试保存了完整的精确快照，先逐行复现，再考虑随机请求。

## 5. 测试与 18 组实验

从课程根目录运行（个人主页仓库先进入 `_courses/deepinfer`）：

```bash
python course/tools/grade.py project2_scheduler --question q1 --failfast
python course/tools/grade.py project2_scheduler --question q2 --unlock
python course/tools/grade.py project2_scheduler
python course/projects/project2_scheduler/experiment.py
```

[experiment.py](experiment.py) 默认运行你的实现，向终端输出 CSV，不覆盖文件。三个固定负载 × 两个并发上限 × 三个 chunk = 18 行。若要保存，自行重定向到公共课程目录之外。`--solution` 只用于显式查看参考实验，不是自己的通过证据。

观察窗口从 0 到最后完成时间，含空闲间隔；SLO 为 TTFT<=4 ticks、TPOT<=1 tick。输出吞吐单位 token/tick，goodput 单位请求/tick。所有有定义的 TPOT 应为 1：预算覆盖全部活动 decode，加上固定轮时长，决定了这一点。因此不声称验证了真实 TTFT/TPOT 性能权衡或 GPU 加速。

## 6. 报告与验收

使用 [报告模板](report_template.md)。阶段自评 50 分：Q1 10；Q2 20；手算、不变式与反例 10；18 组实验、解释与局限 10。第 6 周 Lab/HW/Project 合计 100 分，独立于第 5 周和 Unit1。

至少构造四个反例：单输出请求；同时到达的 FIFO；极长空闲间隔；历史快照别名。报告应说明“还缺什么”而不只列通过测试：第 7 周的 sampling/投机解码、内存与模型执行集成、可校准的成本模型和在线基准都不在本阶段验收内。
