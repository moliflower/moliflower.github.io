# 第 11 讲：Continuous batching，把调度单位缩到一次迭代

先修：[L9](lec09.md)、[L10](lec10.md)、[Project2 Phase1](../projects/project2/README.md)。约 45–60 分钟，配套 [Lab5 Q1](../labs/lab05/README.md)。不需要 GPU。

## 1. 一条请求不是一次模型调用

一条自回归生成请求包括 prompt 的 prefill 和后续 decode。处理完整 prompt 后，最后位置的 logits 可用于采样第一个输出 token；之后每次普通 decode 再生成一个 token。本周不模拟 logits 或采样，只记录“何时产生一个输出”。因此输出长度为 1 的请求在 prefill 完成时就结束，不额外虚构一次 decode。

把两个长度不同的请求绑成不可变化的一批会怎样？假设 A 先完成，B 还需要三次迭代，而 C 已经到达。如果只能等整批结束再加入 C，A 腾出的并发位置就不能及时使用。这里讨论的是不能补入新请求的固定批次；不要误解为所有 static batching 实现都禁止流式返回 A。

[Orca（OSDI 2022）](https://www.usenix.org/conference/osdi22/presentation/yu) 提出了 iteration-level scheduling：执行一轮后重新选择批次，而不是必须执行到整条请求结束。Continuous batching 的关键是批次成员可以在迭代边界变化，不是“任何时间都能打断 GPU kernel”。本课程只实现这个思想的简化调度器，不复现 Orca 的全部架构。

## 2. 三种限制不能混为一谈

| 限制 | 单位 | 限制的对象 |
| :--- | :--- | :--- |
| max_active | 请求数 | 同时在活动集合中的序列数 |
| token_budget | 本轮输入 token 数 | 本轮 prefill 与 decode 的工作量 |
| KV capacity | 块数或字节 | 跨轮保留的上下文状态 |

一轮里给 A 处理 4 个 prompt token，不代表 A 占 4 个请求位置。让 4 个请求各 decode 一次，也不代表它们的历史 KV 只有 4 个 token。并发上限不是显存检查。

Phase1 已练过有限页池准入。Phase2 暂时隔离调度问题，用请求数上限而不接入真实块池；两个实验分别验证不同约束，不能宣称已经完成了显存安全的推理引擎。

## 3. 状态从哪里来

Lab5 的一条活动记录是 `(id, prefill_remaining, output_remaining)`：

- prefill_remaining > 0：仍有 prompt 要处理，暂不能执行 decode。
- prefill_remaining == 0：已完成 prompt，可执行一次 decode。
- output_remaining 始终为正；归零的请求已被引擎移出，不应再传入调度器。

例如 `[("P",5,3),("D",0,2),("E",0,1)]`，预算 1，只能选择 D decode；预算 2 可选择 D、E。不是看到队首 P 就停：这里先按阶段划分优先级，再在同阶段内保留原来的顺序。

Lab5 的函数只返回选择，不修改输入状态。引擎负责应用选择、产生输出时间、完成请求并释放位置。这种“决策与状态变更分开”使你可以独立测试：策略选错了，还是执行器把计数减错了？

## 4. 时间边界决定答案

Phase2 定义一轮为 `[t,t+1)`，所有动作在 t 决策，在 t+1 统一完成：

1. 加入 arrival <= t 的请求，按到达时间排序；同时间按原输入顺序。
2. 按 FIFO 补充活动请求，直到 max_active。
3. 按轮开始时的状态生成一次计划。
4. 执行计划，记录输出时间 t+1，移出完成者。
5. 到下一轮，才用刚腾出的位置接纳等待者。

特别注意：某条请求本轮完成 prefill，不能在同一轮又 decode。允许它在 prefill 结束时产生第一个输出，不等于允许多执行一次模型迭代。请求输出长度为 1 时，会直接释放位置。

时间轴无活动请求、无等待请求时，直接跳到下一次 arrival；不要为了相隔十亿个 tick 的两个请求循环十亿次。

## 5. 手算，再实现

只看 decode 阶段：A 还要 1 个输出，B 还要 3 个；C 正在等待。max_active=2，token_budget=2。

| 轮开始 | 选择 | 轮末变化 |
| :--- | :--- | :--- |
| 0 | A、B 各 decode 1 | A 结束，B 还需 2 |
| 1 | C 被接纳；B 优先 decode | C 能做多少 prefill 由剩余预算决定 |

预测：能否在第 0 轮 A 完成后，立刻在该轮再给 C 一份计划？不能，本模型一轮只有一次决策。若代码这样做，吞吐量会被人为夸大。

完成 Lab5 Q1 后，解释以下问题，再读 [L12](lec12.md)：

- 为什么预算已经用完，也必须验证输入尾部的非法记录？
- 为什么等待 FIFO 不等于所有阶段完全 FIFO？
- 一个请求已经释放并发位置，是否一定释放了全部可缓存的 KV 内容？回顾 L10 的引用与缓存区别。

## 6. 出口与局限

能独立画出 admitted → prefill → decode → finished，并说明只有一个输出 token 的特殊路径，才算掌握本讲。无限请求流中的公平性、抢占、取消、真实 KV 增长和设备执行尚未实现；“本例能结束”不能证明生产服务不会饥饿。
