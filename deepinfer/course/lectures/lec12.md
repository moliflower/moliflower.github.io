# 第 12 讲：Chunked prefill 与 TTFT / TPOT

先修：[L11](lec11.md)。约 60 分钟；配套 [Lab5 Q2](../labs/lab05/README.md)、[HW6](../hw/hw06/README.md)、[Project2 Phase2](../projects/project2_scheduler/README.md)。

## 1. 为什么把 prompt 拆成几轮

Prefill 可以一次处理多个 prompt 位置；普通 decode 每条序列每轮处理一个新位置。长 prompt 若一次占用很大的执行批次，可能延后正在流式生成的请求。Chunked prefill 把剩余 prompt 分成多次处理，使 prefill 工作能与 decode 在调度边界交错。

[vLLM 官方优化文档](https://docs.vllm.ai/en/stable/configuration/optimization/#chunked-prefill) 描述了 decode 优先、剩余 token 预算用于 prefill 的策略，以及预算对 TTFT/ITL 的影响（核对：2026-10-24）。这是理解系统的入口，不代表所有版本、所有模型都使用完全相同的实现。本课额外设定“每请求每轮最多一个 chunk”的教学规则，不把它冒充 vLLM 的配置项或源码复刻。

## 2. 三个数一起取最小值

先给每条入选 decode 请求分配 1 个 token，得到剩余预算 R。然后按活动顺序处理每条 prefill 请求：

```text
本次分配 = min(剩余 prompt token, chunk_size, R)
R = R - 本次分配
```

每条请求本轮最多被选一次。即使仍有预算，也不回头给同一请求第二个 chunk。chunk_size 是上限，不是必须补齐的长度；不能凭空生成 padding 工作，也不能因为不足一个整块就拒绝执行尾部。

例如活动队列 `[(A,5,3),(B,0,2),(C,2,1)]`，预算 4，chunk_size=2：

| 顺序 | 动作 | 消耗 | 剩余预算 |
| :--- | :--- | ---: | ---: |
| 1 | B decode | 1 | 3 |
| 2 | A prefill | 2 | 1 |
| 3 | C prefill | 1 | 0 |

C 仍有 1 个 prompt token 未处理，不能产生首 token。A 的剩余 prompt 是 3。B 的剩余输出数减一。本例没有任何请求完成 prefill 后又在同轮 decode。

## 3. 指标先定义事件，再写公式

设 arrival 是到达时刻，输出 token 时间依次为 t1 ... tn。时间必须来自同一时钟、使用同一单位。

```text
TTFT = t1 - arrival
E2E = tn - arrival
TPOT = (tn - t1) / (n - 1)       n > 1
```

- TTFT 包含排队和 prefill，不只是 prefill kernel 的执行时间。
- TPOT 是该请求在首 token 之后的平均输出间隔。n=1 没有间隔，本课记 None，而非 0。
- ITL 是逐个 token 间隔，例如 `[2,4]`；TPOT 是它们的均值 3。均值不能代替尾延迟。
- E2E/n 混入了等待和首 token 时间，不是这里定义的 TPOT。

手算：arrival=2，输出时间 `[5,7,11]`，TTFT=3，TPOT=3，E2E=9。若单位是秒，就全部是秒；若来自本课模拟器，就全部是 tick，不能改名为毫秒。

多请求汇总也要交代分母。请求 A 的 TPOT 为 4，B 为 1，平均请求 TPOT=(4+1)/2=2.5。若 A 只有一个间隔、B 有三个间隔，合并所有间隔后的平均值则是 7/4。两个统计量都可以有用，但不能混称同一个平均值。

## 4. 吞吐量不等于达标吞吐

定义观察窗口 `[0,T]`，只输入在此窗口内已经完成的请求，且明确排除未完成请求带来的统计偏差：

```text
输出吞吐 = 完成请求的输出 token 总数 / T
请求 goodput = 同时满足 TTFT 与 TPOT 限制的请求数 / T
```

HW6 采用包含等号的 SLO：TTFT <= limit、TPOT <= limit；单 token 请求只检查 TTFT。本课 goodput 单位是请求/单位时间，不是 token/秒。窗口中的空闲时间也算 T，不能为了让数字好看偷偷剔除。真实在线基准还要报告未完成/失败请求、负载和窗口边界处理方法。

例如窗口 6 秒，三个请求 `(arrival,token_times)` 为 `(0,[2,4])`、`(0,[2,6])`、`(1,[2])`。限制 TTFT<=2、TPOT<=2：第一、三条达标，goodput=2/6 请求/秒。总输出吞吐则是 5/6 token/秒，二者不是同一种量。

## 5. 小 chunk 一定更好吗

不一定。较小 chunk 给其他请求留下机会，但长 prompt 要更多轮才能到首 token，也可能增加运行时开销。更大的 token 批次可能改变一次迭代的实际时长，进而改变流式输出间隔。应在相同到达负载、模型、设备与统计口径下比较，不能只凭一个参数推导实际性能。

本周模拟器刻意将每轮固定为 1 tick，并要求 token_budget >= max_active。于是活动的 decode 每轮都能执行，所有多 token 请求的 TPOT 都是 1 tick；实验能展示首 token 排队变化，却**不能重现 GPU 上大 prefill 拖慢 decode 的物理时延**。这是模型的边界，不是“chunk size 不影响真实 TPOT”的证据。以后加入有测量依据的批次成本或真实设备执行，才能讨论物理时间。

## 6. 实验前的预测

用 Phase2 的 long_then_short 负载，固定预算 4，比较 max_active=1/2、chunk=1/2/4：

1. 哪些配置让短请求不必等待长请求整个结束？
2. 为什么 chunk=1 可能留有空预算，反而需要更多轮？
3. 改成所有请求同时到达，结论是否仍成立？
4. 报告中哪些数字来自测量，哪些来自教学假设？

先写预测，再运行实验。一个解释清楚的不理想结果，比一句没有条件的“优化加速了”更有价值。
