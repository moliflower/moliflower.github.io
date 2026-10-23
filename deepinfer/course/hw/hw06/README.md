# HW6：延迟、聚合口径与 SLO goodput

先修 [L12](../../lectures/lec12.md)。约 2–3 小时。修改 [hw06.py](hw06.py)，用 [test_hw06.py](test_hw06.py) 验证。独立于调度器，仅根据已有时间戳计算指标。

## 1. Q1：单请求指标

`request_metrics(arrival,token_times)` 返回 `{"ttft":值,"tpot":值或None,"e2e":值}`。

- arrival 和每个时间必须是 Python int 或 float，有限且非负；bool、NaN、无穷、字符串拒绝。
- token_times 是非空 list，严格递增，每项 >= arrival；允许首 token 时间等于 arrival。
- TTFT=首时间-arrival；E2E=末时间-arrival；TPOT=(末-首)/(token数-1)。单 token 的 TPOT 是 None。
- 不修改输入；非法值抛 ValueError。正常测试使用可安全表示的有限数，不研究浮点溢出。

手算 arrival=2，token_times=[5,7,11]，得到 TTFT=3、TPOT=3、E2E=9。请解释为何不能用 9/3 作为定义上的 TPOT，即使这一组碰巧也是 3；再换 arrival=0、times=[5,7,11] 反驳该错误公式。

## 2. Q2：请求等权聚合

`aggregate_metrics(records)`：records 是 list，每项 tuple/list 恰含 `(arrival,token_times)`，沿用 Q1 契约，返回 `{"mean_ttft":值或None,"mean_tpot":值或None}`。

TTFT 对所有请求等权平均；TPOT 仅对有两个及以上输出的请求等权平均。空 records 两项都为 None；全是单 token 时 mean_tpot=None，不能把“无定义”伪装成零延迟。任何非法记录都抛 ValueError，不可先过滤掉它。不修改输入。

`[(0,[1,5]),(0,[2,3,4,5]),(1,[2])]`：mean_ttft=4/3，mean_tpot=(4+1)/2=2.5。按所有 token 间隔加权会得到 1.75，那是另一个统计量。

## 3. Q3：达标请求吞吐

`slo_goodput(records,ttft_limit,tpot_limit,duration)` 返回达标请求数 / duration。

- records 沿用 Q2；两个 limit 是有限非负 int/float，不含 bool；duration 同类型但必须 >0。
- 窗口固定 `[0,duration]`，所有 arrival 和输出时间必须落在窗口内，包括不达标的请求。
- TTFT <= ttft_limit 且 TPOT <= tpot_limit 才达标，等号算通过。单 token 请求只检查 TTFT。
- 空 records 返回 0，但仍验证全部参数；非法输入 ValueError；保留输入。

窗口 6、限制 2/2：`[(0,[2,4]),(0,[2,6]),(1,[2])]` 有两条通过，返回 2/6。若窗口改成 5，应该报错，而不是把第二条悄悄丢掉。

## 4. 运行与书面推导

从课程根目录运行（个人主页仓库先进入 `_courses/deepinfer`）：

```bash
python course/tools/grade.py hw06 --question q1 --unlock
python course/tools/grade.py hw06 --question q2 --hint 1
python course/tools/grade.py hw06
```

在自己的笔记中完成：单请求公式推导；请求平均与间隔平均的反例；单 token 策略；改变观察窗口后 goodput 的变化；为什么 Python 运行秒数不等于模拟器 tick。时间单位始终写在表头。

本批 HW6 30 分：Q1/Q2/Q3 各 5；公式和单位 6；两个错误口径的反例 6；不用答案口头说明 singleton 和窗口边界 3。测试只占代码部分，不会自动批改书面推导。
