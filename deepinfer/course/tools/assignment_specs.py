"""Explicit question selectors and pre-code prediction checks.

Each entry: unittest selector(s), prediction, answer, three progressive hints.
Prediction answers are strings; the grader never evaluates input as code.
"""

SPECS = {
    "lab04": {
        "q1": ("TestAllocate", "容量 3，已用 2，再申请 2 是否成功？输入 yes/no", "no", "先检查总空闲数。", "失败前不能改 refs。", "按 ID 升序选 refs==0 的块，全部可用后设置为 1。"),
        "q2": ("TestRetain", "引用数 1 的块再被一个请求引用，引用数？", "2", "共享不复制物理块。", "先验证整批 ID。", "拒绝重复、空闲与越界 ID，再统一加一。"),
        "q3": ("TestRelease", "引用数 2 的块释放一次后，引用数？", "1", "另一个请求仍在使用。", "归零才可重新分配。", "先验证整批，再减一；失败不改状态。"),
        "q4": ("TestLocate", "table=[9,2,7]，block_size=4，token=4 的物理块？", "2", "逻辑页不是物理页。", "divmod 得逻辑页与偏移。", "用 table[逻辑页] 转换；返回二元组。"),
    },
    "hw05": {
        "q1": ("TestHit", "公共前缀 7 tokens，块大小 4，可复用几个 tokens？", "4", "只缓存完整块。", "遇到第一个不同 token 就停止。", "公共前缀长度 // block_size * block_size。"),
        "q2": ("TestSharing", "两条长度均为 4 的请求共享长度 4 的完整前缀，共占多少槽？", "4", "共享前缀只保留一次。", "每条尾部各自向上取整。", "prefix + sum(ceil((length-prefix)/B)*B)。"),
        "q3": ("TestEviction", "引用数为 1 的块可以淘汰吗？输入 yes/no", "no", "先过滤仍被引用的块。", "再按访问时间排序。", "refs==0 的 ID 按 (last_used, ID) 排序。"),
    },
    "project2": {
        "q1": ("TestAdmission", "FIFO 队首需要 2 块，但只剩 1 块，能越过它处理后面请求吗？输入 yes/no", "no", "本阶段严格 FIFO。", "先验证整条队列，再开始接纳。", "每请求向上取整页数；遇到第一个放不下的立即停止。"),
        "q2": ("TestReplay", "容量 3，活动请求占 2 块，空闲几块？", "1", "每步检查容量守恒。", "finish 后重试等待队列。", "每步保存 active 字典和 waiting 列表的独立副本。"),
    },
    "foundations": {
        "q0": ("TestExample", "add_example(2,5) 的返回值？", "7", "先运行已经写好的示例。", "参数 2 和 5 分别绑定到两个名字。", "return 把两数之和交给调用者。"),
        "q1": ("TestExpressions", "7 个元素，每元素 2B，总字节数？", "14", "先确定每个参数的单位。", "总量是数量乘单位大小。", "返回 count*element_bytes；负输入抛 ValueError。"),
        "q2": ("TestControl", "7 个元素，每组 3 个，需要几组？", "3", "先数满组，再看剩余元素。", "用 // 和 % 分别得到商和余数。", "余数非零时还需要一组；0 个元素需要 0 组。"),
        "q3": ("TestSequences", "clamp_lengths([2,8,0],5) 输出第二项？", "5", "逐个看长度是否超过上限。", "保留较小值，把过大值换成 limit。", "用新列表保存结果；不要修改输入列表。"),
        "q4": ("TestComposition", "prefix_offsets([2,0,3]) 最后一个边界？", "5", "先写第一个起点 0。", "每读一个长度，就把它加到此前累计值。", "输出含初始 0 和每次累计值，因此比输入多一个元素。"),
    },
    "lab01": {
        "q1": ("TestReduce", "[1,2,3,4,5] 成对归约一轮后有几个元素？", "3", "先写出每轮列表。", "奇数长度最后一个直接带到下一轮。", "每轮创建新列表，相邻两数相加；最后剩一个 float。"),
        "q2": ("TestSoftmax.test_online_state TestSoftmax.test_online_state_empty TestSoftmax.test_online_state_prefix_invariant", "已读 [2]，online state 的 s 是多少？", "1", "s 的指数都相对于当前 m。", "m 变大后，旧 s 需要换基准。", "先算 m_new，再用旧 m 更新 s，最后赋值 m。"),
        "q3": ("TestSoftmax.test_uniform TestSoftmax.test_large_values TestSoftmax.test_sums_to_one TestSoftmax.test_empty TestSoftmax.test_nonuniform_and_shift TestSoftmax.test_softmax_extreme_gap_and_copy", "softmax([1000,1000]) 的第一个概率？", "0.5", "复用 q2，先处理空列表。", "每项指数减去同一个 m。", "返回 exp(v-m)/s；输出列表是新对象。"),
        "q4": ("TestNorm.test_rmsnorm TestNorm.test_rmsnorm_weight TestNorm.test_rmsnorm_empty TestNorm.test_rmsnorm_length_mismatch TestNorm.test_rmsnorm_zero_eps_and_copy", "[3,4] 的均方是多少？", "12.5", "先检查 weight 长度。", "平方和要除以元素数量。", "scale=1/sqrt(mean_sq+eps)，逐项乘 x 和 weight。"),
        "q5": ("TestNorm.test_layernorm TestNorm.test_layernorm_bias TestNorm.test_layernorm_contract", "[1,2,3] 的均值是多少？", "2", "先算均值，再算偏差平方的均值。", "本题方差分母是 n，不是 n-1。", "归一化后乘 weight，再加 bias。"),
        "q6": ("TestTraffic.test_softmax_traffic TestTraffic.test_softmax_traffic_invalid TestTraffic.test_softmax_traffic_boundaries", "n=2，3 次读一次写，float32 请求字节数？", "32", "读写次数和数据类型都要计入。", "总访问次数为 (passes+1)*n。", "只接受 passes=1/2/3；每次访问乘 4B。"),
        "q7": ("TestTraffic.test_rmsnorm_traffic TestTraffic.test_rmsnorm_traffic_boundaries", "n=2，on-chip RMSNorm，仅计 x/y 字节数？", "16", "此函数只数 activation。", "on-chip 读一次，非 on-chip 读两次。", "再加一次输出写；需要 weight 流量时另加 4n。"),
    },
    "lab02": {
        "q1": ("TestBasics.test_matmul TestBasics.test_matmul_mismatch TestBasics.test_matmul_rectangular_and_copy", "(2,3) @ (3,4) 的输出列数？", "4", "给 i/j/k 写出范围。", "每一输出格有独立 total。", "C[i][j] 累加 A[i][p]*B[p][j]。"),
        "q2": ("TestBasics.test_transpose TestBasics.test_transpose_roundtrip_and_copy", "[[1,2,3],[4,5,6]] 转置后行数？", "3", "转置交换行列。", "输出第 j 行来自输入第 j 列。", "外层遍历原列，内层遍历原行。"),
        "q3": ("TestBasics.test_softmax_rows TestBasics.test_softmax_rows_masked_and_invalid", "softmax([0,-inf]) 的第一个概率？", "1", "一行一个归一化分母。", "减该行 max 后再取指数。", "mask 的 exp(-inf)=0；每行至少一个有限值。"),
        "q4": ("TestBasics.test_causal_mask TestBasics.test_causal_mask_shape_and_copy", "长度 3 的 causal 第 1 行保留几个位置？", "2", "位置从 0 编号。", "j<=i 保留，j>i 屏蔽。", "新建矩阵；上三角填 -math.inf。"),
        "q5": ("TestAttention.test_identity TestAttention.test_non_causal TestAttention.test_future_values_cannot_change_past TestAttention.test_attention_rectangular_and_validation", "causal 第 0 行只能看到第几个 token？", "0", "先检查 Q/K/V 的 shape。", "按 matmul→scale→mask→softmax→matmul 组合。", "先 mask 后 softmax，最后 probs@V；不要对输出再 mask。"),
        "q6": ("TestAttention.test_multi_head TestAttention.test_multi_head_invalid TestAttention.test_multi_head_distinct_subspaces TestAttention.test_multi_head_contract", "d_model=4,num_heads=2，每个 head 的维度？", "2", "沿最后一维切片。", "每个 head 用自己的 d_head 做缩放。", "各 head 独立 causal attention，再按列拼接每个 token 的结果。"),
        "q7": ("TestCost.test_flops TestCost.test_flops_causal_rectangular TestCost.test_causal_diagonal TestCost.test_flops_boundaries", "s=3 的含对角线下三角有几个有效配对？", "6", "数 1+2+...+s。", "每个配对两次长度 d 的点积，按 4d FLOPs 计。", "causal 用 s*(s+1)//2；full 用 sq*sk。"),
        "q8": ("TestCost.test_bytes_materialized TestCost.test_bytes_read_once_lower_bound TestCost.test_io_tiny TestCost.test_bytes_boundaries", "三阶段 attention 中 S/P 一共几次完整读写？", "4", "列 Q/K/V 读和 O 写。", "再列 S 写读、P 写读。", "base=b*h*(2sq+2sk)*d；物化额外加 4*b*h*sq*sk，最后乘元素字节。"),
    },
    "lab03": {
        "q1": ("TestContinuous.test_append_get TestContinuous.test_append_invalid TestContinuous.test_append_input_copy_and_atomicity", "空 cache 第一次 append 返回的位置？", "0", "先检查向量长度，再改变状态。", "保存每个输入的新副本。", "append 到 k/v，返回新长度减 1。"),
        "q2": ("TestContinuous.test_copy_semantics TestContinuous.test_get_invalid TestContinuous.test_free TestContinuous.test_get_range_contract", "get(1,3) 返回几个 token？", "2", "区间右端不包含。", "要求 0<=start<=end<=len(cache)。", "复制外层和每一行，free 后允许 get(0,0)。"),
        "q3": ("TestPaged.test_block_allocation TestPaged.test_new_block_id_after_free TestPaged.test_free TestPaged.test_append_boundary_and_atomicity", "length=5,block_size=2，已分配几个 slot？", "6", "只在新页起点分配。", "length%block_size==0 时追加物理块编号。", "将输入副本放入末页，更新 length，返回旧 length。"),
        "q4": ("TestPaged.test_gather_across_blocks TestPaged.test_empty_range TestPaged.test_invalid_range TestPaged.test_physical_ids_are_not_logical_indices TestPaged.test_copies_on_append_and_get TestPaged.test_get_all_ranges", "pos=5,block_size=2，页内 offset？", "1", "商是逻辑页号，余数是 offset。", "逻辑页号还要查 block_table。", "physical=block_table[pos//block_size]；从 blocks[physical] 取副本。"),
    },
    "hw01": {
        "q1": ("TestIntensity", "12 FLOPs / 4 bytes 的 AI？", "3", "先写单位 FLOP/B。", "分母必须为正。", "返回 flops/bytes_moved。"),
        "q2": ("TestRoofline", "搬 1GB，带宽 100GB/s，理想秒数？", "0.01", "计算搬运和计算两个下界。", "单位统一为 B/s、FLOP/s。", "取 max；相等标 memory，与 Lab0 约定相同。"),
        "q3": ("TestKV", "K 和 V 带来的最前面系数？", "2", "写出每层每 token 的标量数。", "使用 KV heads 而非 query heads。", "2*L*Hkv*D*S*batch*bytes。"),
        "q4": ("TestMLA", "latent=512,RoPE=64，每 token 每层存几个标量？", "576", "K/V 共享压缩 latent。", "先将 rank 和 rope_dim 相加。", "L*(rank+rope_dim)*S*batch*bytes，无额外系数 2。"),
        "q5": ("TestAttention", "s=1 的 causal 配对数？", "1", "含对角线，不能直接砍半。", "有效配对是 s(s+1)/2。", "将有效配对乘 4*batch*heads*d。"),
        "q6": ("TestQuantize", "对称 3-bit 整数量化的 qmax？", "3", "这是整数网格。", "qmax=2**(bits-1)-1；全零用 scale=1。", "先 round(v/scale)，再 clamp 到 [-qmax,qmax]。"),
        "q7": ("TestDequantize", "q=2,scale=0.25，反量化结果？", "0.5", "还原到浮点尺度。", "每一个整数码乘相同 scale。", "返回新列表 [q_i*scale]。"),
        "q8": ("TestQuantError", "无 clipping 且 scale=0.2，最大舍入误差上界？", "0.1", "组合量化和反量化。", "比较每项绝对误差。", "取最大值；空输入返回 0.0，但仍要检查 bits。"),
        "q9": ("TestBudget", "每 token 8B，预算 31B，最多存几个 token？", "3", "将容量公式倒过来使用。", "用 Q3 算一个 token 的字节数，剩余不足一个的字节不能使用。", "返回 budget_bytes // per_token；检查 per_token 必须正。"),
    },
    "hw02": {
        "q1": ("TestFlops", "s=2 的 causal 有效配对数？", "3", "复用有效配对推导。", "检查 causal 时 sq==sk。", "4*b*h*d*(s*(s+1)//2)。"),
        "q2": ("TestTraffic", "S 写、S 读、P 写、P 读的系数？", "4", "先画三个阶段的箭头。", "容量计一次，流量按每次读写计。", "返回 (base+4*b*h*sq*sk)*bytes，False 时仅 base。"),
        "q3": ("TestAI", "F=64,B=32，模型 AI？", "2", "组合 q1/q2。", "零字节无法定义 AI。", "拒绝分母为零，返回 FLOPs/模型字节数。"),
        "q4": ("TestBound", "100 TFLOP/s、1000 GB/s，拐点 FLOP/B？", "100", "两个硬件数字都要为正。", "T 与 G 差 1000 倍。", "本题 ai>=ridge 返回 compute；等号是两条限制同时达到。"),
        "q5": ("TestKV", "seq 翻倍时标准 KV 容量翻几倍？", "2", "沿用 HW1 的单位。", "batch 表示同长度序列数量。", "2*L*Hkv*D*S*batch*bytes。"),
        "q6": ("TestGrowth", "每 token 每层 8B，2 层新增 3 token，增长多少 B？", "48", "函数返回增量，不是总量。", "用 new_tokens 替换 seq_len。", "调用 q5 的公式。"),
        "q7": ("TestPages", "10 token 每页 4 slot，需要几页？", "3", "用整数向上取整。", "page_size 必须正，seq_len 非负。", "(seq_len+page_size-1)//page_size。"),
        "q8": ("TestWaste", "10 token 每页 4 slot，浪费几个 slot？", "2", "先求容量再减实际长度。", "复用 num_pages。", "pages*page_size-seq_len。"),
        "q9": ("TestPagedBytes", "1000 token 每页 16 slot，分配几个 slot？", "1008", "按分配长度算存储。", "每条序列的尾页都要保留。", "allocated=num_pages*S_page，然后代入 KV bytes。"),
        "q10": ("TestPrefixSavings", "已有一份前缀，3 个新请求共享它，避免几份额外副本？", "3", "题目中的 batch 是新消费者数。", "已有那份前缀仍然保留。", "省 batch*一份前缀容量；若总共只有 batch 份则是 (batch-1)。"),
        "q11": ("TestRaggedBatch", "[1,16,17] 三条独立请求，page=16，总共几页？", "4", "先分别写出每条请求的尾页。", "每条独立向上取整后求和，不能先把长度相加。", "对每个长度调用 Q9，batch=1，累加字节；空列表也要验证公共参数。"),
    },
    "project1": {
        "q1": ("TestPhase1.test_vector_add TestPhase1.test_vector_add_mismatch TestPhase1.test_vector_add_contract", "[1,2]+[3,4] 逐项相加的第二项？", "6", "可迁移自己的 prep 实现。", "长度不匹配要拒绝。", "输出是新列表；保留输入。"),
        "q2": ("TestPhase1.test_sum_reduce TestPhase1.test_sum_reduce_tree_order", "5 个元素成对归约需要几轮？", "3", "可复用自己 Lab1 的树归约。", "每轮长度向上除以 2。", "直到剩一个元素；空输入为 0.0。"),
        "q3": ("TestPhase1.test_softmax TestPhase1.test_softmax_large TestPhase1.test_softmax_contract", "[1000,1000] softmax 第一项？", "0.5", "先减 max。", "复用归约求指数和。", "保持输入不变，空列表直接返回。"),
        "q4": ("TestPhase1.test_rmsnorm TestPhase1.test_rmsnorm_mismatch TestPhase1.test_rmsnorm_contract", "[3,4] 均方？", "12.5", "迁移 Lab1。", "验证权重长度。", "x*w/sqrt(mean(x²)+eps)。"),
        "q5": ("TestPhase1.test_matmul_naive TestPhase1.test_matmul_inner_mismatch TestPhase1.test_matmul_naive_contract", "(2,3)@(3,2) 输出有几格？", "4", "每格一个点积。", "输出行必须独立。", "i/j/p 三重循环；每格开始前清零。"),
        "q6": ("TestPhase2.test_tiled_small TestPhase2.test_tiled_rectangular TestPhase2.test_tiled_odd_sizes TestPhase2.test_tiled_contract", "tile=16,n=3，每个维度需要几个 tile？", "1", "先完成 q9 引擎。", "q6 只需要提取 q9 返回的矩阵。", "return matmul_tiled_profile(a,b,tile)[0]。"),
        "q7": ("TestPhase2.test_traffic TestPhase2.test_traffic_tile_helps TestPhase2.test_traffic_partial_tiles TestPhase2.test_traffic_invalid TestPhase2.test_traffic_monotonic_and_dtype", "3x3 float32 GEMM，A/B/C 各读或写一次，最少几个字节？", "108", "尾块必须用 ceil。", "q=ceil(n/tile)，每个 A/B 元素读 q 次。", "tiled=(2*n*n*q+n*n)*dtype_bytes。"),
        "q8": ("TestPhase3", "模型时间下界是两个时间的 max 还是 sum？", "max", "复用 Roofline。", "模型与 Python 实测单位要区分。", "计算两条下界，取最大值并标 bound。"),
        "q9": ("TestProfile", "A(2,3),B(3,2),tile=2，A 共需 load 几次？", "6", "先列 output tile，再遍历 k tile。", "A/B 先通过 MatrixMemory 搬进局部数组。", "乘法循环只读局部 tile，C accumulator 跨 k tile 保留，最后每格 store 一次。"),
    },
}
