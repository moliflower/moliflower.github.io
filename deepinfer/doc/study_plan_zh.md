# DeepInfer 学习路线：算子 / 推理 Infra 方向（2028 届秋招）

> 版本：2026-10-22  
> **课程执行版（含本地 HTML 网站）**：[`../course/README.md`](../course/README.md)  
> 适用：想投推理框架、GPU/NPU 算子、高性能计算、AI 系统方向的应届生  
> 时间假设：你说的“28 秋招”按 **2028 届 = 2027 年 7–9 月秋招** 倒排，从今天起约 11 个月。如果你其实指“2028 年秋天”，时间会更宽裕，见附录 D。  
> 结论先行：**DeepInfer 在你这台笔记本上跑不起来，也不应该试图跑起来。把它当成一张工业级代码地图，重点是把里面的机制拆成可在本地复现的小实验。**

---

## 一页纸版本

| 项目 | 结论 |
| :--- | :--- |
| 主攻方向 | 算子/高性能计算 70%，推理引擎 30% |
| 总周期 | 约 46 周：8 周地基 + 16 周算子 + 12 周引擎 + 10 周项目/面试（缓冲穿插） |
| 每周投入 | 标准 12–15 小时；保底 8 小时；冲刺不超过 20 小时 |
| 核心产出 | 12 个 Lab（0 号是环境准备，不计入）、3 个项目、1 个开源 PR、3 篇长文笔记 |
| 最终验收 | 能白板讲清一次请求的生命周期；能独立写出 8 个以上 kernel；能设计一个满足 SLO 的调度方案 |
| 现在立刻做 | ① 初始化 vendor 子模块 ② 在 WSL 建 Python 3.12 环境 ③ 跑通第一个 TileLang kernel ④ 画一张请求生命周期图 ⑤ 建 `notes/` 和 `labs/` |

**三条主线：**

1. **读架构**：DeepInfer 的 Python 引擎 + C++ 调度器 + TileLang 算子 + vendor 库，四层各自解决什么问题。
2. **写算子**：从 memory-bound 融合算子开始，到 paged KV、decode attention、FP8 GEMM、MoE、sampling。
3. **做系统**：用一个 0.5B 小模型 + 自己写的 paged KV/continuous batching，复现推理引擎的核心机制。

**五件不要做的事：**

- 不要按 README 在 Windows 上 `./build.sh`。这个仓库依赖 Linux 头文件（`sys/mman.h`、`unistd.h`、`fcntl.h`）、GCC 13、CUDA 13 和 SM100。
- 不要下载 DeepSeek-V4.1-Flash 的完整 checkpoint。你的 8GB 显存和 7.5GB WSL 内存都不够，下载和加载会浪费几天。
- 不要追 SM100-only 的特性（tcgen05、完整 TMA/cluster、FP4 路线）。你的 RTX 5060 是 SM120，消费级 Blackwell 和 B200 不是一回事。
- 不要从 `fused_transform.py`（1238 行）或 `scheduler.hpp`（1168 行）第一行开始读。先读小文件、先读测试、先读调用方。
- 不要“只读不写”。这个仓库 5.3 万行，读完不等于会；每读一个模块，必须有一个能跑的最小复现。

---

## 1. 你的环境与可行性（实测）

我在你的机器上确认了这些事实：

| 项目 | 实际情况 | 仓库要求 | 结论 |
| :--- | :--- | :--- | :--- |
| 操作系统 | Windows + WSL2 Ubuntu 26.04 | Linux | 用 WSL，不要在 Windows 原生编译 |
| GPU | RTX 5060 Laptop，8GB，compute capability 12.0（SM120） | NVIDIA SM100（B200 级） | 不能跑完整模型；可跑小 kernel 和小模型 |
| 驱动 | 577.05，支持到 CUDA 12.9 | CUDA 13 | 本地用 cu12x 的 PyTorch，不要装 `requirements-nvidia.txt` 里的 `torch==2.13.0+cu132` |
| WSL Python | 3.14.4 | 3.12 | 用 `uv` 或 apt 装 3.12 的 venv |
| WSL 内存 | 7.5GB | 大内存 | 如果主机内存足够，可在 `.wslconfig` 调到 8–12GB（给 Windows 留足内存）；只跑 0.5B–1.5B 模型 |
| WSL 磁盘 | 954GB 可用 | — | 充足 |
| CUDA Toolkit | 未安装（无 `nvcc`、`ncu`、`nsys`） | CUDA 13 | 装 CUDA 12.9 Toolkit；`ncu` 在 WSL 可能受限，优先 `nsys`/`torch.profiler` |
| vendor 子模块 | 6 个目录都是空的 | DeepGEMM/FlashMLA/DeepEP 等 | 必须 `git submodule update --init --recursive` |
| 仓库规模 | 约 52.9k 行 Python/C++/CUDA，93 个 TileLang 算子文件，19 个测试文件 | — | 不可能全读；按 ROI 读 |

### 1.1 能做什么，不能做什么

| 任务 | 本地可行性 | 替代方案 |
| :--- | :--- | :--- |
| 阅读引擎/调度器/模型代码 | ✅ 完全可行 | VS Code + `rg` + 调用图 |
| 写 TileLang / Triton / CUDA 小 kernel | ✅ 可行（小 shape） | 8GB 显存足够做 micro-benchmark |
| 跑 0.5B–1.5B 模型推理 | ✅ 可行（bf16/4bit） | Qwen2.5-0.5B、TinyLlama-1.1B |
| 跑 DeepInfer 完整 engine | ❌ 不可行 | 只读代码；或租 8×H100/A100 做一次端到端 |
| 跑 DeepEP / 多卡 NCCL | ❌ 本地不可行 | 2–8 卡云主机；或写通信模拟器 |
| 跑 Ascend 后端 | ❌ 无硬件 | 只读 `asc/` 代码；目标华为时再深入 |
| 用 `ncu` 做细粒度 profile | ⚠️ WSL 可能受限 | `nsys`、`torch.profiler`、CUDA events；云上补 `ncu` |
| 复现 README 的 Pareto 曲线 | ❌ 需要 4/64 张 SM100 | 用模拟器 + 小规模实测，理解方法而不是数字 |

### 1.2 环境准备清单

按顺序做，每步都要有验收：

```bash
# 1. 进入 WSL
wsl -d Ubuntu

# 2. 基础工具
sudo apt update
sudo apt install -y build-essential git curl wget unzip
# 如果源里有 python3.12，也可以直接装；没有就用下面的 uv
# sudo apt install -y python3.12 python3.12-venv

# 3. 如果 apt 没有 python3.12，用 uv（推荐，能管理任意 Python 版本）
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12

# 4. 建独立环境（不要污染系统 Python）
uv venv ~/venvs/deepinfer-lite --python 3.12
source ~/venvs/deepinfer-lite/bin/activate

# 5. PyTorch：RTX 5060 是 SM120，需要 CUDA 12.8+ 的 wheel
#    具体版本按当时可用 wheel 调整；不要用 cu132
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# 6. TileLang（仓库固定 0.1.15；做实验可以用相近版本）
uv pip install tilelang==0.1.15 numpy

# 7. 验证 GPU
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

验收标准：最后一行打印出 `RTX 5060 ... (12, 0)`，且 `torch.cuda.is_available()` 为 `True`。

CUDA Toolkit 的选择：

- 只跑 PyTorch：wheel 自带 runtime，不一定要装完整 Toolkit。
- 要写 CUDA C++ 或让 TileLang 走 `nvcc`：装 CUDA 12.9 Toolkit。Ubuntu 26.04 的 apt 源可能还没被 CUDA 官方支持，优先用 `conda`/`uv` 的 `cuda-toolkit` 包，或直接用 Docker 镜像。
- `ncu`/`nsys`：先装 Nsight Compute/Systems；WSL 下如果权限或计数器受限，不要死磕，云上补。

初始化 vendor 子模块（读 DeepGEMM/FlashMLA/DeepEP 的源码需要）：

```bash
cd /mnt/c/Users/YuhanDeng/Desktop/DeepInfer
git submodule update --init --recursive
```

这一步会下载几个仓库，可能较慢；如果网络不稳，可以先只初始化最需要的三个：

```bash
git submodule update --init vendor/DeepGEMM vendor/FlashMLA vendor/DeepEP
```

> 注意：`vendor/` 里是独立项目，不要和 DeepInfer 的代码混在一起读。先读它们的 README 和 `tests/`，再读 kernel。

> WSL 访问 `/mnt/c` 的 IO 较慢：读代码可以，编译/跑 benchmark 建议把 `labs/` 放到 `~/labs`，不要在 `/mnt/c` 下建 venv 或编译。

---

## 2. 仓库地图：DeepInfer 到底是什么

### 2.1 一句话

DeepInfer 是 DeepSeek 生产推理框架的开源版：**Python 负责引擎编排，C++ 负责调度和元数据，TileLang 负责融合算子，vendor 库负责 GEMM/Attention/通信**，目标是在 NVIDIA SM100 和华为 Ascend 950 上跑 DeepSeek-V4.1-Flash 的离线推理。

### 2.2 分层架构

```
examples/  eval/  tests/
        │
        ▼
deepinfer/entrypoints/llm.py            ← 用户 API：generate()
        │
        ▼
deepinfer/engine/llm_engine.py          ← 请求生命周期、异步调度、输出处理
        │  pybind
        ▼
csrc/engine/                            ← C++ Scheduler、BlockManager、Sequence、ZOS
        │
        ▼
deepinfer/worker/                       ← Worker、ModelRunner、CacheEngine、CUDA Graph
        │
        ▼
deepinfer/model_executor/models/        ← DeepseekModel、DecoderLayer、Attention、MoE、MTP
        │
        ▼
deepinfer/model_executor/layers/        ← Linear、RMSNorm、RoPE、Sampler
        │
        ▼
tilelang_ops/{cuda,asc}/                ← 融合算子（norm/rope/KV/sampling/...）
        │
        ├── vendor/DeepGEMM               ← FP8/FP4 GEMM、grouped GEMM、mega MoE
        ├── vendor/FlashMLA               ← MLA prefill/decode
        ├── vendor/DeepEP                 ← EP all-to-all
        └── vendor/DeepSelect             ← 稀疏选择
        │
        ▼
GPU / NPU
```

### 2.3 目录与 ROI

| 目录 | 规模（约） | 内容 | 学习优先级 |
| :--- | ---: | :--- | :--- |
| `deepinfer/engine/` | 1.7k 行 | `LLMEngine`、统计、profile server | ★★★★★ |
| `csrc/engine/` | 4.6k 行 | C++ 调度器、block manager、sequence、ZOS、metadata | ★★★★★ |
| `deepinfer/worker/` | 2.4k 行 | ModelRunner、CacheEngine、CUDA Graph、ZOS 状态 | ★★★★★ |
| `deepinfer/model_executor/models/` | 6.1k 行 | DeepSeek 模型、decoder layer、attention、MoE、MTP | ★★★★☆ |
| `deepinfer/model_executor/layers/` | 22.5k 行 | 算子封装 + TileLang 算子 | ★★★★★ |
| `deepinfer/model_executor/layers/tilelang_ops/` | 21.0k 行 | 93 个算子文件，CUDA/Ascend 两套实现 | ★★★★★ |
| `deepinfer/model_executor/parallel_utils/` | 约 1k 行 | DP/EP/CP 进程组、userbuffer、通信 | ★★★★☆ |
| `deepinfer/model_executor/eplb/` | 约 0.5k 行 | 专家负载均衡策略与权重 shuffle | ★★★★☆ |
| `deepinfer/distributed/` | 0.4k 行 | 共享内存广播（MessageQueue） | ★★★☆☆ |
| `deepinfer/structure_output/` | 0.5k 行 | 语法自动机、token mask | ★★★☆☆ |
| `deepinfer/ve/` | 0.8k 行 | 多模态预处理、图像 processor | ★★☆☆☆ |
| `deepinfer/npu_ops/` | 0.2k 行 | Ascend 专用算子 | ★★☆☆☆ |
| `deep_cluster/` | 0.5k 行 | 硬件发现、NUMA/CPU 绑定、拓扑 | ★★★☆☆ |
| `tests/` | 2.5k 行 | 可执行的规格说明 | ★★★★★ |
| `examples/` | 2.6k 行 | 离线推理、benchmark、sweep | ★★★★☆ |
| `vendor/` | 外部仓库 | DeepGEMM/FlashMLA/DeepEP/DeepSelect | ★★★★☆ |

**ROI 分级：**

- **S 级（必须精读）**：scheduler、KV cache、attention、MoE/EP、sampling/spec decode、TileLang 算子、benchmark 方法。
- **A 级（理解即可）**：parallel_state、userbuffer、EPLB、CUDA graph、异步调度、structured output。
- **B 级（按需）**：VE、eval、Ascend、batch invariant/deterministic。
- **C 级（知道存在即可）**：Engram、HC/MHC、YoCo 的模型细节、V4.1 特有配置。

### 2.4 一次请求的生命周期

以 `examples/offline_inference.py` 为例：

1. `LLM.generate()` 进入 `deepinfer/entrypoints/llm.py`。
2. `LLMEngine` 做 tokenize、构造 `Sequence`，加入 C++ `Scheduler`。
3. `Scheduler::_schedule()` 决定这一步做 prefill 还是 decode，分配 KV block，产出 `SchedulerOutputs`。
4. `ModelRunner` 把 `SchedulerOutputs` 变成 `InputMetadata`（`prepare_tensor_input_v4.cpp`：positions、slot mapping、block table、压缩 KV 元数据）。
5. `DeepseekModel.forward()` 逐层执行：embedding → decoder layer → attention/MLP → final norm → LM head。
6. attention 走 FlashMLA + TileLang 融合算子；MoE 走 DeepGEMM/DeepEP；KV 写入 `GPUCacheInterface`。
7. `Sampler` 产生 token（可能包含 DSpark/DFlash 的 draft + verification）。
8. `process_output` 更新 sequence；完成的请求变成 `RequestOutput` 返回。
9. 异步调度把 `schedule(N+1)`、`post-process(N-1)` 与 GPU 的 `forward(N)`、`sample(N)` 重叠。
10. decode 阶段用 CUDA Graph 捕获，减少 launch overhead；ZOS 把元数据搬到大页/固定内存，降低 CPU 开销。

**练习**：不要看代码，先自己画一遍这个流程；然后打开 `llm_engine.py` 和 `model_runner.py`，把每一步对应到函数名。画不出来就说明还没读懂。

### 2.5 八个核心机制

| 机制 | 解决什么问题 | 关键文件 | 面试价值 |
| :--- | :--- | :--- | :--- |
| PagedAttention / BlockManager | KV cache 按固定大小 block 管理，支持前缀共享和碎片控制 | `csrc/engine/block_manager.hpp`、`worker/cache_engine.py` | ★★★★★ |
| Continuous batching + chunked prefill | 动态合批，prefill 分块，控制 TTFT/TPOT | `csrc/engine/scheduler.hpp` | ★★★★★ |
| Prefix cache / KV store | 复用已计算的 KV，降低重复 prefill | Q&A、`cache_engine.py`、外部 kvcache-store | ★★★★★ |
| SWA + state cache | 滑动窗口注意力，固定大小 state slot，支持 bounded replay | `v4_modules/yoco.py`、`cache_engine.py`、`state_cache.py` | ★★★★☆ |
| MLA + sparse indexer | 压缩 KV 的 attention，MQA indexer 选 top-k token | `v4_modules/mla_attn.py`、`index_attn.py`、`attn.py` | ★★★★★ |
| MoE EP + EPLB | 专家并行、冗余专家、按负载重排专家 | `v4_modules/moe.py`、`mlp.py`、`eplb/policy.py` | ★★★★★ |
| 投机解码 DSpark/DFlash | draft token + confidence + 变长验证，提高 decode 吞吐 | `models/mtp/dflash.py`、`layers/sampler.py` | ★★★★★ |
| 异步调度 / CUDA Graph / ZOS | 把 CPU 调度、后处理与 GPU 执行重叠，消除气泡 | `csrc/engine/zos.cpp`、`worker/cuda_graph_runner.py` | ★★★★☆ |

### 2.6 术语表（先背这 30 个）

| 术语 | 一句话解释 | 在仓库哪里 |
| :--- | :--- | :--- |
| Continuous batching | 每个 step 动态加入/移除请求，而不是等整批结束 | `scheduler.hpp` |
| Chunked prefill | 把长 prompt 的 prefill 切成多块，和 decode 混跑 | `scheduler.hpp` |
| Prefix cache | 相同前缀的 KV 直接复用 | `cache_engine.py`、Q&A |
| PagedAttention | KV 按 block 分页，block table 做逻辑到物理映射 | `block_manager.hpp` |
| Block table | 每个 sequence 的物理 block 列表 | `sequence.hpp` |
| KV cache | 缓存 attention 的 K/V，避免重复计算 | `cache_engine.py` |
| SWA | Sliding Window Attention，只保留最近窗口 | `yoco.py`、`state_cache.py` |
| MLA | Multi-head Latent Attention，压缩 KV 的 attention | `mla_attn.py` |
| MQA/GQA | 多个 query head 共享 K/V head | `index_attn.py` |
| MoE | Mixture of Experts，每个 token 只激活少数专家 | `moe.py`、`mlp.py` |
| EP | Expert Parallel，专家分散到不同 GPU | `parallel_state.py` |
| EPLB | Expert Parallel Load Balancing，专家负载均衡 | `eplb/policy.py` |
| DP/TP/PP/CP | 数据/张量/流水/上下文并行 | `parallel_state.py` |
| Speculative decoding | draft 模型先猜，target 模型验证 | `mtp/dflash.py` |
| Draft / Target | 猜测模型 / 验证模型 | `sampler.py` |
| Acceptance rate | draft token 被接受的比例 | `sampler.py` |
| Verification | 用 target 分布验证 draft token | `sampler.py::_verification` |
| CUDA Graph | 把一串 kernel launch 捕获成图，减少 CPU 开销 | `cuda_graph_runner.py` |
| ZOS | Zero-Overhead Scheduling，元数据下沉到 GPU | `zos.cpp`、`zos_status.py` |
| TTFT | Time To First Token | `benchmark.py` |
| TPOT | Time Per Output Token | `benchmark.py` |
| Goodput | 满足 SLO 的有效吞吐 | Q&A、benchmark |
| Roofline | 性能上界 = min(峰值算力, 带宽 × 算术强度) | 你自己算 |
| MFU | Model FLOPs Utilization | 你自己算 |
| Arithmetic intensity | FLOPs / Bytes | 你自己算 |
| TMA | Tensor Memory Accelerator，异步搬 tensor | CUDA 13 / SM90+ |
| Tensor Core | 矩阵乘专用单元 | GEMM |
| FP8/FP4 | 低精度浮点格式 | `cuda/quant.py`、Q&A |
| MXFP8/MXFP4 | 带 block scale 的微缩放格式 | `common.py`、Q&A |
| ue8m0 | 8-bit 无符号指数，用作 scale | `cuda/quant.py` |
| All-to-all | EP 中 token 发给对应专家所在 GPU | DeepEP |
| RDMA / 3FS | 远程直接内存访问 / 分布式文件系统 | Q&A |

---

## 3. 学习路线总览

| 阶段 | 周次 | 主题 | 仓库材料 | 动手产出 | 验收 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| P0 地基 | 1–8 | CUDA、TileLang、量化、attention 基础 | `norm.py`、`gather.py`、`store_kv_cache.py` | 6 个小 kernel + 1 篇架构笔记 | 能独立写 memory-bound kernel |
| P1 算子 | 9–24 | 融合算子、KV/state cache、GEMM/MoE、sampling/spec | `tilelang_ops/cuda/`、`mlp.py`、`moe.py`、`sampler.py` | 10 个 Lab + kernel zoo | 每个 kernel 有正确性/性能/roofline |
| P2 引擎 | 25–36 | 调度、KV 管理、并行通信、投机解码、异步 | `scheduler.hpp`、`cache_engine.py`、`parallel_state.py` | mini engine v1 + 调度模拟器 | 能白板讲清生命周期 |
| P3 项目 | 37–46 | 项目打磨、开源、面试 | 全部 | 3 个项目 + 1 PR + 3 篇博客 | 通过模拟面试 |
| 缓冲 | 穿插 | 复习、补漏、实习 | — | — | 不欠账 |

**时间预算（标准档，12–15 小时/周）：**

- 4 小时读代码/论文（工作日晚 1 小时 × 4）
- 6 小时写代码/跑实验（周末两个下午）
- 2 小时写笔记/博客（周日晚）
- 1.5 小时复盘/整理（周日晚）
- 留 1 天完全休息，避免 11 个月拉伤。

**三档进度：**

| 档位 | 每周 | 适用 | 调整 |
| :--- | :--- | :--- | :--- |
| 保底 | 8h | 课业/实习忙 | 只做 P0 + Track A/B，项目和博客减半 |
| 标准 | 12–15h | 默认 | 按本文执行 |
| 冲刺 | 18–20h | 假期 | 加 Track C/D + 开源，不超过 2 个月 |

---

## 4. Phase 0：地基（第 1–8 周）

目标：**能读懂 TileLang kernel，能写出正确的 memory-bound kernel，能解释 attention 的 online softmax。**

### Week 1：仓库地图 + 环境

读：

- `README.md`（107 行）
- `doc/inference_deployment_qa.md`（312 行，重点读系统概览、EPLB、KV cache、投机解码、异步调度）
- `examples/README.md`（benchmark 方法、指标定义）
- `CONTRIBUTING.md`（测试标记、格式化）
- `deepinfer/entrypoints/llm.py`（173 行）
- `deepinfer/engine/llm_engine.py` 的前 200 行（不要试图一次读完 872 行）

做：

- 画一张自己的架构图（纸笔或 Excalidraw），标出 10 个步骤对应的文件。
- 建 `notes/00_overview.md`，用 300 字回答：DeepInfer 由哪几层组成？每层用什么语言？为什么这样分层？
- 建 `notes/01_glossary.md`，把上面 30 个术语用自己的话写一遍。
- 初始化 vendor 子模块。

验收：能不看资料讲 5 分钟“一次请求怎么变成 token”。

### Week 2–3：CUDA 基础

读/学：

- PMPP（Programming Massively Parallel Processors）第 1–6 章：线程模型、内存层次、coalescing、shared memory、bank conflict、occupancy。
- CUDA C++ Programming Guide 的 execution model、memory hierarchy、best practices。
- `nsys` / `torch.profiler` 的基本用法。

写（每个都要 benchmark）：

1. vector add：理解 grid/block、global memory。
2. reduction：warp shuffle、shared memory、树形归约。
3. softmax：数值稳定性、online softmax。
4. tiled GEMM：shared memory tiling、register blocking、边界处理。

验收：

- 能解释为什么 naive GEMM 慢，tiled GEMM 快在哪。
- GEMM 在 1024×1024×1024 上达到 cuBLAS 的 30%–50%（消费级卡，不要拿 B200 的数字要求自己）。
- 能看懂 `ncu` 的 DRAM throughput、occupancy、warp stall 三个指标（WSL 不行就先用 `nsys`）。

### Week 4–5：TileLang 入门

读：

- `deepinfer/model_executor/layers/tilelang_ops/cuda/norm.py`（298 行）——最适合入门的真实算子。
- `.../cuda/gather.py`——理解 index 型 kernel。
- `.../cuda/store_kv_cache.py`（346 行）——理解 block table / slot mapping。
- `.../cuda/state_cache.py`——理解 circular buffer。
- `.../ops.py`——理解 backend dispatch（`cuda` vs `asc`）。
- `.../common.py`——量化常数、采样数学。

学：

- `@tilelang.jit`、`out_idx`、`target`、`pass_configs`。
- `T.Kernel`、`T.Parallel`、`T.copy`、`T.alloc_shared`、`T.Pipelined`、`T.gemm`。
- `T.dynamic`、`T.StridedTensor`、`T.assume`。
- 线程映射、vectorization、layout、tail handling。

写：

1. TileLang vector add。
2. TileLang RMSNorm（支持 residual、bf16/fp16）。
3. TileLang paged gather（给定 block table 和 slot mapping）。

验收：

- 三个 kernel 与 PyTorch 参考实现误差在 bf16 容差内。
- RMSNorm 达到 HBM 带宽的 50% 以上（用 `bytes / time` 算）。
- 能解释 `T.Parallel` 生成的线程映射和 `blk_d`/`vec_size` 的关系。

> 提示：不要一上来读 `fused_transform.py`（1238 行）。它是多个算子的融合，等你能独立写 norm + rope + store KV 之后再回头看。

### Week 6：量化基础

读：

- `tilelang_ops/common.py` 的 FP8/FP4 常数和 `customized_finfo_amax`。
- `tilelang_ops/cuda/quant.py` 的 `ue8m0_exp_scale`。
- `tilelang_ops/cuda/norm.py` 的 `do_fp8_cast` 分支。
- Q&A 的 “Low-precision computation” 一节：主 KV per-16 e4m3、indexer KV per-32 ue8m0 MXFP4、MoE 权重 per-32 ue8m0 MXFP4。

学：

- FP32/FP16/BF16/FP8 E4M3/E5M2/FP4 E2M1 的位布局、动态范围、精度。
- per-tensor / per-token / per-channel / per-128-group 量化。
- scale 的存储：FP32 scale、ue8m0、packed int32、TMA-aligned col-major。
- 量化误差、amax、clipping、stochastic rounding（知道即可）。

写：

- 实现 per-tensor、per-token、per-128-group 的 FP8 量化+反量化。
- 在随机正态和长尾分布上测 max/mean relative error。
- 写一页笔记：为什么 DeepSeek 用 per-32 ue8m0，而不是 per-tensor FP32 scale？

验收：能解释 MXFP8/MXFP4 的 “block scale” 是什么，为什么能省带宽。

### Week 7：Attention 基础

读：

- FlashAttention 1/2 论文（先读 1，理解 online softmax 和 tiling；再读 2 的改进）。
- `deepinfer/model_executor/layers/attention_backend.py`（135 行）——FlashMLA 的 prefill/decode 封装。
- `deepinfer/model_executor/models/v4_modules/mla_attn.py`（265 行）——MLA 的投影容器。
- `.../index_attn.py`（532 行）——MQA indexer，先看类名和 forward 结构，不要逐行。

写：

- 实现 causal attention 的 naive 版本和 online-softmax 版本（PyTorch 或 Triton 均可）。
- 对比 `torch.nn.functional.scaled_dot_product_attention`，测误差和速度。
- 画一张 attention 的 roofline：prefill（compute-bound）vs decode（memory-bound）。

验收：能解释为什么 decode attention 是 memory-bound，为什么需要 split-KV / FlashDecoding。

### Week 8：复盘 + 第一篇笔记

做：

- 回看前 7 周的代码和笔记，补一张“CUDA/TileLang 速查表”。
- 写博客 #1：《DeepInfer 架构总览：从请求到 token》。
- 更新学习计划：哪些地方超时，哪些地方需要补课。

验收：博客能发给别人看懂；你自己能在 10 分钟内讲完。

---

## 5. Phase 1：算子（第 9–24 周）

### 5.1  kernel 实验的标准流程

每做一个 kernel，都按这 9 步走，形成肌肉记忆：

1. **参考实现**：PyTorch 写清楚语义，覆盖边界（空 tensor、dynamic shape、非连续 stride）。
2. **朴素 kernel**：先正确，不优化。
3. **正确性**：`torch.testing.assert_close`，bf16 用 `rtol/atol`；index 类必须 exact。
4. **Benchmark**：warmup 10 次，计时 100 次，取 median/p90；用 CUDA events，不要用 `time.time()`。
5. **Roofline**：算 Bytes 和 FLOPs，算 arithmetic intensity，判断 memory-bound 还是 compute-bound。
6. **Profile**：`nsys` 看 timeline，`ncu` 看 DRAM/L2/occupancy/stall（能跑的话）。
7. **一次只优化一个瓶颈**：vectorization → layout → shared memory → pipeline → occupancy。
8. **写一页笔记**：问题、假设、数据、瓶颈、下一步。
9. **加测试**：放进 `labs/` 的 pytest，防止回归。

### 5.2 Track A（第 9–12 周）：Memory-bound 融合算子

**为什么先学这个**：面试最常手写 layernorm/softmax/rope；工业界最常写融合算子；你的 8GB 卡最容易跑出接近带宽上限的结果。

读（按顺序）：

- `deepinfer/model_executor/layers/layernorm.py`（RMSNorm 封装）
- `deepinfer/model_executor/layers/rotary_embedding.py`（RoPE/YaRN）
- `tilelang_ops/cuda/norm.py`（298 行）
- `tilelang_ops/cuda/fused_transform.py`（570 行）——只读 wrapper 和 kernel 签名，理解融合了哪些步骤
- `tilelang_ops/fused_transform.py`（1238 行）——最后读，重点是 `fused_norm_rope_hadamard_cast_store_kvcache` 的参数和调用路径
- `tilelang_ops/cuda/quant.py`（FP8 scale）
- `tilelang_ops/cuda/outlier_detection.py`（可选）

核心问题：

- 一个 token 的 hidden 是 4096/7168，怎么切 block？一个 CTA 处理几行？
- 什么时候用 `T.Parallel`，什么时候手写 thread mapping？
- 怎么保证 128-bit vectorized load/store？tail 怎么处理？
- residual + RMSNorm + FP8 quant + RoPE + KV store 能不能一次读完/写完？
- 融合的收益来自哪里：省 HBM 往返、省 launch、还是省中间 buffer？

Lab：

| Lab | 任务 | 验收 |
| :--- | :--- | :--- |
| A1 | RMSNorm + residual（bf16） | 误差 < 1e-2；≥ 60% HBM 带宽 |
| A2 | RMSNorm + residual + per-128 FP8 quant | 误差在量化容差内；输出 scale 布局正确 |
| A3 | RoPE + paged KV store | 与参考实现 exact（index）；随机 block table 正确 |
| A4 | 把 A1–A3 融成一个 kernel | 端到端比 3 个独立 kernel 快 1.5× 以上，或带宽提升 30% |

加分：

- 支持 dynamic batch/seqlen。
- 支持非连续 stride。
- 用 `T.Pipelined` 做 software pipeline。
- 写一个 autotune 脚本，扫 `block_m`/`threads`/`vec_size`。

### 5.3 Track B（第 13–16 周）：KV cache 与 state cache

**为什么学**：PagedAttention/KV cache 是推理 infra 的第一高频题；DeepInfer 的 state cache/SWA 是它的特色。

读：

- `deepinfer/worker/cache_engine.py`（820 行）——先读 `KVCache`、`StateBlock`、`KVCacheManager`、`GPUCacheInterface`、`CacheEngine` 的类定义和 `__init__`，再读 `select_state_slice_to_store`、`load_state_from_store_to_prefill_cache`。
- `tilelang_ops/cache_common.py`（396 行）——`V41_FORMAT`、布局常量。
- `tilelang_ops/cuda/load_paged_kv_cache.py`（346 行）
- `tilelang_ops/cuda/store_kv_cache.py`（346 行）
- `tilelang_ops/cuda/state_cache.py`——circular buffer 的 select/load。
- `tilelang_ops/cuda/compress.py`（587 行）——压缩 KV 的 prefill/decode 更新。
- `tilelang_ops/cuda/candidate_blocks.py`、`swa_index.py`、`get_kv_index.py`、`kv_index_cat.py`——稀疏/窗口索引。
- `tests/worker/test_state_cache_ops.py`（612 行）——当作可执行规格读。

核心问题：

- block table、slot mapping、page size、KVSTORE_GRAIN 的关系。
- 一个 token 的 KV 字节数怎么算？main KV / indexer K / SWA state 各占多少？
- prefill 和 decode 的 KV 写入路径有什么不同？
- 为什么 SWA 用固定 slot + circular buffer，而不是 paged block？
- 128-token 粒度、6 个额外位置、bounded replay 的边界条件。

Lab：

| Lab | 任务 | 验收 |
| :--- | :--- | :--- |
| B1 | paged KV gather（decode） | 随机 block table/seqlen 下 exact；≥ 50% 带宽 |
| B2 | paged KV scatter（prefill） | 处理跨 block、tail、空序列 |
| B3 | state cache select/load | 环形 buffer 回绕正确；与测试语义一致 |
| B4 | 计算 V4.1 配置的 KV/state 占用 | 写一个脚本，输入 config，输出每 token/每序列字节数 |

加分：

- 实现一个简单的 block manager（allocate/free/prefix share）。
- 模拟不同 page size 对碎片和带宽的影响。
- 读 `test_state_cache_ops.py`，自己加一个边界测试。

### 5.4 Track C（第 17–20 周）：GEMM 与 MoE

**为什么学**：MoE/EP 是 DeepSeek 的核心，也是推理 infra 面试的区分点；GEMM 是所有算子的基础。

读：

- `deepinfer/model_executor/models/v4_modules/mlp.py`（549 行）——`DeepseekMoE`、`process_moe_weight`、`_grouped_gemm`、`mega_moe_forward`。
- `.../moe.py`——`TopKGate`、`FusedExperts`、`ReplicatedSharedExpert`。
- `deepinfer/model_executor/eplb/policy.py`（123 行）——`_batch_greedy_indices`、`rebalance_experts_global`。
- `.../eplb/shuffle.py`——专家权重 shuffle。
- `deepinfer/model_executor/parallel_utils/parallel_state.py`（181 行）、`userbuffer.py`（211 行）。
- vendor `DeepGEMM` / `DeepEP` 的 README 和 tests。
- Q&A 的 “Expert-parallel load balancing” 一节。

核心问题：

- top-k gate 的 scoring（sqrtsoftplus）、noaux_tc、load-correction bias 分别做什么？
- token 怎么 permute 到本地专家？怎么 combine 回来？
- grouped GEMM 和普通 GEMM 的区别？为什么需要 contiguous layout？
- FP8/FP4 的 scale 在 GEMM 的哪个位置乘？per-128-group 怎么对齐？
- EP 的 all-to-all 通信量怎么算？为什么 decode 用 Wide-EP？
- EPLB 的 “logical/physical expert” 和 “redundant expert” 是什么？greedy 策略的优化目标是什么？

Lab：

| Lab | 任务 | 验收 |
| :--- | :--- | :--- |
| C1 | top-k gate + token permutation/combine | 与 PyTorch 参考 exact；支持 capacity factor |
| C2 | grouped GEMM（Triton 或 TileLang） | 与 `torch.bmm` 对比；≥ 50% cuBLAS |
| C3 | FP8 per-128-group GEMM | 误差在容差内；解释 scale 布局 |
| C4 | EPLB 模拟器 | 合成 Zipf 负载下，max/mean 从 1.5 降到 1.1 以下 |

加分：

- 用 `torch.distributed` + gloo 模拟 2/4 rank 的 EP dispatch/combine。
- 读 DeepEP 的 `dispatch`/`combine` 接口，画一张通信图。
- 复现 `_batch_greedy_indices` 的一个简化版，和随机分配对比。

### 5.5 Track D（第 21–24 周）：Sampling 与投机解码

**为什么学**：sampling 是“最后一个 kernel”，投机解码是 decode 吞吐的关键；两者都容易在面试里被问细节。

读：

- `deepinfer/model_executor/layers/sampler.py`（611 行）——先读 `forward`，再读 `_verification`、`_get_so_mask`。
- `tilelang_ops/cuda/sampler.py`（379 行）——`prepare_verification_metadata`、`get_accept_result`、`finalize_verification`。
- `tilelang_ops/cuda/fused_sample.py`（490 行）
- `tilelang_ops/cuda/random_sampling.py`、`sum_smaller_probs.py`
- `tilelang_ops/cuda/structure_output.py`——token bitmask。
- `tilelang_ops/cuda/schedule_mtp_varlen.py`——按 confidence 算变长验证长度。
- `deepinfer/model_executor/models/mtp/dflash.py`（348 行）
- `.../v4_modules/heads.py` 的 `AcceptRatePredictor`
- `deepinfer/structure_output/`（compiler/matcher/mask）

核心问题：

- top-k/top-p/temperature/repetition penalty 的顺序和数值稳定性。
- Gumbel-max 采样为什么等价于按概率采样？
- 投机解码的接受概率 `min(1, p/q)` 怎么推导？拒绝时怎么修正分布？
- 为什么需要变长验证？confidence head 预测什么？
- structured output 的自动机怎么变成 token bitmask？怎么保证不破坏流式输出？
- batch invariant / deterministic sampling 为什么难？哪些操作会引入非确定性？

Lab：

| Lab | 任务 | 验收 |
| :--- | :--- | :--- |
| D1 | top-k/top-p/temperature sampling | 与参考实现 exact（固定 seed） |
| D2 | rejection sampling 验证 | 分布检验（卡方/KS）通过；接受率符合理论 |
| D3 | 投机解码循环（小模型） | 输出分布与 target 一致；测 acceptance/speedup |
| D4 | regex/JSON schema bitmask | 生成结果 100% 符合语法；流式不丢字符 |

加分：

- 实现 deterministic sampling，验证 batch 不变性。
- 用 `structure_output` 的 matcher 读一遍，写一个最小 JSON schema 例子。
- 画一张 speculative decoding 的加速比公式图（acceptance rate vs draft length）。

### 5.6 12 个 Lab 总表

| # | Lab | 对应仓库 | 预计 | 难度 |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 环境 + TileLang smoke test | `cuda/norm.py` | 2–3 天 | ★ |
| 1 | vector add / reduction / softmax / GEMM | CUDA 基础 | 2 周 | ★★ |
| 2 | RMSNorm + residual | `norm.py` | 1 周 | ★★ |
| 3 | RMSNorm + FP8 quant | `norm.py`、`quant.py` | 1 周 | ★★★ |
| 4 | RoPE + paged KV store | `fused_transform.py`、`store_kv_cache.py` | 1 周 | ★★★ |
| 5 | paged KV gather/scatter | `load_paged_kv_cache.py` | 1 周 | ★★★ |
| 6 | state cache select/load | `state_cache.py` | 1 周 | ★★★ |
| 7 | decode attention（online softmax + split-KV） | `attention_backend.py` | 2 周 | ★★★★ |
| 8 | FP8 grouped GEMM | `mlp.py`、DeepGEMM | 2 周 | ★★★★ |
| 9 | top-k gate + permutation | `moe.py` | 1 周 | ★★★ |
| 10 | sampling + rejection verification | `sampler.py` | 1 周 | ★★★ |
| 11 | structured output bitmask | `structure_output/` | 1 周 | ★★★ |
| 12 | EPLB 模拟器 | `eplb/policy.py` | 1 周 | ★★ |

> 0 号是环境准备，不计入 12 个核心 Lab；1–12 才是。不要 12 个全做：标准档做 0–7 + 9–10；冲刺档加 8、11、12。

---

## 6. Phase 2：引擎（第 25–36 周）

目标：**能从零讲清一次请求的生命周期，能写一个 mini engine，能解释 DeepInfer 每个设计选择背后的 trade-off。**

### 6.1 Scheduler 精读（2 周）

读的顺序：

1. `csrc/engine/sequence.hpp`（364 行）：Sequence、SequenceMetadata、status、cache。
2. `csrc/engine/block_manager.hpp`（202 行）：block 分配/释放、prefix sharing。
3. `csrc/engine/outputs.hpp`（273 行）：`SchedulerOutputs` 的字段。
4. `csrc/engine/scheduler.hpp`（1168 行）分块读：
   - `_schedule()`：prefill vs decode 的选择、DP 同步。
   - `_schedule_prefill()`：chunk size、block 分配、preemption。
   - `_schedule_decode()`：running 排序、MTP slot、finished/ignored。
   - `_preempt()`：重算 vs 换出。
   - `update_mtp_runtime()`：投机解码的动态长度。
5. `csrc/engine/input_metadata/prepare_tensor_input_v4.cpp`（588 行）：元数据怎么变成 tensor。
6. `csrc/engine/prepare_sample.cpp`、`pybind_scheduler.cpp`：Python 边界。
7. `deepinfer/engine/llm_engine.py`（872 行）：异步调度、`DeferredRelease`、输出处理。

要回答的问题：

- 为什么 prefill 和 decode 要分开调度？chunked prefill 的 chunk 大小怎么定？
- `max_num_batched_tokens`、`max_num_seqs`、`max_model_len` 三个限制怎么共同作用？
- 什么时候触发 preemption？重算和换出的代价分别是什么？
- DP group 的 all-reduce MIN 在解决什么问题？
- `SchedulerOutputs` 里的 `clone_states`、`clone_blocks`、`zero_states` 是干什么的？
- 异步调度下，`schedule(N+1)` 为什么不需要等 `sample(N)` 的结果？哪些信息是“乐观估计”？

练习：

- 用 Python 写一个离散事件模拟器：请求到达、continuous batching、block 分配、prefill/decode 切换；输出 TTFT/TPOT/throughput。
- 对比 static batching vs continuous batching，画一张吞吐-延迟曲线。

验收：能不看代码写出 scheduler 的伪代码；能解释每个字段的必要性。

### 6.2 KV cache 管理（1 周）

读：

- `deepinfer/worker/cache_engine.py` 的 `KVCacheManager`、`GPUCacheInterface`、`CacheEngine`。
- `deepinfer/worker/base.py` 的 `profile_num_available_blocks`、`init_cache_engine`。
- `tests/worker/test_state_cache_ops.py`。
- Q&A 的 “KV cache storage” 一节。

练习：

- 写一个 KV cache 容量计算器：输入层数、head 数、head_dim、block_size、并发数、序列长度，输出 GB。
- 写一个 prefix cache 模拟器：LRU/LLC，测 hit rate vs capacity。
- 解释 DeepInfer 为什么把 block KV 放 3FS、SWA 放 host memory、state 放 GPU slot。

验收：能解释 “KV cache 命中率每提高 1% 对成本意味着什么”，能算账。

### 6.3 并行与通信（2 周）

读：

- `deepinfer/model_executor/parallel_utils/parallel_state.py`（181 行）：DP/EP/CP/ENP group 的创建。
- `.../userbuffer.py`（211 行）：对称内存、all-gather/reduce-scatter、SM 限制。
- `.../communication_op.py`：broadcast、gather。
- `deepinfer/distributed/shm_broadcast.py`（402 行）：共享内存 ring buffer。
- `deep_cluster/hardware/`、`deep_cluster/topology/bind.py`：NUMA/CPU 绑定。
- `deepinfer/model_executor/eplb/`。
- DeepEP 的 README（dispatch/combine、NVLink/RDMA）。

练习：

- 用 `torch.distributed` + gloo 在单机起 2 个进程，做 all-reduce/all-gather/broadcast；测延迟和带宽。
- 写一个 EP dispatch/combine 的模拟器：token→expert→rank，算通信量。
- 画一张 64 卡 Wide-EP 的 decode 数据流图。

验收：能解释 DP/EP/TP/PP/CP 的区别；能算 all-to-all 的通信量；能解释为什么 decode 用 Wide-EP。

### 6.4 投机解码、异步调度、CUDA Graph（2 周）

读：

- `deepinfer/worker/model_runner.py`（569 行）：`execute_model`、`sample`、CUDA Graph 选择。
- `deepinfer/worker/cuda_graph_runner.py`：capture/replay、batch size 对齐。
- `deepinfer/worker/dflash_cuda_graph_runner.py`。
- `csrc/engine/zos.cpp`（281 行）、`deepinfer/worker/zos_status.py`：零开销调度。
- `deepinfer/model_executor/models/mtp/dflash.py`（348 行）。
- `deepinfer/model_executor/layers/sampler.py` 的 `_verification`。
- Q&A 的 “Speculative decoding” 和 “Asynchronous scheduling”。

练习：

- 画一张 decode step 的 CPU/GPU 时间线：schedule → forward → sample → post-process；标出同步点和气泡。
- 写一个 speculative decoding 的模拟器：给定 acceptance rate、draft length、draft/target 成本比，画 speedup 曲线。
- 用 PyTorch 实现一个 mini CUDA Graph（capture 一个简单的 MLP），比较 eager vs graph 的 launch 开销。

验收：能解释 “异步调度为什么能消除气泡”“CUDA Graph 解决什么、不解决什么”。

### 6.5 Mini engine（3 周）

目标：用 Qwen2.5-0.5B 或 TinyLlama-1.1B，写一个能跑的 mini engine：

- M1：naive `generate`，逐 token 跑。
- M2：paged KV cache + continuous batching。
- M3：chunked prefill + prefix cache。
- M4：sampling（top-k/top-p/temperature）。
- M5：speculative decoding（draft 用同一个模型的小变体，或 n-gram）。
- M6：benchmark + 报告。

验收：

- 与 HF `generate` 的输出在贪心解码下一致（或误差可解释）。
- 吞吐比 HF `generate` 高 2× 以上（continuous batching 的功劳）。
- 能画出 TTFT/TPOT/throughput 随并发变化的曲线。
- 代码有 README、测试、benchmark 脚本。

### 6.6 可选：VE 与 structured output（1 周）

- `deepinfer/ve/`：多模态预处理、image processor、ViT。
- `deepinfer/structure_output/`：compiler/matcher/mask。
- `deepinfer/model_executor/models/vision_encoder/`。
- 只有投多模态推理或想做端到端 demo 时再深入。

---

## 7. Phase 3：项目与求职（第 37–46 周）

### 7.1 三个项目

#### 项目 A：mini inference engine（必做）

**一句话**：用 PyTorch 从零实现一个支持 paged KV、continuous batching、chunked prefill、prefix cache、speculative decoding 的推理引擎。

| 里程碑 | 内容 | 验收 |
| :--- | :--- | :--- |
| M1 | 单请求 generate | 与 HF 贪心输出一致 |
| M2 | paged KV + batching | 并发 8 请求，吞吐 ≥ 2× HF |
| M3 | chunked prefill + prefix cache | 长 prompt TTFT 下降；cache hit 可观测 |
| M4 | sampling + spec decode | 分布正确；acceptance/speedup 可测 |
| M5 | benchmark + 文档 | 一张吞吐-延迟曲线；一篇设计文档 |

**简历写法（填数字）：**

> 从零实现 PyTorch 推理引擎，支持 PagedAttention、continuous batching、chunked prefill 与 prefix cache；在 Qwen2.5-0.5B 上相比 HF `generate` 吞吐提升 X%，4K 上下文 TTFT 降低 Y%；实现投机解码，acceptance rate Z%，端到端加速 W×。

#### 项目 B：kernel zoo（必做）

**一句话**：8–10 个手写 kernel，每个都有正确性测试、benchmark、roofline 和 profile 笔记。

建议清单：

1. RMSNorm + residual
2. RMSNorm + FP8 per-128 quant
3. RoPE + paged KV store
4. paged KV gather/scatter
5. state cache select/load
6. decode attention（online softmax + split-KV）
7. top-k/top-p sampling
8. rejection sampling verification
9. FP8 grouped GEMM
10. structured output bitmask

**README 必须有的表：**

| Kernel | Shape | PyTorch (ms) | Mine (ms) | Speedup | % Peak BW/FLOPs | Bottleneck |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |

**简历写法：**

> 实现 10 个 TileLang/CUDA 算子，覆盖 norm/rope/KV cache/attention/sampling/GEMM；RMSNorm 达到 HBM 带宽的 X%，decode attention 相比 PyTorch SDPA 加速 Y×；用 ncu 定位 Z 瓶颈并优化。

#### 项目 C：EPLB / MoE 模拟器（选做，但很加分）

**一句话**：用合成负载和真实 trace 模拟 MoE 专家负载，比较 random / round-robin / greedy / DeepInfer `_batch_greedy_indices`，输出 imbalance、吞吐、通信量。

| 里程碑 | 内容 |
| :--- | :--- |
| M1 | 合成 Zipf 负载，专家数/EP size 可配 |
| M2 | 实现 3 种分配策略 |
| M3 | 加通信成本模型（all-to-all） |
| M4 | 画图 + 报告：imbalance vs throughput |

**简历写法：**

> 构建 MoE EPLB 模拟器，复现 DeepInfer 的 greedy 重排策略；在 Zipf 负载下将 max/mean 从 X 降到 Y，估算端到端吞吐提升 Z%。

#### 项目 D（冲刺）：开源贡献

- 给 DeepInfer 提文档/测试/小 bug fix（先从 `tests/`、`doc/`、`examples/` 开始）。
- 给 vLLM/SGLang/TileLang 提 kernel 优化或 bug fix。
- 目标：1 个 merged PR；面试时能讲清“我改了什么、为什么、怎么验证”。

### 7.2 博客/笔记体系

建议 3 篇长文 + 每周短笔记：

1. 《DeepInfer 架构总览：从请求到 token》——Phase 0 结束写。
2. 《从 DeepInfer 看推理引擎的 KV cache 与调度》——Phase 2 结束写。
3. 《我如何手写一个 decode attention kernel》——Track B/D 结束写。

每篇结构：

- 问题是什么？
- 朴素方案为什么不行？
- DeepInfer 怎么做？
- 我复现了什么？数据是多少？
- 还有什么没解决？

### 7.3 面试准备

#### 高频八股（按主题）

| 主题 | 必须能答 | 仓库对应 |
| :--- | :--- | :--- |
| CUDA 执行模型 | grid/block/warp/thread、occupancy、warp divergence | 基础 |
| 内存层次 | global/shared/register/L2、coalescing、bank conflict | 基础 |
| GEMM | tiling、register blocking、split-K、epilogue、tensor core | `mlp.py`、DeepGEMM |
| Attention | online softmax、FlashAttention、decode split-KV、MLA | `attention_backend.py` |
| 量化 | FP8/FP4、per-tensor/per-token/per-group、scale 布局 | `quant.py`、Q&A |
| KV cache | PagedAttention、block table、prefix cache、eviction | `cache_engine.py` |
| 调度 | continuous batching、chunked prefill、preemption、SLO | `scheduler.hpp` |
| 并行 | DP/TP/PP/EP/CP、all-reduce/all-to-all、Wide-EP | `parallel_state.py` |
| MoE | top-k gate、grouped GEMM、EPLB、冗余专家 | `moe.py`、`eplb/` |
| 投机解码 | draft/target、acceptance、verification、变长 | `dflash.py`、`sampler.py` |
| 推理系统 | TTFT/TPOT/goodput、disaggregation、3FS | Q&A |
| 性能分析 | roofline、MFU、nsys/ncu、bottleneck 定位 | `benchmark.py` |

#### 手撕清单

- reduction（warp shuffle）
- softmax（数值稳定 + online）
- layernorm / rmsnorm
- tiled GEMM
- causal attention
- top-k / top-p sampling
- LRU cache
- ring all-reduce（概念 + 伪代码）
- 二叉树/链表/DP 常规题（保持手感，每周 3 题）

#### 项目讲述模板

1. **背景**：为什么做？解决什么瓶颈？
2. **方案**：整体架构、关键数据结构、为什么选它。
3. **实现**：你写了哪部分？最难的技术点是什么？
4. **数据**：吞吐/延迟/带宽/加速比；怎么测的？基线是什么？
5. **trade-off**：还有什么方案？为什么没选？如果给你 10× 资源会怎么改？
6. **反思**：哪里做得不好？如果重来怎么做？

#### 模拟面试

- 从第 6 个月开始，每 2 周一次，找人互面或自录。
- 每次 45 分钟：10 分钟项目 + 20 分钟八股/手撕 + 15 分钟系统设计。
- 记录问题，下一周补洞。

### 7.4 投递时间线（按 2027 秋招）

| 时间 | 动作 |
| :--- | :--- |
| 2026.11–2027.02 | 简历 1.0；投暑期实习（算子/推理加速/高性能计算）；目标 1–2 个 offer |
| 2027.03–06 | 实习；把实习工作做成项目 A/B 的一部分；继续 kernel zoo |
| 2027.06–07 | 简历 2.0；秋招提前批；刷题 + 八股 + 模拟面试 |
| 2027.07–09 | 正式批；每周复盘；记录每家公司的面试题 |
| 2027.09–10 | 补录/谈薪；保持开源和博客更新 |

> 具体时间以你学校/目标公司的校招公告为准；提前批通常比正式批更重要。

---

## 8. 每周节奏与学习方法

### 8.1 一周模板

| 时间 | 内容 | 时长 |
| :--- | :--- | :--- |
| 周一–周四晚 | 读代码/论文 1h + 写代码 1h | 8h |
| 周六 | 大块实验：kernel/项目 | 4h |
| 周日 | 补实验 + 写笔记 + 复盘 | 3h |
| 周日晚 | 规划下周 | 0.5h |

### 8.2 笔记模板

```markdown
# Week N

## 本周问题
（用一句话写清楚要解决什么）

## 读了什么
- 文件/论文：
- 关键结论：

## 写了什么
- 代码：
- 测试：

## 数据
| 实验 | 基线 | 我的 | 提升 |
| :--- | ---: | ---: | ---: |

## 没搞懂
- 

## 下周
- 
```

### 8.3 代码阅读方法

1. **先读测试**：`tests/` 是可执行的规格说明；比注释可靠。
2. **先读调用方**：从 `llm_engine.py` / `model_runner.py` 往下追，不要从最底层往上猜。
3. **用 `rg` 追符号**：`rg -n "class Scheduler" csrc`、`rg -n "fused_norm_rope" deepinfer`。
4. **画调用图**：纸笔或 Mermaid；每读一个模块，补一张图。
5. **加断言/打印**：能跑的小复现里插 `print`/`assert`；不能跑就写伪代码。
6. **读 git history**：`git log -p -- <file>` 看某个设计是什么时候、为什么加的。
7. **合上书写总结**：200 字，不看代码；写不出来就是没懂。

### 8.4 如何用 AI（包括我）

- 让 AI 解释代码结构、画调用图、找相关文件、生成测试骨架。
- 不要让 AI 直接写完整 kernel；先自己写，再让 AI review。
- 对 AI 的结论要验证：跑测试、看源码、对数据。
- 面试前关掉 AI，手写一遍。

### 8.5 防止“看了等于会了”

- 每个模块至少有一个“可运行产物”：kernel、测试、模拟器、benchmark。
- 每周日做一次“闭卷复述”：白板/纸笔讲 5 分钟。
- 每月做一次“从零重写”：选一个 kernel，不看旧代码重写。
- 每季度做一次“模拟面试”：讲项目 + 八股 + 手撕。

---

## 9. 资源清单

### 9.1 课程与书

| 资源 | 用途 | 什么时候 |
| :--- | :--- | :--- |
| PMPP（Programming Massively Parallel Processors） | CUDA 系统教材 | Week 2–3 |
| CUDA C++ Programming Guide / Best Practices | 查漏补缺 | 全程 |
| GPU MODE 课程/lectures | 现代 GPU kernel 优化 | Week 3–8 |
| Stanford CS149 / CMU 15-418 | 并行计算体系 | 可选 |
| MIT 6.172 | 性能工程 | 可选 |
| CUTLASS docs | GEMM/tensor core | Track C |
| Triton tutorials | 快速写 kernel | Week 4–5 |
| TileLang docs/examples | 本仓库算子 | Week 4–5 |

### 9.2 论文（按顺序）

1. *Attention Is All You Need*（基础）
2. *FlashAttention* / *FlashAttention-2*（online softmax、tiling）
3. *DeepSeek-V2*（MLA）
4. *DeepSeek-V3*（MoE、FP8、EP）
5. *PagedAttention / vLLM*（KV cache 分页）
6. *Orca*（continuous batching）
7. *Sarathi-Serve*（chunked prefill）
8. *DeepSeek-V3/R1 推理系统*（EP/EPLB，知乎博客）
9. *DeepEP* / *DeepGEMM* / *FlashMLA* README（工程实现）
10. *Speculative Decoding*（Leviathan et al.）/ *Medusa* / *EAGLE*（投机解码）
11. *DistServe* / *Splitwise* / *Mooncake*（P/D 分离）
12. *3FS*（分布式文件系统）
13. *YoCo*（如果你对 SWA/长上下文感兴趣）

读论文的方法：先读 abstract/intro/figures，回答“解决什么问题、核心 idea、代价是什么”，再决定要不要读实现细节。

### 9.3 仓库

| 仓库 | 用途 |
| :--- | :--- |
| vLLM | 推理引擎的事实标准；读 scheduler、KV cache、attention backend |
| SGLang | RadixAttention、前缀缓存、调度 |
| TensorRT-LLM | 工业级 kernel/engine |
| llama.cpp | 轻量推理、量化 |
| CUTLASS | GEMM/tensor core 模板 |
| Triton | 高层 kernel DSL |
| TileLang | 本仓库算子 DSL |
| DeepGEMM | FP8/FP4 GEMM |
| FlashMLA | MLA attention |
| DeepEP | EP 通信 |
| DeepSelect | 稀疏选择 |
| 3FS | 分布式存储 |
| nccl-tests | 通信性能测试 |

### 9.4 工具

- `nsys` / `ncu`：timeline / kernel 计数器。
- `torch.profiler`：Python 侧 + CUDA。
- `nvidia-smi dmon`、`gpustat`：GPU 利用率/显存。
- `py-spy`：Python 进程采样。
- `perf`：CPU 侧。
- `rg`、`fd`、`bat`：代码搜索。
- `clang-format`、`ruff`：格式化（仓库已配置）。
- `pytest`：`-m ci` / `-m ci_npu` / `-m benchmark`。

### 9.5 社区

- GPU MODE（Discord/lectures/kernel competitions）
- DeepSeek GitHub issues / discussions
- vLLM Slack / GitHub
- 知乎、公众号（DeepSeek、MLSys、GPU 优化）
- 校招群/学长学姐（获取时间线）

---

## 10. 风险与避坑

1. **不要在本地追完整构建**：Linux 头文件、GCC 13、CUDA 13、SM100，四个条件你都不满足。构建失败不是你的问题，别耗一周。
2. **不要下载完整 checkpoint**：V4.1-Flash 的权重远超你的显存和内存；用 0.5B/1.1B 模型做实验。
3. **不要把 SM120 当 SM100**：tcgen05、TMA、cluster、FP4 的很多特性不一样；性能数字不可直接类比。
4. **不要用 `torch==2.13.0+cu132`**：你的驱动 577.05 只到 CUDA 12.9；本地用 cu12x。
5. **不要忽略 vendor 子模块**：DeepGEMM/FlashMLA/DeepEP 是理解算子的关键；先 `git submodule update`。
6. **不要从大文件开始**：`fused_transform.py` 1238 行、`scheduler.hpp` 1168 行、`sampler.py` 611 行；先读小文件、测试、调用方。
7. **不要平均用力**：Engram、HC/MHC、YoCo 的模型细节是 C 级；先把 S 级吃透。
8. **不要只做 kernel 不懂引擎**：算子岗也会问 KV cache、continuous batching、PagedAttention；反过来，引擎岗也会问你写过什么 kernel。
9. **不要过早分布式**：单卡都没写明白，不要上 8 卡；先单卡 kernel，再 2 进程通信，再模拟 EP。
10. **不要追 Ascend**：没有硬件就只读代码；投华为/昇腾时再深入 `asc/`。
11. **不要 tutorial hell**：看 10 个视频不如写 1 个 kernel；每学一个概念，必须有一个可运行产物。
12. **不要被终端编码吓到**：仓库里的中文注释是 UTF-8；Windows PowerShell 的默认代码页可能显示成乱码，用 VS Code 打开即可。
13. **不要熬夜硬扛**：11 个月是马拉松；每周留一天休息，保护眼睛和腰。

---

## 11. 90 天行动清单

### 第 1 周：地图 + 环境

- [ ] 读 `README.md`、`doc/inference_deployment_qa.md`、`examples/README.md`
- [ ] `git submodule update --init --recursive`
- [ ] WSL 建 Python 3.12 环境，装 torch cu12x + tilelang
- [ ] 跑通 TileLang vector add
- [ ] 画请求生命周期图
- [ ] 建 `notes/`、`labs/`、`benchmarks/`
- [ ] 写 `notes/00_overview.md`（300 字）

### 第 2–3 周：CUDA

- [ ] vector add
- [ ] reduction（warp shuffle）
- [ ] softmax（online）
- [ ] tiled GEMM
- [ ] 每个 kernel 写 benchmark + 一页笔记
- [ ] 学 `nsys`/`torch.profiler`

### 第 4–5 周：TileLang

- [ ] 读 `cuda/norm.py`、`cuda/gather.py`、`cuda/store_kv_cache.py`
- [ ] TileLang vector add
- [ ] TileLang RMSNorm
- [ ] TileLang paged gather
- [ ] 写 `notes/02_tilelang.md`

### 第 6 周：量化

- [ ] 读 `common.py`、`cuda/quant.py`、Q&A 低精度一节
- [ ] 实现 per-tensor/per-token/per-128 FP8 量化
- [ ] 测误差
- [ ] 写 `notes/03_quant.md`

### 第 7 周：Attention

- [ ] 读 FlashAttention 1/2
- [ ] 读 `attention_backend.py`、`mla_attn.py`
- [ ] 实现 causal attention + online softmax
- [ ] 对比 SDPA
- [ ] 画 roofline

### 第 8 周：复盘

- [ ] 写博客 #1《DeepInfer 架构总览》
- [ ] 自评：能白板讲生命周期吗？
- [ ] 调整下一阶段计划

### 第 9–12 周：Track A

- [ ] A1 RMSNorm + residual
- [ ] A2 RMSNorm + FP8 quant
- [ ] A3 RoPE + paged KV store
- [ ] A4 融合 kernel
- [ ] 每个 kernel 有测试 + benchmark + 笔记
- [ ] 更新 kernel zoo README

### 第 13 周：复盘 + 下一阶段

- [ ] 回看 12 周笔记
- [ ] 选 1 个 kernel 从零重写
- [ ] 规划 Track B/C/D
- [ ] 更新简历草稿

---

## 12. 毕业标准（Self-assessment rubric）

| 维度 | L1 | L2 | L3 | L4 |
| :--- | :--- | :--- | :--- | :--- |
| 算子 | 能写 naive kernel | 能优化 memory-bound kernel | 能写 GEMM/attention，能读 ncu | 能设计融合算子，达到接近峰值 |
| 引擎 | 知道请求生命周期 | 能解释 scheduler/KV cache | 能写 mini engine | 能按 SLO 设计多卡方案 |
| 系统 | 知道 TP/EP | 能解释通信/并行 | 能模拟/测量通信 | 能设计大规模部署 |
| 表达 | 能写笔记 | 能写博客 | 能讲清 trade-off | 能开源/带人 |

**2027 年 6 月目标：**

- 算子 L3：至少 8 个 kernel，有正确性/性能/roofline。
- 引擎 L2–L3：mini engine 能跑，能白板讲 scheduler。
- 系统 L2：能解释 EP/EPLB、all-to-all、P/D 分离。
- 表达 L3：3 篇博客，1 次公开分享。

**Ready checklist：**

- [ ] 能 10 分钟白板讲清 DeepInfer 一次请求的生命周期。
- [ ] 能 20 分钟讲清 PagedAttention + continuous batching + chunked prefill。
- [ ] 能手写 RMSNorm、softmax、tiled GEMM、causal attention。
- [ ] 能解释 FP8/FP4 的 scale 布局和误差来源。
- [ ] 能解释 MoE 的 top-k gate、grouped GEMM、EP、EPLB。
- [ ] 能解释投机解码的接受/拒绝和变长验证。
- [ ] 能用 nsys/ncu 定位一个 kernel 的瓶颈。
- [ ] 有 1 个 merged PR。
- [ ] 有 3 篇能拿得出手的博客。
- [ ] 模拟面试 5 次以上，项目讲述不卡壳。

---

## 附录 A：关键文件索引（按主题）

| 主题 | 文件 | 行数（约） | 读法 |
| :--- | :--- | ---: | :--- |
| 入口 | `deepinfer/entrypoints/llm.py` | 173 | 通读 |
| 引擎 | `deepinfer/engine/llm_engine.py` | 872 | 先读类/方法签名，再读 generate 路径 |
| 调度 | `csrc/engine/scheduler.hpp` | 1168 | 分 `_schedule` / `_schedule_prefill` / `_schedule_decode` 读 |
| 序列 | `csrc/engine/sequence.hpp` | 364 | 通读 |
| Block | `csrc/engine/block_manager.hpp` | 202 | 通读 |
| 输出 | `csrc/engine/outputs.hpp` | 273 | 通读 |
| ZOS | `csrc/engine/zos.cpp` | 281 | 通读 |
| 元数据 | `csrc/engine/input_metadata/prepare_tensor_input_v4.cpp` | 588 | 先看函数名，再看一个分支 |
| 采样 C++ | `csrc/engine/prepare_sample.cpp` | — | 配合 `sampler.py` |
| Worker | `deepinfer/worker/model_runner.py` | 569 | 通读 |
| KV cache | `deepinfer/worker/cache_engine.py` | 820 | 先类定义，再关键方法 |
| CUDA Graph | `deepinfer/worker/cuda_graph_runner.py` | — | 通读 |
| 模型 | `deepinfer/model_executor/models/deepseek.py` | 453 | 通读 |
| Decoder | `deepinfer/model_executor/models/decoder_layer.py` | 209 | 通读 |
| Attention | `deepinfer/model_executor/models/v4_modules/attn.py` | 916 | 先 `V4AttentionBase`，再 `V4Attention` |
| MLA | `.../v4_modules/mla_attn.py` | 265 | 通读 |
| Indexer | `.../v4_modules/index_attn.py` | 532 | 先类/forward，再细节 |
| MoE | `.../v4_modules/mlp.py` | 549 | 先 `DeepseekMoE.forward` |
| MoE 容器 | `.../v4_modules/moe.py` | — | 通读 |
| EPLB | `deepinfer/model_executor/eplb/policy.py` | 123 | 通读 |
| 并行 | `.../parallel_utils/parallel_state.py` | 181 | 通读 |
| Userbuffer | `.../parallel_utils/userbuffer.py` | 211 | 通读 |
| 共享内存 | `deepinfer/distributed/shm_broadcast.py` | 402 | 先 `MessageQueue` |
| Norm | `.../tilelang_ops/cuda/norm.py` | 298 | 精读 |
| Fused transform | `.../tilelang_ops/cuda/fused_transform.py` | 570 | 先签名，再 kernel |
| KV load | `.../tilelang_ops/cuda/load_paged_kv_cache.py` | 346 | 精读 |
| KV store | `.../tilelang_ops/cuda/store_kv_cache.py` | 346 | 精读 |
| State cache | `.../tilelang_ops/cuda/state_cache.py` | — | 精读 |
| Sampler | `deepinfer/model_executor/layers/sampler.py` | 611 | 先 `forward`，再 `_verification` |
| 采样 kernel | `.../tilelang_ops/cuda/sampler.py` | 379 | 精读 |
| Fused sample | `.../tilelang_ops/cuda/fused_sample.py` | 490 | 精读 |
| 投机解码 | `.../models/mtp/dflash.py` | 348 | 通读 |
| Benchmark | `examples/benchmark.py` | 517 | 读指标定义 |
| 规格 | `tests/worker/test_state_cache_ops.py` | 612 | 当 spec 读 |

## 附录 B：公式速查

**KV cache 大小（每 token）：**

```
bytes_per_token = 2 × n_layers × n_kv_heads × head_dim × bytes_per_elem
```

MLA 压缩后把 `n_kv_heads × head_dim` 换成 `kv_lora_rank + qk_rope_head_dim`，再乘上 scale 的字节数。用 checkpoint 的 `config.json` 代入，不要背数字。

**Roofline：**

```
P ≤ min(π, β × I)
I = FLOPs / Bytes
```

- `π`：峰值算力（FLOP/s）
- `β`：峰值带宽（Byte/s）
- `I`：算术强度（FLOP/Byte）

**GEMM 算术强度：**

```
I ≈ 2MNK / ((MK + KN + MN) × bytes_per_elem)
```

**MFU：**

```
MFU = 实际 FLOP/s / 峰值 FLOP/s
```

**投机解码（简化）：**

```
E[accepted] = (1 - α^(γ+1)) / (1 - α)
speedup ≈ E[accepted] / (1 + γ × c)
```

- `α`：acceptance rate
- `γ`：draft length
- `c`：draft 成本 / target 成本

**All-reduce 通信量（ring）：**

```
2 × (N - 1) / N × S
```

`S` 是数据大小，`N` 是 rank 数。

**EP 负载不均衡：**

```
imbalance = max_rank_load / mean_rank_load
```

**Little's Law：**

```
concurrency = throughput × latency
```

## 附录 C：常用命令

```bash
# 代码搜索
rg -n "class Scheduler" csrc
rg -n "fused_norm_rope" deepinfer
rg -n "def _schedule" csrc/engine/scheduler.hpp

# 测试（需要 GPU/模型）
python -m pytest -v -s -m ci tests
python -m pytest -v -s -m ci_npu tests
python -m pytest -v -s tests

# 格式化
ruff check --fix && ruff format
bash scripts/format_csrc.sh

# benchmark
python examples/benchmark.py --help
python examples/benchmark.py --decode_num_requests 1
PROFILE=1 python examples/benchmark.py

# profile
nsys profile -o trace python your_kernel.py
ncu --set full -o report python your_kernel.py

# GPU 状态
nvidia-smi
nvidia-smi dmon
```

## 附录 D：如果你其实是 2028 年秋招（两年版）

如果“28 秋招”指 2028 年秋天，你有约 23 个月，可以把计划拉长：

| 时间 | 重点 |
| :--- | :--- |
| 2026.10–2027.06 | Phase 0 + Phase 1 前半（地基 + memory-bound/KV） |
| 2027.07–2027.12 | Phase 1 后半（GEMM/MoE/sampling）+ 项目 B |
| 2028.01–2028.06 | Phase 2（引擎）+ 项目 A/C + 实习 |
| 2028.07–2028.09 | Phase 3（面试）+ 高级专题 |

多出来的时间建议加：

- **CUTLASS/CUDA 深入**：手写 tensor core GEMM、TMA、pipeline。
- **编译器**：TVM/TileLang 的 lowering、schedule、autotune。
- **分布式**：NCCL/DeepEP 源码、2–8 卡实验、通信模拟。
- **Ascend**：CANN、Ascend C、`asc/` 算子移植（如果有硬件）。
- **论文复现**：选一篇 MLSys/OSDI/NSDI 的推理系统论文，做小规模复现。
- **开源**：在 vLLM/SGLang/TileLang 里做 2–3 个有分量的 PR。

## 附录 E：第一周每天做什么

| Day | 任务 | 产出 |
| :--- | :--- | :--- |
| Day 1 | 读 README + Q&A 前半；建目录 | `notes/00_overview.md` 草稿 |
| Day 2 | 读 Q&A 后半 + examples/README；画架构图 | 架构图 v1 |
| Day 3 | WSL 环境：Python 3.12、torch、tilelang | `python -c "import torch"` 成功 |
| Day 4 | 跑通 TileLang vector add；学 `@tilelang.jit` | `labs/lab00_add.py` |
| Day 5 | 读 `cuda/norm.py`；写 RMSNorm 参考实现 | `labs/lab01_rmsnorm_ref.py` |
| Day 6 | 写 TileLang RMSNorm；测正确性 | `labs/lab01_rmsnorm.py` |
| Day 7 | 写笔记 + 复盘；规划第 2 周 | `notes/week01.md` |

---

## 最后：一句话

**把 DeepInfer 当成“工业级代码地图”，把每个机制拆成一个你能在 8GB 显卡上跑起来的小实验；11 个月后，你带走的不是“读过 5 万行代码”，而是 12 个 Lab、3 个项目、1 个 PR，以及一套能迁移到任何推理系统的分析方法。**
