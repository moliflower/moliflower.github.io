/* DeepInfer 101 — course data.
 *
 * This file is the single source of truth for the local course site.
 * Edit it to add weeks, links, textbook chapters, or resources.
 * No build step: open index.html directly.
 */

window.COURSE = {
  meta: {
    title: "DeepInfer 101",
    subtitle: "从算子到推理系统",
    term: "2026 秋 · 自学节奏",
    startDate: "2026-10-26",
    description:
      "从第一行 Python 开始，用图解、推导、手算与分步实验理解推理系统。已发布的第 1–6 周重新整理为自学教材，暂不新增后续周次。",
    release: "起步 + Unit1 + Unit2 第 5–6 周已发布",
    tagline: "先写对，再写快，再讲清楚。",
    repo: "https://github.com/deepseek-ai/DeepInfer",
  },

  nav: [
    { id: "home", label: "首页", icon: "⌂" },
    { id: "calendar", label: "周历", icon: "▦" },
    { id: "practice", label: "练习中心", icon: "✓" },
    { id: "syllabus", label: "课程大纲", icon: "§" },
    { id: "textbook", label: "教材", icon: "▤" },
    { id: "resources", label: "资源", icon: "⊕" },
    { id: "videos", label: "视频", icon: "▷" },
    { id: "sections", label: "讨论课", icon: "◎" },
    { id: "career", label: "Infra 秋招", icon: "◇" },
    { id: "staff", label: "关于", icon: "☺" },
    { id: "contact", label: "联系", icon: "✉" },
  ],

  announcements: [
    {
      date: "2026-10-24",
      title: "教材重整：先把第 1–6 周真正学懂",
      body: "逐章重组动机、图解、推导、完整案例、渐进练习和自测；统一先修与命令入口。保留原题号与评分，不重置已有学习进度。",
      link: "notes/learning-path.html",
    },
    {
      date: "2026-10-24",
      title: "第 6 周发布：逐迭代调度与延迟指标",
      body: "新增 L11/L12、Lab5、HW6、Project2 Phase2 与 18 组可复现实验。区分虚拟 tick 与真实 GPU 时延；保留第 5 周代码和独立评分。",
      link: "notes/lec11.html",
    },
    {
      date: "2026-10-23",
      title: "第 5 周发布：共享块池 → Mini Engine Phase1",
      body: "新增 L9/L10、Lab4、HW5 与 Project2 Phase1，含单题预测、提示与测试；Unit1 评分保持 100 分。练习中心集中展示已发布任务。",
      link: "notes/lec09.html",
    },
    {
      date: "2026-10-22",
      title: "零基础入口与 Unit1 质量修订",
      body: "新增 Python 起步四节与 Q0–Q4；修订教材公式、单题反馈、数值/状态测试与项目实验。教材页按先修连接正文和练习。",
      link: "notes/foundations.html",
    },
    {
      date: "2026-10-22",
      title: "第一单元已发布",
      body:
        "已有讲义 1–8、实验 0–3、作业 1/2、项目 1、小测 1/2。已补数值与边界契约；按掌握度推进，Unit2–4 仍为待发布计划。",
      link: "#/calendar",
    },
    {
      date: "2026-10-22",
      title: "课程网站上线",
      body:
        "本地静态网站可直接打开；讲义、预备课、实验和作业都有排版网页，提示与小测答案可折叠。",
      link: "#/syllabus",
    },
    {
      date: "2026-10-22",
      title: "Infra 秋招路线",
      body:
        "15 周课程之后接 46 周长期路线：算子 → 引擎 → 项目 → 面试。见 doc/study_plan_zh.md。",
      link: "../../doc/study_plan_zh.md",
    },
  ],

  weeks: [
    {
      n: 0,
      dates: "正式第 1 周之前 · 按掌握程度推进",
      theme: "入门：Python 起步 → 调试与测试",
      items: [
        { type: "reading", label: "入门上篇", title: "Python 起步：从一行计算到一批请求", link: "notes/foundations.html", key: "foundations-guide" },
        { type: "lab", label: "起步 0", title: "表达式与函数 → Q1", link: "notes/foundations00.html", key: "foundations-q1" },
        { type: "lab", label: "起步 1", title: "条件与整数 → Q2", link: "notes/foundations01.html", key: "foundations-q2" },
        { type: "lab", label: "起步 2", title: "列表与循环 → Q3", link: "notes/foundations02.html", key: "foundations-q3" },
        { type: "lab", label: "起步 3", title: "累计状态与测试 → Q4", link: "notes/foundations03.html", key: "foundations-q4" },
        { type: "reading", label: "入门下篇", title: "调试与测试：找到程序第一次走错的地方", link: "notes/prep.html", key: "prep-guide" },
        { type: "lab", label: "预备 0", title: "运行、变量跟踪与报错", link: "notes/prep00.html", key: "prep-q0" },
        { type: "lab", label: "预备 1", title: "修复最大值：最小复现与错误假设", link: "notes/prep01.html", key: "prep-q1" },
        { type: "lab", label: "预备 2", title: "修复矩阵：副本、索引与累计值", link: "notes/prep02.html", key: "prep-q2" },
        { type: "lab", label: "预备 3", title: "浮点测试与独立向量加法", link: "notes/prep03.html", key: "prep-q3" },
      ],
      right: [
        { type: "reading", title: "Python 官方：错误与异常（中文）", link: "https://docs.python.org/zh-cn/3/tutorial/errors.html" },
        { type: "reading", title: "完成后：第一讲核心", link: "notes/lec01.html" },
      ],
    },
    {
      n: 1,
      dates: "10月26日 – 11月1日",
      theme: "GPU 心智模型",
      items: [
        { type: "lecture", label: "讲义 1", title: "核心：从 Python 建立执行模型", link: "notes/lec01.html", key: "w1-l1" },
        { type: "lecture", label: "讲义 2", title: "访存推导；occupancy 与 CUDA 选读", link: "notes/lec02.html", key: "w1-l2" },
        { type: "lab", label: "实验 0", title: "Q0–Q6：示例、填空、独立算子与修错", link: "notes/lab00.html", key: "w1-lab0" },
        { type: "reading", label: "项目预览", title: "Kernel Zoo：现在只看目标，不要求实现", link: "notes/project1.html", key: "w1-p1" },
      ],
      right: [
        { type: "reading", title: "PMPP 第 1–2 章", link: "https://docs.nvidia.com/cuda/cuda-c-programming-guide/" },
        { type: "video", title: "B站：CUDA 编程基础入门系列", link: "https://www.bilibili.com/video/BV1sM4y1x7of/" },
        { type: "video", title: "YouTube：GPU MODE", link: "https://www.youtube.com/@GPUMODE" },
      ],
    },
    {
      n: 2,
      dates: "11月2日 – 11月8日",
      theme: "Reduction、softmax 与 tiling",
      items: [
        { type: "lecture", label: "讲义 3", title: "Reduction、softmax 与 RMSNorm", link: "notes/lec03.html", key: "w2-l3" },
        { type: "lecture", label: "讲义 4", title: "Tiling、GEMM 与 roofline", link: "notes/lec04.html", key: "w2-l4" },
        { type: "lab", label: "实验 1", title: "Softmax & RMSNorm + traffic", link: "notes/lab01.html", key: "w2-lab1" },
        { type: "quiz", label: "小测 1", title: "GPU 模型、roofline、softmax、索引", link: "notes/quiz01.html", key: "w2-quiz1" },
        { type: "project", label: "项目 1", title: "Checkpoint 1：Phase 1", link: "notes/project1.html", key: "w2-p1c1" },
      ],
      right: [
        { type: "reading", title: "FlashAttention（arXiv）", link: "https://arxiv.org/abs/2205.14135" },
        { type: "video", title: "B站：GPU 并行计算（科大周老师）", link: "https://www.bilibili.com/video/BV1kx411m7Fk/" },
        { type: "video", title: "MIT 6.172（OCW）", link: "https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/" },
      ],
    },
    {
      n: 3,
      dates: "11月9日 – 11月15日",
      theme: "Attention 与 FlashAttention",
      items: [
        { type: "lecture", label: "讲义 5", title: "从零实现 Attention", link: "notes/lec05.html", key: "w3-l5" },
        { type: "lecture", label: "讲义 6", title: "FlashAttention 与 decode attention", link: "notes/lec06.html", key: "w3-l6" },
        { type: "lab", label: "实验 2", title: "从零实现 Attention", link: "notes/lab02.html", key: "w3-lab2" },
        { type: "reading", label: "代价练习", title: "实验 2 Q7/Q8：本周只做 attention 代价", link: "notes/lab02.html", key: null },
        { type: "project", label: "项目 1", title: "Checkpoint 2：Phase 2", link: "notes/project1.html", key: "w3-p1c2" },
      ],
      right: [
        { type: "reading", title: "图解 Transformer", link: "https://jalammar.github.io/illustrated-transformer/" },
        { type: "video", title: "B站：FlashAttention 论文分享", link: "https://www.bilibili.com/video/BV1zs4y1J7tb/" },
        { type: "video", title: "GPU MODE", link: "https://www.youtube.com/@GPUMODE" },
      ],
    },
    {
      n: 4,
      dates: "11月16日 – 11月22日",
      theme: "KV cache 与量化",
      items: [
        { type: "lecture", label: "讲义 7", title: "KV cache 基础", link: "notes/lec07.html", key: "w4-l7" },
        { type: "lecture", label: "讲义 8", title: "量化：整数网格、误差与存储预算", link: "notes/lec08.html", key: "w4-l8" },
        { type: "lab", label: "实验 3", title: "连续 + Paged KV cache", link: "notes/lab03.html", key: "w4-lab3" },
        { type: "quiz", label: "小测 2", title: "Attention、KV cache、量化", link: "notes/quiz02.html", key: "w4-quiz2" },
        { type: "homework", label: "作业 1", title: "Roofline、KV cache 与量化（先修已齐）", link: "notes/hw01.html", key: "w1-hw1" },
        { type: "homework", label: "作业 2", title: "Attention 与 KV 数学（可分两次完成）", link: "notes/hw02.html", key: "w2-hw2" },
        { type: "project", label: "项目 1", title: "最终报告 + 5 分钟讲解", link: "notes/project1.html", key: "w4-p1due" },
      ],
      right: [
        { type: "reading", title: "vLLM：PagedAttention", link: "https://blog.vllm.ai/2023/06/20/vllm.html" },
        { type: "video", title: "B站：PagedAttention 代码详解", link: "https://www.bilibili.com/video/BV14a5r6YEhY/" },
        { type: "reading", title: "DeepSeek 部署 Q&A", link: "../../doc/inference_deployment_qa.md" },
      ],
    },
    {
      n: 5,
      dates: "11月23日 – 11月29日",
      theme: "PagedAttention 与前缀缓存",
      items: [
        { type: "lecture", label: "讲义 9", title: "分页映射与共享块池", link: "notes/lec09.html", key: "w5-l9" },
        { type: "lecture", label: "讲义 10", title: "前缀命中、共享容量与淘汰", link: "notes/lec10.html", key: "w5-l10" },
        { type: "lab", label: "实验 4", title: "有限块池与共享所有权", link: "notes/lab04.html", key: "w5-lab4" },
        { type: "homework", label: "作业 5", title: "前缀命中与容量推导", link: "notes/hw05.html", key: "w5-hw5" },
        { type: "project", label: "项目 2", title: "Phase1：请求准入、等待与释放", link: "notes/project2.html", key: "w5-p2phase1" },
      ],
      right: [
        { type: "reading", title: "PagedAttention（arXiv）", link: "https://arxiv.org/abs/2309.06180" },
        { type: "video", title: "B站：PagedAttention 论文精读", link: "https://www.bilibili.com/video/BV1GWjjzfE1b/" },
      ],
    },
    {
      n: 6,
      dates: "11月30日 – 12月6日",
      theme: "Batching 与调度",
      items: [
        { type: "lecture", label: "讲义 11", title: "Continuous batching", link: "notes/lec11.html", key: "w6-l11" },
        { type: "lecture", label: "讲义 12", title: "Chunked prefill 与延迟指标", link: "notes/lec12.html", key: "w6-l12" },
        { type: "lab", label: "实验 5", title: "Decode 优先与分块预填充", link: "notes/lab05.html", key: "w6-lab5" },
        { type: "homework", label: "作业 6", title: "TTFT、TPOT 与 SLO goodput", link: "notes/hw06.html", key: "w6-hw6" },
        { type: "project", label: "项目 2", title: "Phase2：逐迭代调度与实验", link: "notes/project2-scheduler.html", key: "w6-p2phase2" },
      ],
      right: [
        { type: "reading", title: "Anyscale：continuous batching", link: "https://www.anyscale.com/blog/continuous-batching-llm-inference" },
        { type: "reading", title: "Orca（OSDI'22）", link: "https://www.usenix.org/conference/osdi22/presentation/yu" },
        { type: "video", title: "B站：vLLM 整体架构", link: "https://www.bilibili.com/video/BV1YcywBiEJ1/" },
      ],
    },
    {
      n: 7,
      dates: "12月7日 – 12月13日",
      theme: "Sampling 与投机解码",
      items: [
        { type: "lecture", label: "讲义 13", title: "Top-k / top-p / temperature", link: null, key: null, soon: true },
        { type: "lecture", label: "讲义 14", title: "投机解码", link: null, key: null, soon: true },
        { type: "lab", label: "实验 6", title: "Sampling + rejection", link: null, key: null, soon: true },
        { type: "homework", label: "作业 7", title: "接受率与加速比", link: null, key: null, soon: true },
        { type: "project", label: "项目 2", title: "Checkpoint 2", link: null, key: null, soon: true },
      ],
      right: [
        { type: "reading", title: "Speculative Decoding（arXiv）", link: "https://arxiv.org/abs/2211.17192" },
        { type: "video", title: "B站：投机解码方案详解", link: "https://www.bilibili.com/video/BV1Q5KWzQEhn" },
      ],
    },
    {
      n: 8,
      dates: "12月14日 – 12月20日",
      theme: "推理引擎架构",
      items: [
        { type: "lecture", label: "讲义 15", title: "推理引擎架构", link: null, key: null, soon: true },
        { type: "lecture", label: "讲义 16", title: "异步调度与 CUDA Graph", link: null, key: null, soon: true },
        { type: "lab", label: "实验 7", title: "引擎 trace", link: null, key: null, soon: true },
        { type: "homework", label: "作业 8", title: "引擎设计", link: null, key: null, soon: true },
        { type: "project", label: "项目 2", title: "最终：mini engine", link: null, key: null, soon: true },
      ],
      right: [
        { type: "reading", title: "DeepSeek 部署 Q&A", link: "../../doc/inference_deployment_qa.md" },
        { type: "video", title: "B站：vLLM 整体架构", link: "https://www.bilibili.com/video/BV1YcywBiEJ1/" },
      ],
    },
    {
      n: 9,
      dates: "12月21日 – 12月27日",
      theme: "MoE 与专家并行",
      items: [
        { type: "lecture", label: "讲义 17", title: "MoE：gate、专家、grouped GEMM", link: null, key: null, soon: true },
        { type: "lecture", label: "讲义 18", title: "EP 与 EPLB", link: null, key: null, soon: true },
        { type: "lab", label: "实验 8", title: "Top-k gate / EPLB", link: null, key: null, soon: true },
        { type: "homework", label: "作业 9", title: "MoE 负载均衡", link: null, key: null, soon: true },
        { type: "project", label: "项目 3", title: "EPLB 模拟器", link: null, key: null, soon: true },
      ],
      right: [
        { type: "reading", title: "Hugging Face：MoE 讲解", link: "https://huggingface.co/blog/moe" },
        { type: "reading", title: "DeepSeek-V3（arXiv）", link: "https://arxiv.org/abs/2412.19437" },
        { type: "video", title: "B站：MoE 可视化解读", link: "https://www.bilibili.com/video/BV1Gj9ZYdE4N/" },
      ],
    },
    {
      n: 10,
      dates: "12月28日 – 1月3日",
      theme: "并行与通信",
      items: [
        { type: "lecture", label: "讲义 19", title: "DP/TP/PP/EP/CP", link: null, key: null, soon: true },
        { type: "lecture", label: "讲义 20", title: "All-reduce、all-to-all、DeepEP", link: null, key: null, soon: true },
        { type: "lab", label: "实验 9", title: "All-reduce 模拟", link: null, key: null, soon: true },
        { type: "homework", label: "作业 10", title: "通信量", link: null, key: null, soon: true },
        { type: "project", label: "项目 3", title: "Checkpoint 1", link: null, key: null, soon: true },
      ],
      right: [
        { type: "reading", title: "DeepEP", link: "https://github.com/deepseek-ai/DeepEP" },
        { type: "reading", title: "DeepGEMM", link: "https://github.com/deepseek-ai/DeepGEMM" },
        { type: "video", title: "B站：NCCL 源码解读", link: "https://www.bilibili.com/video/BV17WUwBxE7Y/" },
      ],
    },
    {
      n: 11,
      dates: "1月4日 – 1月10日",
      theme: "Profiling 与 Benchmark",
      items: [
        { type: "lecture", label: "讲义 21", title: "nsys / ncu / roofline", link: null, key: null, soon: true },
        { type: "lecture", label: "讲义 22", title: "Benchmark 方法", link: null, key: null, soon: true },
        { type: "lab", label: "实验 10", title: "Profile 一个 kernel", link: null, key: null, soon: true },
        { type: "homework", label: "作业 11", title: "Benchmark 报告", link: null, key: null, soon: true },
        { type: "project", label: "项目 3", title: "Checkpoint 2", link: null, key: null, soon: true },
      ],
      right: [
        { type: "reading", title: "NVIDIA Nsight 文档", link: "https://docs.nvidia.com/nsight-compute/" },
        { type: "reading", title: "examples/benchmark.py", link: "https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/examples/benchmark.py" },
        { type: "video", title: "B站：nsys + ncu 实操", link: "https://www.bilibili.com/video/BV1se8B6bELY/" },
      ],
    },
    {
      n: 12,
      dates: "1月11日 – 1月17日",
      theme: "论文与开源",
      items: [
        { type: "lecture", label: "讲义 23", title: "论文：DeepSeek-V3", link: null, key: null, soon: true },
        { type: "lecture", label: "讲义 24", title: "论文：vLLM / PagedAttention", link: null, key: null, soon: true },
        { type: "lab", label: "实验 11", title: "论文复现 / PR", link: null, key: null, soon: true },
        { type: "homework", label: "作业 12", title: "论文笔记", link: null, key: null, soon: true },
        { type: "project", label: "项目 3", title: "最终：EPLB 报告", link: null, key: null, soon: true },
      ],
      right: [
        { type: "reading", title: "vLLM PagedAttention", link: "https://arxiv.org/abs/2309.06180" },
        { type: "reading", title: "DeepSeek-V3", link: "https://arxiv.org/abs/2412.19437" },
        { type: "video", title: "B站：Transformer 论文精读", link: "https://www.bilibili.com/video/BV1pu411o7BE/" },
      ],
    },
    {
      n: 13,
      dates: "1月18日 – 1月24日",
      theme: "最终项目",
      items: [
        { type: "lecture", label: "讲义 25", title: "复习与补漏", link: null, key: null, soon: true },
        { type: "lecture", label: "讲义 26", title: "进阶专题", link: null, key: null, soon: true },
        { type: "lab", label: "实验 12", title: "最终项目开题", link: null, key: null, soon: true },
        { type: "project", label: "最终项目", title: "Proposal 截止", link: null, key: null, soon: true },
      ],
      right: [
        { type: "reading", title: "46 周路线", link: "../../doc/study_plan_zh.md" },
      ],
    },
    {
      n: 14,
      dates: "1月25日 – 1月31日",
      theme: "实现与模拟面试",
      items: [
        { type: "lecture", label: "讲义 27", title: "最终项目 workshop", link: null, key: null, soon: true },
        { type: "lecture", label: "讲义 28", title: "模拟面试", link: null, key: null, soon: true },
        { type: "project", label: "最终项目", title: "实现 + Benchmark", link: null, key: null, soon: true },
      ],
      right: [
        { type: "reading", title: "面试清单", link: "../../doc/study_plan_zh.md" },
      ],
    },
    {
      n: 15,
      dates: "2月1日 – 2月7日",
      theme: "展示与收尾",
      items: [
        { type: "lecture", label: "讲义 29", title: "展示", link: null, key: null, soon: true },
        { type: "lecture", label: "讲义 30", title: "收尾与下一步", link: null, key: null, soon: true },
        { type: "project", label: "最终项目", title: "Demo + 博客 + 简历 bullet", link: null, key: null, soon: true },
      ],
      right: [
        { type: "reading", title: "简历模板", link: "../../doc/study_plan_zh.md" },
      ],
    },
  ],

  syllabus: [
    {
      title: "你的入门起点",
      html: `<p>完全零基础先做<a href="notes/foundations.html">Python 起步</a>；会函数、循环、列表之后再做<a href="notes/prep.html">调试预备</a>。先修复小程序，再做第一讲与 Lab0，不要求先学会 C++ 或安装 CUDA。</p>
        <p><strong>分层：</strong>核心必修是 Python 手算、实现、测试；CUDA/TileLang 与 profiling 等补过 C++ 后再做。Unit1 与 Unit2 第 5–6 周可运行；第 7–15 周待发布。</p>`,
    },
    {
      title: "课程描述",
      html: `<p>这门课讲两件事：<strong>算子</strong>（GPU 上的 vector add、reduction、softmax、RMSNorm、GEMM、attention）和<strong>推理系统</strong>（KV cache、PagedAttention、continuous batching、MoE/EP、投机解码）。</p>
        <p>课程以 DeepInfer（DeepSeek 的开源推理框架）为工业级参考实现，但<strong>不要求你运行完整 DeepInfer</strong>。做法是：用纯 Python SIMT 模拟器复现机制，用 roofline 复现性能模型，用仓库真实代码做对照阅读，用 Project 把机制组合成可运行的小系统。</p>
        <p class="small muted">这不是“CUDA API 大全”，也不是“DeepInfer 源码逐行讲解”。它是一条从写对到写快、从算子到系统的路径。</p>`,
    },
    {
      title: "学习成果",
      html: `<p>学完 Unit 1（Week 1–4），你应该能：</p>
        <ul>
          <li>解释 grid/block/warp/thread、coalescing、shared memory、bank conflict、occupancy；</li>
          <li>独立写出 vector add、tree reduction、softmax、RMSNorm、naive/tiled matmul；</li>
          <li>用 roofline 判断 memory-bound / compute-bound，并估算理论时间；</li>
          <li>计算 KV cache 大小，解释 MLA 为什么省显存；</li>
          <li>解释 per-tensor/per-token/per-group 量化的差别和误差来源。</li>
        </ul>
        <p>学完第二、三单元，你应该能解释 PagedAttention、continuous batching、chunked prefill、MoE/EP/EPLB、投机解码，并写一个 mini inference engine。</p>`,
    },
    {
      title: "形式与投入",
      html: `<table>
        <thead><tr><th>环节</th><th>形式</th><th>目的</th></tr></thead>
        <tbody>
          <tr><td>讲义</td><td>分段阅读、手算、预测、推导</td><td>将公式连接到代码变量</td></tr>
          <tr><td>实验</td><td>按题推进，可分多次；分级提示</td><td>从示例走到独立实现和修错</td></tr>
          <tr><td>作业</td><td>3–5 小时，数学 + 小函数</td><td>补上“为什么”和边界情况</td></tr>
          <tr><td>项目</td><td>2–3 周，phase + checkpoint</td><td>组合成作品</td></tr>
          <tr><td>Checkoff</td><td>3 个问题，写在 notes/</td><td>防止“看了等于会了”</td></tr>
          <tr><td>小测</td><td>10 分钟闭卷</td><td>检查核心数字和直觉</td></tr>
        </tbody>
      </table>
      <p><strong>标准节奏</strong>：15 周，12–15 小时/周。如果每周只有 8 小时，用 30 周：每一周拆成两周，实验和作业不变。</p>`,
    },
    {
      title: "已发布内容的评分",
      html: `<p>Lab0–3 共 40 分，HW1–2 共 20 分，Project1 共 30 分，Quiz1–2 共 10 分，总计 100 分。Python 起步与调试预备不计分；未发布单元不进入分母。</p>
      <p>Unit2 第 5、6 周各自单独验收 100 分（Lab 20、HW 30、Project 阶段 50），不并入 Unit1。教材重整不改变已有评分或进度。</p>
      <p>按代码、推导、反例、解释与复现实验给出证据，详见<a href="notes/assessment.html">完整评分与掌握度 rubric</a>。网站勾选只记录进度，不代表测试通过或成绩。</p>`,
    },
    {
      title: "协作",
      html: `<table>
        <thead><tr><th>类型</th><th>规则</th></tr></thead>
        <tbody>
          <tr><td>讲义 / 笔记</td><td>随便分享</td></tr>
          <tr><td>实验</td><td>可以讨论、互相看代码；每个人自己敲一遍</td></tr>
          <tr><td>作业</td><td>可以讨论思路；代码自己写；checkoff 自己答</td></tr>
          <tr><td>项目</td><td>可以找伙伴 pair；每个人要能讲清全部代码</td></tr>
          <tr><td>Checkoff</td><td>必须独立回答，不能看答案</td></tr>
        </tbody>
      </table>
      <p>记录独立完成与参考后修复的部分，方便复习时找到需要重做的概念。</p>`,
    },
    {
      title: "AI 使用政策",
      html: `<p><strong>可以用 AI：</strong>解释代码/论文、画调用图、列大纲、review kernel、生成边界测试、解释报错。</p>
        <p><strong>不要用 AI：</strong>直接生成 Lab/Homework/Project 的完整实现；在你还没尝试之前问答案；让 AI 替你写 checkoff；把 AI 生成的代码贴进项目而不理解。</p>
        <div class="callout"><p><strong>简单规则</strong>：如果你不能在 5 分钟内给一个朋友讲清这段代码，就不要提交它。</p></div>`,
    },
    {
      title: "运行与保存作业",
      html: `<p>课程根目录应能直接看到 course 文件夹。ZIP 使用解压根目录；个人主页源码仓库先执行 <code>cd _courses/deepinfer</code>。</p>
        <pre><code>python course/tools/grade.py lab00</code></pre>
        <p>在自己控制的学习目录保存手算、调试和 checkoff；无需为了完成练习提交 Git，更不需要把答案或私人笔记推送到公开仓库。<a href="notes/learning-path.html">查看完整自学路线 →</a></p>`,
    },
    {
      title: "这门课不做什么",
      html: `<ul>
          <li>不教你 CUDA 的所有 API；</li>
          <li>不逐行讲 DeepInfer 的 5 万行代码；</li>
          <li>不要求你先运行完整上游模型或购买 GPU；</li>
          <li>不承诺 15 周后你能通过所有面试；它只是把基础打牢。</li>
        </ul>
        <p><strong>一句话：</strong>先写对，再写快，再讲清楚。测试是规格，checkoff 是镜子，项目是作品。</p>`,
    },
  ],

  videos: [
    {
      title: "CUDA / GPU 入门",
      intro: "先建立执行模型和内存层次，再谈优化。不要从“高级优化技巧”开始。",
      items: [
        {
          title: "CUDA 编程基础入门系列（持续更新）",
          source: "B站 · 权双",
          length: "系列 · 约 10 小时",
          level: "入门",
          url: "https://www.bilibili.com/video/BV1sM4y1x7of/",
          why: "从装环境、线程模型到内存模型，适合完全没写过 CUDA 的人。",
          watch: "第 1–3 集装环境；线程模型 / 全局索引；内存模型和 reduction。",
        },
        {
          title: "NVIDIA 官方 CUDA 课程（中英字幕）",
          source: "B站 · NVIDIA",
          length: "系列",
          level: "入门",
          url: "https://www.bilibili.com/video/BV1JJ4m1P7xW/",
          why: "官方视角，覆盖编程模型、内存、性能分析。适合和权双的课交叉看。",
          watch: "Programming Model、Memory Hierarchy、Performance 三部分。",
        },
        {
          title: "CUDA Crash Course（CoffeeBeforeArch）",
          source: "B站 · 搬运",
          length: "12+ 集",
          level: "入门",
          url: "https://www.bilibili.com/video/BV127411G76m/",
          why: "每集一个 kernel，从 vector add 到 reduction，代码感很强。",
          watch: "前 12 集；重点看 reduction 的 shared memory / warp shuffle 版本。",
        },
        {
          title: "GPU 并行计算与 CUDA 编程（科大周老师）",
          source: "B站",
          length: "系列",
          level: "进阶",
          url: "https://www.bilibili.com/video/BV1kx411m7Fk/",
          why: "更偏体系结构：SIMT、内存层次、并行模式。",
          watch: "先看 GPU 硬件、内存模型、并行模式；优化技巧可以后看。",
        },
        {
          title: "NVIDIA CUDA 编程实战与 GPU 并行计算（13 讲）",
          source: "B站 · 中英字幕",
          length: "13 讲",
          level: "系统",
          url: "https://www.bilibili.com/video/BV1VeQFBDEyU/",
          why: "NVIDIA / OLCF 的经典训练系列，适合系统补课。",
          watch: "共享内存、coalescing、occupancy、reduction、GEMM 几讲。",
        },
      ],
    },
    {
      title: "Attention / FlashAttention",
      intro: "先看 Transformer 的直觉，再看 FlashAttention 的 IO 和 online softmax。",
      items: [
        {
          title: "Transformer 论文逐段精读",
          source: "B站 · 跟李沐学 AI",
          length: "1:27:05",
          level: "入门",
          url: "https://www.bilibili.com/video/BV1pu411o7BE/",
          why: "把 attention / multi-head / positional encoding 讲透，配合 Lab 2 看。",
          watch: "全看；重点 20:00–50:00 的 self-attention 和 multi-head。",
        },
        {
          title: "论文分享：新型注意力算法 FlashAttention",
          source: "B站 · 先进编译实验室",
          length: "约 30 分钟",
          level: "进阶",
          url: "https://www.bilibili.com/video/BV1zs4y1J7tb/",
          why: "从 IO 角度解释 FlashAttention 为什么快，适合先建立直觉。",
          watch: "全看；重点看“标准 attention 的 HBM 读写”和 tiling 图。",
        },
        {
          title: "FlashAttention V1 原理篇：不会 CUDA 也能看懂",
          source: "B站",
          length: "约 20 分钟",
          level: "入门",
          url: "https://www.bilibili.com/video/BV1VVDLB4Ecw/",
          why: "用图讲清楚分块和 online softmax，适合配合 Lecture 6。",
          watch: "全看；看完能画出 S/P 不落 HBM 的流程图。",
        },
        {
          title: "FlashAttention 论文精读：IO 感知的精确注意力",
          source: "B站",
          length: "约 40 分钟",
          level: "进阶",
          url: "https://www.bilibili.com/video/BV1Bz42187PZ/",
          why: "更贴近论文的算法和复杂度分析。",
          watch: "重点看 Algorithm 1、IO complexity、causal mask 的处理。",
        },
        {
          title: "Flash Attention 详解（Umar Jamil）",
          source: "B站 · 精译",
          length: "7:38:18",
          level: "深入",
          url: "https://www.bilibili.com/video/BV1t2UxY7Ep9/",
          why: "从零推导 + 代码，适合想彻底吃透的人。不要第一遍就看。",
          watch: "先看前 2 小时；等 Lab 2 做完再回来看实现。",
        },
      ],
    },
    {
      title: "vLLM / PagedAttention",
      intro: "先看论文精读，再看代码；否则很容易陷进 block table 的细节。",
      items: [
        {
          title: "AI INFRA 学习 02：vLLM PagedAttention 论文精读",
          source: "B站 · AI Infra 学习",
          length: "约 1 小时",
          level: "进阶",
          url: "https://www.bilibili.com/video/BV1GWjjzfE1b/",
          why: "把 PagedAttention 的动机、block table、共享讲清楚。",
          watch: "全看；配合 Lab 3 的连续 / paged 两种布局。",
        },
        {
          title: "Decode 阶段 & PagedAttention 代码详解",
          source: "B站 · 从 0 手撸 vLLM",
          length: "约 1 小时",
          level: "深入",
          url: "https://www.bilibili.com/video/BV14a5r6YEhY/",
          why: "看 block_table 怎么变成 slot_mapping，以及 decode kernel 怎么读 KV。",
          watch: "重点看 block table → slot mapping → KV gather 的调用链。",
        },
        {
          title: "vLLM 整体架构与请求预处理",
          source: "B站",
          length: "系列",
          level: "系统",
          url: "https://www.bilibili.com/video/BV1YcywBiEJ1/",
          why: "把 vLLM 的 Processor、Scheduler、PagedAttention、continuous batching 串起来。",
          watch: "先看架构总览和 PagedAttention；调度部分配合 Unit 2。",
        },
      ],
    },
    {
      title: "MoE / 专家并行",
      intro: "先看可视化和 CS336，再看 DeepSeek 的细粒度专家和 EPLB。",
      items: [
        {
          title: "MoE 终于迎来可视化解读",
          source: "B站",
          length: "14 集 · 约 6 小时",
          level: "入门",
          url: "https://www.bilibili.com/video/BV1Gj9ZYdE4N/",
          why: "从 1991 年 MoE 讲到 DeepSeekMoE，图和动画很多。",
          watch: "01–02 概览；04–08 论文；11–12 手撕代码。",
        },
        {
          title: "斯坦福 CS336 第四讲：详解 MoE 架构",
          source: "B站 · 中英双语",
          length: "约 1 小时",
          level: "进阶",
          url: "https://www.bilibili.com/video/BV1Wy7ozhEoJ/",
          why: "从系统和训练视角讲 MoE，适合配合 DeepSeek-V3 论文。",
          watch: "全看；重点看 routing、capacity、load balancing。",
        },
        {
          title: "3 小时深度拆解 MoE：ST-MoE / DeepSeek-MoE / MLA",
          source: "B站",
          length: "约 3 小时",
          level: "深入",
          url: "https://www.bilibili.com/video/BV14sMC6sEbD/",
          why: "把 MoE 和 MLA 放在一起讲，适合 Unit 3 前补课。",
          watch: "先看 DeepSeek-MoE 和 MLA 两段；训练细节可以跳过。",
        },
      ],
    },
    {
      title: "投机解码",
      intro: "先看拒绝采样和接受率，再看 EAGLE / MTP / 树形投机。",
      items: [
        {
          title: "Speculative Decoding 推测解码方案详解",
          source: "B站 · AI Infra 学习",
          length: "约 40 分钟",
          level: "进阶",
          url: "https://www.bilibili.com/video/BV1Q5KWzQEhn",
          why: "从 draft / verify / rejection sampling 讲到 EAGLE。",
          watch: "全看；重点看 rejection sampling 的无损性证明。",
        },
        {
          title: "从 0 理解投机解码：拒绝采样、MTP、自投机、树形投机",
          source: "B站",
          length: "22:01",
          level: "入门",
          url: "https://www.bilibili.com/video/BV1zV5i6gEZT/",
          why: "短小，适合先建立直觉；配合 Lecture 14。",
          watch: "全看；看完能解释“为什么最多多生成一个 token”。",
        },
        {
          title: "基于草稿自拟的投机解码（车万翔）",
          source: "B站 · 哈工大",
          length: "约 30 分钟",
          level: "进阶",
          url: "https://www.bilibili.com/video/BV1HFC8YqEZj/",
          why: "不需要额外 draft 模型的思路，适合理解“自投机”。",
          watch: "重点看 draft 生成和并行验证两部分。",
        },
      ],
    },
    {
      title: "性能分析 / Profiling",
      intro: "先 nsys 看全局，再 ncu 下钻到单个 kernel；不要反过来。",
      items: [
        {
          title: "nsys + ncu 实操：从压测下钻到算子",
          source: "B站",
          length: "约 55 分钟",
          level: "进阶",
          url: "https://www.bilibili.com/video/BV1se8B6bELY/",
          why: "用 vLLM / SGLang 做完整 profiling，包含时间戳和判断表。",
          watch: "00:00 分工；09:52 基线指标；15:22 nsys；36:06 ncu；41:54 22.66% / 81.59% 案例；49:20 调参复测。",
        },
        {
          title: "CUDA 向量化到底快在哪？用 NCU 指标拆给你看",
          source: "B站",
          length: "约 20 分钟",
          level: "进阶",
          url: "https://www.bilibili.com/video/BV1unHj6JEtB/",
          why: "把 vectorization、sector、wavefront 这些 NCU 指标讲清楚。",
          watch: "全看；配合 Lab 0 的 coalescing 数字。",
        },
        {
          title: "深入理解 Nsight Systems 和 Nsight Compute",
          source: "B站",
          length: "约 30 分钟",
          level: "入门",
          url: "https://www.bilibili.com/video/BV13w411o7cu/",
          why: "工具入门：两个工具分别看什么、怎么读报告。",
          watch: "全看；先建立工具地图，再去看上面的实操。",
        },
      ],
    },
    {
      title: "TileLang / 算子开发",
      intro: "TileLang 是 DeepInfer 的算子 DSL；先看训练营的 GPGPU 和算子开发部分。",
      items: [
        {
          title: "TileLang 开源训练营",
          source: "B站 · TileLang",
          length: "系列",
          level: "系统",
          url: "https://www.bilibili.com/video/BV182Ko6fE9A/",
          why: "从 GPGPU 硬件、SIMT、coalescing 到 TileLang 写 GEMM / attention。",
          watch: "06–09 GPGPU / 性能优化；19–23 TileLang 语法；32–38 算子开发；36 Roofline。",
        },
        {
          title: "CANN TileLang 算子开发实操",
          source: "B站 · CANN",
          length: "约 1 小时",
          level: "进阶",
          url: "https://www.bilibili.com/video/BV148dzBhEDa/",
          why: "如果目标包含昇腾 / 国产芯片，可以看 TileLang 在 CANN 上的落地。",
          watch: "重点看 TileLang 到 Ascend 的 lowering 和算子示例。",
        },
      ],
    },
    {
      title: "NCCL / 分布式通信",
      intro: "先理解 all-reduce 的 ring/tree，再看 NCCL 的拓扑和协议。",
      items: [
        {
          title: "NCCL 集合通信源码解读",
          source: "B站",
          length: "系列 · 很长",
          level: "深入",
          url: "https://www.bilibili.com/video/BV17WUwBxE7Y/",
          why: "从 ncclCommInitRank 到 ring/tree、LL/LL128 协议，适合查漏补缺。",
          watch: "1.1–1.4 初始化；3.1 双二叉树；LL / LL128 协议；对称内存。不要一次看完。",
        },
      ],
    },
  ],

  textbook: [
  {
    "id": "python-start",
    "num": "00A",
    "title": "Python 起步",
    "intro": "沿着一批请求的长度，从计算字节数读到推导存储边界；正文在一页内连续阅读。",
    "understanding": "先预测运行结果，再写返回值、分支、列表循环，最后独立实现连续存储边界。",
    "keyPoints": [
      "先修：整数四则运算。",
      "产物：foundations Q0–Q4；每题有公开规格与逐层提示。"
    ],
    "links": [
      {
        "label": "打开起步教材与练习",
        "url": "notes/foundations.html"
      }
    ]
  },
  {
    "id": "debug-start",
    "num": "00B",
    "title": "调试、矩阵与测试",
    "intro": "把能写小函数变成能定位错误。",
    "understanding": "从 traceback 找到一个最小失败输入；先手算期望，再修复副本、矩阵累计和浮点比较。",
    "keyPoints": [
      "先修：函数、列表、for、if。",
      "产物：prep Q0–Q3 与一份调试记录。"
    ],
    "links": [
      {
        "label": "打开预备单元",
        "url": "notes/prep.html"
      }
    ]
  },
  {
    "id": "gpu-model",
    "num": "01",
    "title": "执行模型与地址",
    "intro": "先在 Python 中追踪编号、尾部和读写。",
    "understanding": "列出 bid/tid 对应的元素，说明何时需要 i<n。SIMT 模拟器记录算法访问，不模拟 GPU 的真实时钟。",
    "keyPoints": [
      "先修：调试预备。",
      "产物：Lab0 Q0–Q6；C++/CUDA 选读。"
    ],
    "formula": "global_index = block_id * block_dim + thread_id",
    "links": [
      {
        "label": "第 1 讲",
        "url": "notes/lec01.html"
      },
      {
        "label": "Lab0",
        "url": "notes/lab00.html"
      }
    ]
  },
  {
    "id": "memory",
    "num": "02",
    "title": "访存：先说统计单位",
    "intro": "一次访问触及多少地址，需要先固定模型。",
    "understanding": "Lab0 以 128B line 统计；对齐的 32 个连续 int32 触及 1 条 line，同时覆盖 4 个 32B sector。它们不能混称为真实硬件的同一种 transaction。",
    "keyPoints": [
      "先修：L1 的 word/byte 地址。",
      "核心：连续与跨步访问；shared memory、occupancy 选读。"
    ],
    "links": [
      {
        "label": "第 2 讲",
        "url": "notes/lec02.html"
      }
    ]
  },
  {
    "id": "reduction",
    "num": "03",
    "title": "Reduction、Softmax 与归一化",
    "intro": "先稳定地算对，再讨论数据要读几次。",
    "understanding": "用极大 logits 检验减 max 的作用；用零向量检查 eps；先检查长度和有限值再比较浮点误差。",
    "keyPoints": [
      "先修：列表累计、平方与平均值。",
      "产物：Lab1，包含空输入与参数契约。"
    ],
    "formula": "softmax(x)[i] = exp(x[i]-m) / sum(exp(x[j]-m))",
    "links": [
      {
        "label": "第 3 讲",
        "url": "notes/lec03.html"
      },
      {
        "label": "Lab1",
        "url": "notes/lab01.html"
      }
    ]
  },
  {
    "id": "roofline",
    "num": "04",
    "title": "矩阵乘、Tiling 与 Roofline",
    "intro": "用可观察的读写计数解释复用。",
    "understanding": "手算非整除 tile 的尾块，与 MatrixMemory 的计数对照。是否 compute-bound 取决于流量模型、数据类型与设备拐点，不能只凭矩阵名字判断。",
    "keyPoints": [
      "先修：prep 矩阵乘、Lab0 时间单位、Lab1。",
      "产物：Project1 Phase2 与实验。"
    ],
    "formula": "T_lower = max(bytes / bandwidth, FLOPs / peak_FLOPs_per_second)",
    "links": [
      {
        "label": "第 4 讲",
        "url": "notes/lec04.html"
      },
      {
        "label": "Project1",
        "url": "notes/project1.html"
      }
    ]
  },
  {
    "id": "attention",
    "num": "05",
    "title": "从矩阵组合到 Attention",
    "intro": "把点积、缩放、mask、归一化与加权和逐步连接。",
    "understanding": "手算一个输出行，明确 Q/K/V 的 shape。含对角线的 causal 有 s(s+1)/2 个有效配对；稠密计算再 mask 仍执行完整乘法。",
    "keyPoints": [
      "先修：矩阵乘、转置、稳定 softmax。",
      "产物：Lab2 Q1–Q6。"
    ],
    "formula": "FLOPs = 4 * batch * heads * effective_pairs * head_dim",
    "links": [
      {
        "label": "第 5 讲",
        "url": "notes/lec05.html"
      },
      {
        "label": "Lab2",
        "url": "notes/lab02.html"
      }
    ]
  },
  {
    "id": "flash",
    "num": "06",
    "title": "Attention 流量与在线归一化",
    "intro": "区分要存多少、请求搬多少、设备实际搬多少。",
    "understanding": "分块合并时维护最大值和归一化量；三阶段 materialized 模型计 S/P 各一次写和读。Q/K/V/O 各一次访问只是理想下界。",
    "keyPoints": [
      "先修：L3、L4、L5。",
      "产物：Lab2 Q7–Q8；完整 HW1/2 留到 L7、L8 后统一做。",
      "prefill/decode 的瓶颈必须带 shape、模型和设备条件。"
    ],
    "links": [
      {
        "label": "第 6 讲",
        "url": "notes/lec06.html"
      }
    ]
  },
  {
    "id": "kv-cache",
    "num": "07",
    "title": "KV Cache 容量与页映射",
    "intro": "把历史数据保留与空间管理分开理解。",
    "understanding": "缓存避免重复计算历史 K/V；单步 attention 仍读取历史。Lab3 用 append/get/free 测试单序列映射、副本隔离和清空；有限容量、共享引用与失败原子性在 L9/Lab4 学习。",
    "keyPoints": [
      "先修：L5、整数分组、列表副本。",
      "产物：Lab3；补齐 HW2 的 KV 先修，不依赖尚未发布的单元。"
    ],
    "formula": "MHA/GQA KV bytes = 2 * layers * kv_heads * head_dim * seq * batch * element_bytes",
    "links": [
      {
        "label": "第 7 讲",
        "url": "notes/lec07.html"
      },
      {
        "label": "Lab3",
        "url": "notes/lab03.html"
      }
    ]
  },
  {
    "id": "quantization",
    "num": "08",
    "title": "量化、误差与显存预算",
    "intro": "先理解数值映射，再比较整数与低精度浮点格式。",
    "understanding": "明确 scale 的单位、舍入和裁剪，单独计算元数据成本。HW1 的均匀整数编码不是 FP8/FP4 仿真。",
    "keyPoints": [
      "先修：浮点误差、归一化、KV 容量。",
      "产物：HW1/2、Quiz2 与独立预算题。"
    ],
    "links": [
      {
        "label": "第 8 讲",
        "url": "notes/lec08.html"
      },
      {
        "label": "HW1",
        "url": "notes/hw01.html"
      },
      {
        "label": "HW2",
        "url": "notes/hw02.html"
      }
    ]
  },
  {
    "id": "project-evidence",
    "num": "09",
    "title": "综合项目：把推导变成证据",
    "intro": "组合已学算子，核对流量，再写可复现的实验报告。",
    "understanding": "正确性与访问计数分别检查。Python wall time 与模型的 GPU 时间下界分别报告，不从一次耗时推断 GPU 提速。",
    "keyPoints": [
      "先修：Lab0/1、L4；按项目 checkpoint 推进。",
      "产物：Kernel Zoo、实验记录、报告和口头解释。"
    ],
    "links": [
      {
        "label": "Project1 规格",
        "url": "notes/project1.html"
      },
      {
        "label": "验收与评分",
        "url": "notes/assessment.html"
      }
    ]
  },
  {
    "id": "engine-start",
    "num": "Unit2 · 第 5 周",
    "title": "有限容量与请求生命周期",
    "intro": "从页表图解、引用生命周期和前缀容量推导，逐步走到请求准入与释放。",
    "understanding": "区分物理页、逻辑 token、所有权引用与请求状态。先验证容量守恒，再讨论调度速度。",
    "keyPoints": ["先修：L7、Lab3、HW2。", "产物：Lab4、HW5 与 Project2 Phase1 报告。", "本批单独验收，不改变 Unit1 的 100 分分母。"],
    "links": [
      {"label": "L9 共享块池", "url": "notes/lec09.html"},
      {"label": "L10 前缀缓存", "url": "notes/lec10.html"},
      {"label": "Lab4", "url": "notes/lab04.html"},
      {"label": "HW5", "url": "notes/hw05.html"},
      {"label": "Project2 Phase1", "url": "notes/project2.html"}
    ]
  },
  {
    "id": "engine-scheduling",
    "num": "Unit2 · 第 6 周",
    "title": "逐迭代调度与延迟口径",
    "intro": "用同负载调度时间线、预算分配图和逐步实现，理解 decode、prefill 与首 token。",
    "understanding": "把准入、策略、执行、指标分开；模型 tick 不是 GPU 毫秒。",
    "keyPoints": ["先修：Phase1、L11/L12。", "产物：Lab5、HW6、Phase2 与 18 组实验。", "独立批次评分；尚未集成真实 KV 或模型执行。"],
    "links": [
      {"label": "L11 Continuous batching", "url": "notes/lec11.html"},
      {"label": "L12 Chunked prefill", "url": "notes/lec12.html"},
      {"label": "Lab5", "url": "notes/lab05.html"},
      {"label": "HW6", "url": "notes/hw06.html"},
      {"label": "Project2 Phase2", "url": "notes/project2-scheduler.html"}
    ]
  },
  {
    "id": "future",
    "num": "后续",
    "title": "采样、通信与负载均衡（待发布）",
    "intro": "第 7–15 周是路线规划，尚没有完整教材、starter 与测试。",
    "understanding": "完成第 6 周后可补 Linux/C++、数据结构、并发与网络。完整 Mini Engine 与 EPLB 项目仍待后续发布。",
    "keyPoints": [
      "未发布内容不计入当前成绩或完成率。"
    ],
    "links": [
      {
        "label": "长期路线",
        "url": "../../doc/study_plan_zh.md"
      }
    ]
  }
],

  resources: [
    {
      title: "课程与讲座",
      intro: "系统课，不要刷完；每看 20 分钟就写代码。",
      items: [
        { title: "GPU MODE", url: "https://www.youtube.com/@GPUMODE", desc: "现代 GPU kernel / CUDA / 性能优化讲座", tag: "video" },
        { title: "MIT 6.172", url: "https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/", desc: "性能工程、cache、并行", tag: "course" },
        { title: "Stanford CS149", url: "https://gfxcourses.stanford.edu/cs149/fall23/", desc: "GPU 架构、数据并行", tag: "course" },
        { title: "CUDA C++ Guide", url: "https://docs.nvidia.com/cuda/cuda-c-programming-guide/", desc: "官方编程模型和内存层次", tag: "docs" },
      ],
    },
    {
      title: "博客与笔记",
      intro: "完成本地教材和练习之后的扩展阅读。",
      items: [
        { title: "图解 Transformer", url: "https://jalammar.github.io/illustrated-transformer/", desc: "attention 的直观图解", tag: "blog" },
        { title: "vLLM：PagedAttention", url: "https://blog.vllm.ai/2023/06/20/vllm.html", desc: "KV cache 分页的动机和实现", tag: "blog" },
        { title: "Anyscale：continuous batching", url: "https://www.anyscale.com/blog/continuous-batching-llm-inference", desc: "continuous batching 的吞吐收益", tag: "blog" },
        { title: "Hugging Face：MoE 讲解", url: "https://huggingface.co/blog/moe", desc: "MoE 基础、路由、并行", tag: "blog" },
        { title: "siboehm: CUDA matmul", url: "https://siboehm.com/articles/22/CUDA-MMM", desc: "从 naive 到 tiled GEMM 的逐步优化", tag: "blog" },
        { title: "NVIDIA: inference optimization", url: "https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/", desc: "量化、KV cache、batching 综述", tag: "blog" },
      ],
    },
    {
      title: "论文",
      intro: "按课程顺序读，不要从最难的开始。",
      items: [
        { title: "FlashAttention", url: "https://arxiv.org/abs/2205.14135", desc: "IO-aware attention", tag: "paper" },
        { title: "FlashAttention-2", url: "https://arxiv.org/abs/2307.08691", desc: "更好的并行和 work partitioning", tag: "paper" },
        { title: "PagedAttention", url: "https://arxiv.org/abs/2309.06180", desc: "vLLM 的 KV cache 分页", tag: "paper" },
        { title: "Orca", url: "https://www.usenix.org/conference/osdi22/presentation/yu", desc: "continuous batching / iteration-level scheduling", tag: "paper" },
        { title: "Sarathi-Serve", url: "https://arxiv.org/abs/2403.02310", desc: "chunked prefill + stall-free batching", tag: "paper" },
        { title: "DeepSeek-V2", url: "https://arxiv.org/abs/2405.04434", desc: "MLA / compressed KV", tag: "paper" },
        { title: "DeepSeek-V3", url: "https://arxiv.org/abs/2412.19437", desc: "MoE / FP8 / EP", tag: "paper" },
        { title: "Speculative Decoding", url: "https://arxiv.org/abs/2211.17192", desc: "draft + verification", tag: "paper" },
      ],
    },
    {
      title: "代码仓库",
      intro: "读 README 和 tests，不要一上来读全部 kernel。",
      items: [
        { title: "DeepInfer", url: "https://github.com/deepseek-ai/DeepInfer", desc: "本仓库；生产推理框架", tag: "repo" },
        { title: "vLLM", url: "https://github.com/vllm-project/vllm", desc: "PagedAttention / continuous batching", tag: "repo" },
        { title: "SGLang", url: "https://github.com/sgl-project/sglang", desc: "RadixAttention / 调度", tag: "repo" },
        { title: "TensorRT-LLM", url: "https://github.com/NVIDIA/TensorRT-LLM", desc: "工业级 kernel/engine", tag: "repo" },
        { title: "CUTLASS", url: "https://github.com/NVIDIA/cutlass", desc: "GEMM / tensor core 模板", tag: "repo" },
        { title: "Triton", url: "https://github.com/triton-lang/triton", desc: "高层 kernel DSL", tag: "repo" },
        { title: "TileLang", url: "https://github.com/tile-ai/tilelang", desc: "DeepInfer 使用的 kernel DSL", tag: "repo" },
        { title: "DeepGEMM", url: "https://github.com/deepseek-ai/DeepGEMM", desc: "FP8/FP4 GEMM", tag: "repo" },
        { title: "FlashMLA", url: "https://github.com/deepseek-ai/FlashMLA", desc: "MLA attention", tag: "repo" },
        { title: "DeepEP", url: "https://github.com/deepseek-ai/DeepEP", desc: "EP all-to-all", tag: "repo" },
        { title: "3FS", url: "https://github.com/deepseek-ai/3FS", desc: "分布式文件系统", tag: "repo" },
      ],
    },
    {
      title: "工具",
      intro: "本地能装的先装；ncu 在 WSL 可能受限。",
      items: [
        { title: "Nsight Compute", url: "https://docs.nvidia.com/nsight-compute/", desc: "kernel 计数器 / roofline", tag: "tool" },
        { title: "Nsight Systems", url: "https://docs.nvidia.com/nsight-systems/", desc: "timeline / CPU-GPU 重叠", tag: "tool" },
        { title: "PyTorch profiler", url: "https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html", desc: "Python + CUDA 采样", tag: "tool" },
        { title: "LeetGPU", url: "https://leetgpu.com/", desc: "在线 GPU kernel 练习", tag: "tool" },
      ],
    },
    {
      title: "B站 / 中文视频",
      intro: "搜索链接比固定 BV 更稳；选播放量高、更新时间近的。",
      items: [
        { title: "B站：CUDA 编程基础入门系列", url: "https://www.bilibili.com/video/BV1sM4y1x7of/", desc: "先看 1.5×，再动手", tag: "bilibili" },
        { title: "B站：GPU 并行计算（科大周老师）", url: "https://www.bilibili.com/video/BV1kx411m7Fk/", desc: "体系结构 / 并行模型", tag: "bilibili" },
        { title: "B站：FlashAttention 论文分享", url: "https://www.bilibili.com/video/BV1zs4y1J7tb/", desc: "论文精读 + 实现", tag: "bilibili" },
        { title: "B站：vLLM", url: "https://www.bilibili.com/video/BV14a5r6YEhY/", desc: "PagedAttention / 源码", tag: "bilibili" },
        { title: "B站：TileLang", url: "https://search.bilibili.com/all?keyword=TileLang", desc: "DSL / kernel 开发", tag: "bilibili" },
        { title: "B站：MoE 可视化解读", url: "https://www.bilibili.com/video/BV1Gj9ZYdE4N/", desc: "路由 / EP / EPLB", tag: "bilibili" },
      ],
    },
  ],

  sections: [
    {
      n: 0,
      title: "讨论 0：课程怎么运作",
      goals: "把课程流程、checkoff、AI 使用、提交方式讲清楚。",
      questions: [
        "你为什么学这个仓库？你想投算子、引擎、还是编译器？",
        "你每周能投入多少小时？标准档 12–15h，慢速版 30 周。",
        "你的 checkoff 准备怎么记录？建议 notes/ + 每周复盘。",
      ],
    },
    {
      n: 1,
      title: "讨论 1：GPU 模型与合并访问",
      goals: "能画 grid/block/warp/thread；能算一个 warp 的 transaction 数。",
      questions: [
        "一个 warp 读 32 个连续 int32，起始 word 16，要几个 128B transaction？",
        "为什么 `i = tid*grid + bid` 比 `i = bid*block + tid` 慢？用 stride 解释。",
        "shared memory 的 32-way bank conflict 怎么产生？怎么修？",
      ],
    },
    {
      n: 2,
      title: "讨论 2：Roofline 与 memory-bound kernel",
      goals: "能判断 memory/compute bound；能估算 softmax/RMSNorm 的 traffic。",
      questions: [
        "A100 2TB/s、312TFLOP/s，AI 拐点是多少？",
        "4096 维 float32 softmax，3-pass 和 2-pass 差多少字节？",
        "为什么融合 norm + residual + quant 能省时间？省的是算力还是带宽？",
      ],
    },
    {
      n: 3,
      title: "讨论 3：Attention",
      goals: "能写 attention 的参考实现；能解释 prefill/decode 的瓶颈差异。",
      questions: [
        "QK^T 和 AV 各多少 FLOPs？causal 为什么减半？",
        "prefill 为什么 compute-bound，decode 为什么 memory-bound？",
        "FlashAttention 省的是 FLOPs 还是 HBM IO？",
      ],
    },
    {
      n: 4,
      title: "讨论 4：KV cache 与 PagedAttention",
      goals: "能算 KV 大小；能解释 block table 和 prefix sharing。",
      questions: [
        "32 层、8 KV head、128 dim、2048 seq、bf16 的 KV 多大？",
        "MLA 为什么不需要乘 2？它存的是什么？",
        "PagedAttention 解决什么问题？代价是什么？",
      ],
    },
    {
      n: 5,
      title: "讨论 5：Batching 与调度",
      goals: "能解释 continuous batching / chunked prefill 的 trade-off。",
      questions: [
        "TTFT 和 TPOT 分别受什么影响？",
        "chunked prefill 为什么能降低 TPOT 抖动？",
        "什么情况下应该优先增大 batch，而不是增大 chunk？",
      ],
    },
    {
      n: 6,
      title: "讨论 6：MoE 与 EP",
      goals: "能解释 top-k gate、grouped GEMM、EPLB。",
      questions: [
        "MoE 的 all-to-all 通信量怎么估？",
        "EPLB 的 logical/physical expert 有什么区别？",
        "为什么 decode 用 Wide-EP，prefill 不用？",
      ],
    },
    {
      n: 7,
      title: "讨论 7：投机解码",
      goals: "能推导接受概率；能估算 speedup。",
      questions: [
        "接受概率为什么是 min(1, p/q)？拒绝时怎么修正分布？",
        "draft 长度 γ 越大越好吗？成本怎么算？",
        "变长验证的收益来自哪里？",
      ],
    },
    {
      n: 8,
      title: "讨论 8：Benchmark 与面试",
      goals: "能把项目讲成 STAR；能设计一个 benchmark。",
      questions: [
        "你的 kernel 比 baseline 快，是算法、带宽、还是测量误差？",
        "如果让你把 mini engine 再快 2×，你先做什么？",
        "用 3 分钟讲清你的 Project 1：问题、方案、数据、trade-off。",
      ],
    },
  ],

  career: {
    intro:
      "目标是 2028 届 Infra 秋招：算子、推理框架、高性能计算、编译器、性能工程。课程给你技术底座；这一页把底座翻译成简历、项目和面试准备。时间线按 2028 届（2027 年秋招）倒排；如果你指 2028 年秋天，整体后移一年。",
    timeline: [
      { period: "2026.10–2027.02", action: "打基础 + 简历 1.0 + 投春招实习", output: "实验 0–3、作业 1/2、项目 1、1 篇博客" },
      { period: "2027.03–2027.06", action: "实习 + Project 2", output: "Mini Engine、benchmark、2 篇博客" },
      { period: "2027.06–2027.07", action: "简历 2.0 + 提前批", output: "3 个项目、1 个 PR、3 篇博客" },
      { period: "2027.07–2027.09", action: "正式批 + 面试复盘", output: "5 次模拟面试、题库、投递记录" },
      { period: "2027.09–2027.10", action: "补录 / 谈薪 / 选 offer", output: "决策表 + 备选方案" },
    ],
    roles: [
      {
        title: "Kernel / 算子工程师",
        focus: "写 CUDA/TileLang/Triton kernel，做 memory-bound 融合、GEMM、attention、sampling。",
        keywords: "CUDA, TileLang, Triton, roofline, coalescing, GEMM, FlashAttention",
        project: "项目 1 + GPU 轨道",
      },
      {
        title: "推理框架工程师",
        focus: "调度、KV cache、continuous batching、PagedAttention、投机解码、异步。",
        keywords: "vLLM, PagedAttention, continuous batching, KV cache, spec decode",
        project: "项目 2 + DeepInfer 源码阅读",
      },
      {
        title: "性能工程师",
        focus: "用 nsys/ncu 找瓶颈，做 benchmark、roofline、端到端优化。",
        keywords: "nsys, ncu, roofline, MFU, benchmark, profiling",
        project: "项目 1 报告 + 项目 2 benchmark",
      },
      {
        title: "分布式 / MoE 工程师",
        focus: "EP/EPLB、all-to-all、DeepEP、Wide-EP、P/D 分离。",
        keywords: "NCCL, DeepEP, EP, EPLB, MoE, all-to-all",
        project: "项目 3",
      },
      {
        title: "编译器 / ML Systems",
        focus: "TileLang/TVM/MLIR、kernel lowering、autotune、图优化。",
        keywords: "TVM, TileLang, MLIR, schedule, autotune",
        project: "项目 1 GPU 轨道 + 开源 PR",
      },
    ],
    projectMap: [
      { project: "项目 1 · Kernel Zoo", roles: "Kernel / 性能", evidence: "8 个 kernel、roofline、nsys/ncu、报告" },
      { project: "项目 2 · Mini Engine", roles: "推理框架", evidence: "paged KV、batching、chunked prefill、spec decode、benchmark" },
      { project: "项目 3 · EPLB 模拟器", roles: "分布式 / MoE", evidence: "合成 trace、greedy 重排、imbalance 报告" },
      { project: "开源 PR", roles: "全部", evidence: "1 个 merged PR + 变更说明 + 测试" },
    ],
    bullets: [
      "实现 10 个 TileLang/CUDA 算子，覆盖 norm/rope/KV cache/attention/sampling/GEMM；RMSNorm 达到 HBM 带宽的 X%，decode attention 相比 PyTorch SDPA 加速 Y×。",
      "从零实现 PyTorch 推理引擎，支持 PagedAttention、continuous batching、chunked prefill 与 prefix cache；Qwen2.5-0.5B 上相比 HF generate 吞吐提升 X%，4K 上下文 TTFT 降低 Y%。",
      "构建 MoE EPLB 模拟器，复现 DeepInfer 的 greedy 重排策略；Zipf 负载下 max/mean 从 X 降到 Y，估算端到端吞吐提升 Z%。",
    ],
    questions: [
      {
        topic: "CUDA / kernel",
        items: [
          "grid/block/warp/thread 的关系？一个 block 最多多少 thread？",
          "什么是 coalescing？stride-8 的 warp 读 32 个 int32 要几个 transaction？",
          "shared memory 的 32 个 bank 怎么工作？bank conflict 怎么修？",
          "occupancy 受什么限制？高 occupancy 一定快吗？",
          "reduction 的 warp shuffle 怎么写？树有多深？",
          "softmax 为什么要减 max？online softmax 怎么更新？",
          "GEMM 的 tiling 为什么能减少 global traffic？tile 越大越好吗？",
          "tensor core 的 mma 是什么？FP16/BF16/FP8 的累加精度？",
          "TMA / cp.async 解决什么问题？double buffering 怎么工作？",
          "split-K 什么时候用？它带来什么额外开销？",
          "roofline 的横纵轴是什么？AI 拐点怎么算？",
          "nsys 和 ncu 分别看什么？DRAM throughput 低说明什么？",
        ],
      },
      {
        topic: "Inference engine",
        items: [
          "PagedAttention 解决什么问题？block table 怎么工作？",
          "continuous batching 和 static batching 的区别？",
          "chunked prefill 为什么能降低 TPOT 抖动？",
          "prefix cache 的命中率受什么影响？",
          "KV cache 大小怎么算？MLA 为什么省？",
          "prefill 和 decode 的瓶颈分别是什么？",
          "投机解码的接受概率怎么推导？",
          "变长验证的收益来自哪里？",
          "异步调度消除了哪些气泡？",
          "CUDA Graph 解决什么？不解决什么？",
        ],
      },
      {
        topic: "Systems / distributed",
        items: [
          "DP/TP/PP/EP/CP 的区别？各自适合什么？",
          "all-reduce 的 ring 算法通信量是多少？",
          "EP 的 all-to-all 通信量怎么估？",
          "为什么 decode 用 Wide-EP，prefill 不用？",
          "EPLB 的 logical/physical expert 有什么区别？",
          "P/D 分离的收益和代价？",
          "3FS / KV store 在推理系统里扮演什么角色？",
          "如何设计一个满足 TTFT/TPOT SLO 的调度器？",
        ],
      },
      {
        topic: "Coding / hand-written",
        items: [
          "手写 reduction（warp shuffle）。",
          "手写 softmax / layernorm / rmsnorm。",
          "手写 tiled GEMM。",
          "手写 causal attention。",
          "手写 top-k / top-p sampling。",
          "手写 LRU cache。",
          "手写 ring all-reduce 的伪代码。",
          "手写一个简单的 paged KV cache。",
        ],
      },
      {
        topic: "Project deep-dive",
        items: [
          "你的 kernel 比 baseline 快，是算法、带宽、还是测量误差？",
          "你的 mini engine 的 TTFT/TPOT 是多少？怎么测的？",
          "如果给你 10× GPU，你会怎么改？",
          "你遇到的最大 bug 是什么？怎么定位？",
          "如果重来一次，你会先做什么？",
        ],
      },
    ],
    companies: [
      {
        group: "推理框架 / 模型公司",
        names: ["DeepSeek", "字节 Seed / AML", "阿里 PAI / 通义", "腾讯混元", "百度", "蚂蚁", "月之暗面", "智谱", "MiniMax", "阶跃星辰", "硅基流动", "无问芯穹"],
      },
      {
        group: "芯片 / 算子",
        names: ["NVIDIA", "AMD", "华为昇腾", "寒武纪", "摩尔线程", "壁仞", "燧原", "昆仑芯", "海光"],
      },
      {
        group: "云 / 基础设施",
        names: ["阿里云", "腾讯云", "华为云", "火山引擎", "AWS", "Azure"],
      },
    ],
    sprint: [
      { week: "1", focus: "实验 0/1 + 简历 1.0", deliverable: "测试全绿 + 一页简历" },
      { week: "2", focus: "实验 2/3 + 作业 2", deliverable: "attention + paged KV" },
      { week: "3", focus: "项目 1 Phase 1/2", deliverable: "kernel zoo 雏形" },
      { week: "4", focus: "项目 1 报告 + 博客", deliverable: "roofline 报告 + 博客" },
      { week: "5", focus: "Mini Engine M1/M2", deliverable: "paged KV + batching" },
      { week: "6", focus: "Mini Engine M3/M4", deliverable: "chunked prefill + spec decode" },
      { week: "7", focus: "benchmark + 简历 2.0", deliverable: "吞吐/延迟曲线 + bullets" },
      { week: "8", focus: "模拟面试 + 投递", deliverable: "5 次 mock + 投递记录" },
    ],
    mock: [
      "45 分钟：10 分钟项目 + 20 分钟八股/手撕 + 15 分钟系统设计。",
      "每次录音；复盘时只记 3 个问题：哪里卡住、为什么卡、下次怎么改。",
      "每两周一次；面试前一个月每周一次。",
      "把 Project 1/2/3 各讲 3 分钟，录下来，检查是否有数字、trade-off、反思。",
    ],
  },

  staff: [
    {
      avatar: "🧑‍💻",
      name: "你",
      role: "讲师 / 学生",
      bio: "这门课没有正式老师。写代码、做 checkoff、改课程内容的人都是你。你也是唯一能决定“学没学会”的人。",
    },
    {
      avatar: "🤖",
      name: "AI 助教",
      role: "AI 助教 / Review",
      bio: "可以解释代码、画调用图、出题、review 你的 kernel。不要让它替你写完整实现；先自己写，再让它挑毛病。",
    },
    {
      avatar: "🧑‍🤝‍🧑",
      name: "学习伙伴",
      role: "学习伙伴 / 模拟面试",
      bio: "找一个也在准备 infra 秋招的人，每周一次 60 分钟：互相讲 checkpoint、pair debug、模拟面试。没有伙伴就录音自讲。",
    },
  ],

  contact: `
    <p class="lede">这门课是本地、自 paced 的。没有 Piazza，也没有 office hours 队列；但有三种可靠的“求助方式”。</p>
    <div class="contact-grid">
      <div class="card">
        <div class="card-kicker">1 · Ask a good question</div>
        <h3>最小复现 + 期望/实际</h3>
        <p>不要问“我的代码不 work”。写清楚：你期望什么、实际什么、最小复现、你试过什么。把这段贴给 AI TA，或者写进 notes/。</p>
      </div>
      <div class="card">
        <div class="card-kicker">2 · Checkoff</div>
        <h3>讲给一个朋友听</h3>
        <p>每个 Lab 做完，回答 README 最后的 3 个问题。讲不出来就回去重做；这比刷更多题有用。</p>
      </div>
      <div class="card">
        <div class="card-kicker">3 · Open an issue</div>
        <h3>在仓库里留痕</h3>
        <p>发现课程 bug、链接失效、题目歧义？在仓库开 issue，或者直接改 data.js / README，提一个 PR。开源贡献从课程本身开始。</p>
      </div>
    </div>
    <div class="callout">
      <p><strong>Office hours（自学版）</strong>：每周固定两个 30 分钟 block，关掉手机，只做一件事：把本周没搞懂的问题写成 3 行，然后逐个查/问。不要“等有空”。</p>
    </div>
    <h2>提问模板</h2>
    <pre><code>我在做：Lab 0 / Q1
期望：vector_add([1,2],[3,4]) -> [4,6]
实际：返回 [4,6,0]
最小复现：...
我试过：...
我猜原因：...</code></pre>
    <h2>常用链接</h2>
    <div class="chapter-links">
      <a class="chip" href="../../START_HERE.md">START_HERE.md</a>
      <a class="chip" href="../README.md">Course README</a>
      <a class="chip" href="../../doc/study_plan_zh.md">46-week roadmap</a>
      <a class="chip" href="https://github.com/deepseek-ai/DeepInfer/issues">DeepInfer issues</a>
    </div>
  `,
};
