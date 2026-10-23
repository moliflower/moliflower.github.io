# DeepInfer 101 · 本地课程网站

这是一个**本地、CS61A 风格**的课程网站。不需要服务器、不需要构建、不需要安装依赖。

## 打开方式

双击 `index.html`，或者：

```bash
# Windows（也可以直接双击 open.bat）
start course/site/index.html

# macOS
open course/site/index.html

# Linux
xdg-open course/site/index.html
```

它可以直接用 `file://` 打开，因为所有内容都在 `data.js` 里，没有 `fetch()`。

## 页面

| 页面 | 内容 |
| :--- | :--- |
| **首页** | 课程介绍、公告、当前周、快速开始 |
| **周历** | 15 周计划，带讲义 / 实验 / 作业 / 项目 / 阅读 / 视频 badge |
| **练习中心** | 按先修顺序集中显示已发布 Lab、HW、Project checkpoints 与 Quiz |
| **课程大纲** | 形式、评分、协作、AI 政策、提交 |
| **教材** | 按先修组织的本地正文、学习目标、练习入口与后续发布状态 |
| **资源** | 课程、博客、论文、仓库、工具、B站 |
| **视频** | 精选 B 站视频：CUDA、FlashAttention、vLLM、MoE、投机解码、Profiling、TileLang、NCCL |
| **讨论课** | 每周 60 分钟的讨论 / checkoff 提纲 |
| **Infra 秋招** | 时间线、岗位地图、简历 bullet、面试题库、8 周冲刺 |
| **关于** | 你、AI 助教、学习伙伴 |
| **联系** | 如何提问、office hours、模板 |

## 功能

- **讲义/实验/作业渲染**：Markdown 自动生成排版好的 HTML，带目录、上一篇/下一篇、标记完成；
- **新手入口**：首页和周历提供Python 起步 → 调试预备 → 第一讲核心 → Lab0；提示和小测答案默认折叠；
- **视频页**：精选 B 站视频，按 CUDA / FlashAttention / vLLM / MoE / 投机解码 / Profiling / TileLang / NCCL 分组，写明“看哪几段”；
- **学习进度**：周历里的 checkbox 保存在 `localStorage`，sidebar 显示进度条；
- **主题**：暗色 / 亮色切换，保存在 `localStorage`；
- **搜索**：按主题过滤周次（attention、KV、MoE、roofline...）；
- **当前周**：根据 `meta.startDate` 自动计算（默认 `2026-10-26`）；
- **响应式**：手机端 sidebar 收起。

## 修改课程内容

所有内容都在 [`data.js`](data.js)：

- `meta`：标题、开始日期、简介
- `weeks`：周历；每项有 `type`、`label`、`title`、`link`、`key`、`soon`
- `textbook`：教材章节；`understanding` 描述本章要解决的问题，`links` 连接正文与练习
- `resources`：资源分组
- `syllabus`：大纲的 HTML 段落
- `sections`、`staff`、`contact`：讨论课、关于、联系

`app.js` 负责主站渲染，`doc.js` 负责讲义交互，二者使用 `storage.js` 保存进度和主题。新页面需要在 `nav`、`PAGE_LABELS` 和 `PAGES` 中注册；新教材需要同时更新 `build_notes.py` 的 DOCS 和周历/教材入口，运行检查核对链接。

进度是自我记录，不是自动测试结果。共享存储会验证 JSON 结构，并在焦点恢复、返回页面和其他标签页修改时同步。保存被浏览器禁止时会明确提示。`file://` 的跨文件存储行为由浏览器决定；需要可靠共享时建议在仓库根目录运行 `python -m http.server 8000 --bind 127.0.0.1`，打开 `http://127.0.0.1:8000/course/site/index.html`，始终使用同一地址。不同 origin 的旧进度不会自动迁移。

## 重建讲义 HTML

`course/site/notes/` 下的页面是由 Markdown 自动生成的。改完讲义/实验/作业的 `.md` 后，运行：

```bash
cd /path/to/DeepInfer
python course/tools/build_notes.py
```

它会按 `build_notes.py` 的 DOCS 清单重建全部页面，包括起步四节、教材索引与评分标准。生成器是纯 Python 标准库，不需要安装 Markdown 包。支持独立行的 `<details>` / `<summary>`，用于分级提示；不支持任意原始 HTML。

## 说明

- 讲义、预备课、实验、作业的链接指向生成的 HTML；每页仍提供 Markdown 源文件链接。
- 网站刻意不依赖 CDN；断网也能用。
- 课程正文在 [`../README.md`](../README.md)，长期路线图在 [`../../doc/study_plan_zh.md`](../../doc/study_plan_zh.md)。

## 文件结构

```
course/site/
├── index.html     # 外壳：sidebar、topbar、内容挂载点
├── styles.css     # CS61A 风格的暗色/亮色主题
├── data.js        # 所有课程内容
├── app.js         # 路由 + 渲染 + 进度/主题/搜索
├── open.bat       # Windows 双击打开
└── README.md      # 本文件
```

## 可选：部署

如果想分享：

- 发布网页时需一起保留题目源文件、Markdown 与仓库对照文件，或重写这些下载链接；单独上传 `course/site/` 会让指向站外目录的源码链接失效。
- 本地共享进度推荐从仓库根目录提供 HTTP 服务，使用上面的回环地址命令。

> 生成页面不需要浏览器读取 Markdown，也不依赖 CDN。修改正文后运行生成器即可。
