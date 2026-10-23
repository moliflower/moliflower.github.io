#!/usr/bin/env python3
"""Build nicely formatted HTML pages for course notes.

Source of truth is the Markdown under course/. This script converts it into
static HTML under course/site/notes/, so the local site works from file://
without fetch(), a server, or a Markdown library.

Usage:
    python course/tools/build_notes.py
"""

from __future__ import annotations

import html
import posixpath
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "course" / "site"
OUT = SITE / "notes"

DOCS = [
    dict(src="course/TEXTBOOK.md", out="textbook.html", category="课程指南", label="教材导航", key=None, title_zh="教材：从零基础到可验证的推理算子"),
    dict(src="course/ASSESSMENT.md", out="assessment.html", category="课程指南", label="掌握度验收", key=None, title_zh="掌握度与已发布单元评分"),
    dict(src="course/foundations/README.md", out="foundations.html", category="Python 起步", label="起步指南", key="foundations-guide", title_zh="Python 起步：从第一行代码到可测试的函数"),
    dict(src="course/foundations/00-expressions.md", out="foundations00.html", category="Python 起步", label="起步 0", key="foundations-q1", title_zh="起步 0：让一个函数把结果交回来"),
    dict(src="course/foundations/01-control.md", out="foundations01.html", category="Python 起步", label="起步 1", key="foundations-q2", title_zh="起步 1：让边界条件决定分支"),
    dict(src="course/foundations/02-sequences.md", out="foundations02.html", category="Python 起步", label="起步 2", key="foundations-q3", title_zh="起步 2：遍历列表，构造自己的输出"),
    dict(src="course/foundations/03-composition.md", out="foundations03.html", category="Python 起步", label="起步 3", key="foundations-q4", title_zh="起步 3：用状态表示已经完成的工作"),
    dict(src="course/prep/README.md", out="prep.html", category="预备", label="预备单元", key="prep-guide", title_zh="预备单元：把看得懂变成能定位错误"),
    dict(src="course/prep/00-run.md", out="prep00.html", category="预备", label="预备 0", key="prep-q0", title_zh="预备 0：运行程序，读懂第一条反馈"),
    dict(src="course/prep/01-debug.md", out="prep01.html", category="预备", label="预备 1", key="prep-q1", title_zh="预备 1：用最小复现修复最大值"),
    dict(src="course/prep/02-matrix.md", out="prep02.html", category="预备", label="预备 2", key="prep-q2", title_zh="预备 2：把矩阵手算连接到代码"),
    dict(src="course/prep/03-tests.md", out="prep03.html", category="预备", label="预备 3", key="prep-q3", title_zh="预备 3：让测试帮助你"),
    dict(src="course/lectures/lec01_gpu.md", out="lec01_gpu.html", category="进阶", label="第一讲选读", key=None, title_zh="第一讲进阶：硬件吞吐与 CUDA 阅读"),
    # lectures
    dict(src="course/lectures/lec01.md", out="lec01.html", category="讲义", label="讲义 1", key="w1-l1", title_zh="第 1 讲：从 Python 建立 GPU 执行模型"),
    dict(src="course/lectures/lec02.md", out="lec02.html", category="讲义", label="讲义 2", key="w1-l2", title_zh="第 2 讲：合并访问、共享内存与 Occupancy"),
    dict(src="course/lectures/lec03.md", out="lec03.html", category="讲义", label="讲义 3", key="w2-l3", title_zh="第 3 讲：Reduction、Softmax 与 RMSNorm"),
    dict(src="course/lectures/lec04.md", out="lec04.html", category="讲义", label="讲义 4", key="w2-l4", title_zh="第 4 讲：Tiling、GEMM 与 Roofline"),
    dict(src="course/lectures/lec05.md", out="lec05.html", category="讲义", label="讲义 5", key="w3-l5", title_zh="第 5 讲：从零实现 Attention"),
    dict(src="course/lectures/lec06.md", out="lec06.html", category="讲义", label="讲义 6", key="w3-l6", title_zh="第 6 讲：FlashAttention 与 Decode Attention"),
    dict(src="course/lectures/lec07.md", out="lec07.html", category="讲义", label="讲义 7", key="w4-l7", title_zh="第 7 讲：KV Cache 基础"),
    dict(src="course/lectures/lec08.md", out="lec08.html", category="讲义", label="讲义 8", key="w4-l8", title_zh="第 8 讲：量化（FP8 / FP4）"),
    # labs
    dict(src="course/labs/lab00/README.md", out="lab00.html", category="实验", label="实验 0", key="w1-lab0", title_zh="实验 0：GPU 心智模型"),
    dict(src="course/labs/lab01/README.md", out="lab01.html", category="实验", label="实验 1", key="w2-lab1", title_zh="实验 1：Reduction、Softmax 与 RMSNorm"),
    dict(src="course/labs/lab02/README.md", out="lab02.html", category="实验", label="实验 2", key="w3-lab2", title_zh="实验 2：从零实现 Attention"),
    dict(src="course/labs/lab03/README.md", out="lab03.html", category="实验", label="实验 3", key="w4-lab3", title_zh="实验 3：KV Cache（连续 + Paged）"),
    # homework
    dict(src="course/hw/hw01/README.md", out="hw01.html", category="作业", label="作业 1", key="w1-hw1", title_zh="作业 1：Roofline、KV Cache 与量化"),
    dict(src="course/hw/hw02/README.md", out="hw02.html", category="作业", label="作业 2", key="w2-hw2", title_zh="作业 2：Attention 与 KV Cache 数学"),
    # project
    dict(src="course/projects/project1/README.md", out="project1.html", category="项目", label="项目 1", key="w4-p1due", title_zh="项目 1：Kernel Zoo"),
    dict(src="course/projects/project1/report_template.md", out="project1-report.html", category="报告模板", label="项目报告", key=None, title_zh="Project1 报告模板"),
    dict(src="course/hw/hw01/REPORT_TEMPLATE.md", out="hw01-report.html", category="报告模板", label="HW1 报告", key=None, title_zh="HW1 推导与证据模板"),
    dict(src="course/hw/hw02/REPORT_TEMPLATE.md", out="hw02-report.html", category="报告模板", label="HW2 报告", key=None, title_zh="HW2 推导与证据模板"),
    # quizzes
    dict(src="course/quizzes/quiz01.md", out="quiz01.html", category="小测", label="小测 1", key="w2-quiz1", title_zh="小测 1：GPU 模型、Roofline、Softmax 与索引"),
    dict(src="course/quizzes/quiz02.md", out="quiz02.html", category="小测", label="小测 2", key="w4-quiz2", title_zh="小测 2：Attention、KV Cache、量化"),
    dict(src="course/lectures/lec09.md", out="lec09.html", category="讲义", label="讲义 9", key="w5-l9", title_zh="第 9 讲：分页映射与共享块池"),
    dict(src="course/lectures/lec10.md", out="lec10.html", category="讲义", label="讲义 10", key="w5-l10", title_zh="第 10 讲：前缀命中、共享与淘汰"),
    dict(src="course/labs/lab04/README.md", out="lab04.html", category="实验", label="实验 4", key="w5-lab4", title_zh="Lab4：有限块池与共享所有权"),
    dict(src="course/hw/hw05/README.md", out="hw05.html", category="作业", label="作业 5", key="w5-hw5", title_zh="HW5：前缀命中与容量推导"),
    dict(src="course/projects/project2/README.md", out="project2.html", category="项目", label="项目 2 Phase1", key="w5-p2phase1", title_zh="Project2 Phase1：请求生命周期"),
    dict(src="course/projects/project2/report_template.md", out="project2-report.html", category="报告模板", label="项目 2 报告", key=None, title_zh="Project2 Phase1 报告模板"),
]

for doc in DOCS:
    doc["src_path"] = ROOT / doc["src"]
    if not doc["src_path"].is_file():
        raise FileNotFoundError(doc["src_path"])

SRC_TO_OUT = {doc["src"]: doc["out"] for doc in DOCS}
CURRENT_SRC_DIR = ""


def rewrite_url(url: str) -> str:
    """Rewrite a Markdown link so it works from site/notes/<page>.html.

    - links to another generated note -> notes/<page>.html
    - links to a directory with README.md -> its generated page
    - other repo files -> correct relative path from site/notes/
    """
    if not url or url.startswith(("http://", "https://", "#", "mailto:")):
        return url

    base, _, fragment = url.partition("#")
    if not base:
        return url

    target = posixpath.normpath(posixpath.join(CURRENT_SRC_DIR, base))
    target = target.rstrip("/")

    out = SRC_TO_OUT.get(target)
    if out is None:
        out = SRC_TO_OUT.get(target + "/README.md")
    if out is not None:
        result = out
    else:
        result = posixpath.relpath(target, "course/site/notes")

    return result + (("#" + fragment) if fragment else "")

# prev/next within the same category
_groups = defaultdict(list)
for doc in DOCS:
    _groups[doc["category"]].append(doc)
for group in _groups.values():
    for i, doc in enumerate(group):
        doc["prev"] = group[i - 1] if i > 0 else None
        doc["next"] = group[i + 1] if i + 1 < len(group) else None


# ---------------------------------------------------------------------------
# Markdown (a deliberately small subset, enough for this course)
# ---------------------------------------------------------------------------
def inline(text: str) -> str:
    """Inline Markdown -> HTML. Code spans and links are protected first."""
    placeholders: list[str] = []

    def stash(fragment: str) -> str:
        placeholders.append(fragment)
        return f"\x00{len(placeholders) - 1}\x00"

    # images
    text = re.sub(
        r"!\[([^\]]*)\]\(([^)\s]+)\)",
        lambda m: stash(
            f'<img src="{html.escape(rewrite_url(m.group(2)), quote=True)}" alt="{html.escape(m.group(1), quote=True)}">'
        ),
        text,
    )
    # links
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: stash(
            f'<a href="{html.escape(rewrite_url(m.group(2)), quote=True)}">{html.escape(m.group(1), quote=False)}</a>'
        ),
        text,
    )
    # code spans
    text = re.sub(
        r"`([^`\n]+)`",
        lambda m: stash(f"<code>{html.escape(m.group(1), quote=False)}</code>"),
        text,
    )
    # autolinks <https://...>
    text = re.sub(
        r"<(https?://[^>\s]+)>",
        lambda m: stash(
            f'<a href="{html.escape(m.group(1), quote=True)}">{html.escape(m.group(1), quote=False)}</a>'
        ),
        text,
    )

    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)

    for i, fragment in enumerate(placeholders):
        text = text.replace(f"\x00{i}\x00", fragment)
    return text


def split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [cell.strip() for cell in s.split("|")]


def render_table(lines: list[str]) -> str:
    rows = [split_row(line) for line in lines]
    header = rows[0]
    body = rows[2:] if len(rows) >= 2 else []
    head = "".join(f"<th>{inline(cell)}</th>" for cell in header)
    out = [f"<table><thead><tr>{head}</tr></thead><tbody>"]
    for row in body:
        cells = "".join(f"<td>{inline(cell)}</td>" for cell in row)
        out.append(f"<tr>{cells}</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def is_block_start(lines: list[str], i: int) -> bool:
    line = lines[i]
    stripped = line.strip()
    if not stripped:
        return True
    if stripped == "<details>":
        return True
    if stripped.startswith(("```", ">")):
        return True
    if re.match(r"^#{1,4}\s+", stripped):
        return True
    if re.match(r"^-{3,}$", stripped):
        return True
    if re.match(r"^\s*[-*]\s+", line) or re.match(r"^\s*\d+\.\s+", line):
        return True
    if stripped.startswith("|") and i + 1 < len(lines):
        nxt = lines[i + 1].strip()
        if nxt.startswith("|") and "---" in nxt:
            return True
    return False


def convert_blocks(lines: list[str], headings=None) -> tuple[str, list[tuple[int, str, str]]]:
    body: list[str] = []
    if headings is None:
        headings = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        # Only allow these two structural tags, never arbitrary raw HTML.
        # Contents still pass through the escaping Markdown renderer.
        if line.strip() == "<details>":
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            summary = re.fullmatch(r"<summary>(.*?)</summary>", lines[i].strip()) if i < len(lines) else None
            if not summary:
                raise ValueError("<details> 后需要独立一行的 <summary>标题</summary>")
            i += 1
            buf = []
            depth, fenced = 1, False
            while i < len(lines):
                stripped = lines[i].strip()
                if stripped.startswith("```"):
                    fenced = not fenced
                if not fenced:
                    if stripped == "<details>":
                        depth += 1
                    elif stripped == "</details>":
                        depth -= 1
                        if depth == 0:
                            break
                buf.append(lines[i])
                i += 1
            if depth:
                raise ValueError("<details> 缺少 </details>")
            inner, _ = convert_blocks(buf, headings)
            body.append(f'<details class="study-hint"><summary>{inline(summary.group(1))}</summary>{inner}</details>')
            i += 1
            continue

        # fenced code
        if line.lstrip().startswith("```"):
            lang = line.lstrip()[3:].strip()
            i += 1
            buf: list[str] = []
            while i < len(lines) and not lines[i].lstrip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1  # closing fence
            cls = f' class="language-{html.escape(lang)}"' if lang else ""
            body.append(f"<pre><code{cls}>{html.escape(chr(10).join(buf), quote=False)}</code></pre>")
            continue

        # heading
        m = re.match(r"^(#{1,4})\s+(.*)$", line.strip())
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            hid = f"h-{len(headings) + 1}"
            headings.append((level, text, hid))
            body.append(f'<h{level} id="{hid}">{inline(text)}</h{level}>')
            i += 1
            continue

        # horizontal rule
        if re.match(r"^-{3,}$", line.strip()):
            body.append("<hr>")
            i += 1
            continue

        # table
        if line.lstrip().startswith("|") and i + 1 < len(lines) and "---" in lines[i + 1]:
            table_lines: list[str] = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            body.append(render_table(table_lines))
            continue

        # blockquote
        if line.lstrip().startswith(">"):
            buf: list[str] = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            if any(re.match(r"^\s*[-*]\s+", item) for item in buf):
                inner, _ = convert_blocks(buf, headings)
                body.append(f"<blockquote>{inner}</blockquote>")
            else:
                # Metadata blocks such as "课前 / 课后 / 一句话" should keep
                # their line breaks instead of collapsing into one sentence.
                text = "<br>".join(inline(item) for item in buf if item.strip())
                body.append(f"<blockquote><p>{text}</p></blockquote>")
            continue

        # lists
        if re.match(r"^\s*[-*]\s+", line) or re.match(r"^\s*\d+\.\s+", line):
            ordered = bool(re.match(r"^\s*\d+\.\s+", line))
            items: list[str] = []
            while i < len(lines):
                cur = lines[i]
                m = (
                    re.match(r"^\s*\d+\.\s+(.*)$", cur)
                    if ordered
                    else re.match(r"^\s*[-*]\s+(.*)$", cur)
                )
                if not m:
                    break
                items.append(inline(m.group(1).strip()))
                i += 1
            tag = "ol" if ordered else "ul"
            start = re.match(r"^\s*(\d+)\.", line) if ordered else None
            attr = f' start="{start.group(1)}"' if start and start.group(1) != "1" else ""
            body.append(f"<{tag}{attr}>" + "".join(f"<li>{item}</li>" for item in items) + f"</{tag}>")
            continue

        # paragraph
        buf: list[str] = []
        while i < len(lines) and lines[i].strip() and not is_block_start(lines, i):
            buf.append(lines[i].strip())
            i += 1
        if buf:
            body.append(f"<p>{inline(' '.join(buf))}</p>")

    return "\n".join(body), headings


def parse_markdown(md: str, fallback_title: str) -> tuple[str, str, list[tuple[int, str, str]]]:
    lines = md.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    title = fallback_title
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        lines = lines[1:]
    body, headings = convert_blocks(lines)
    return title, body, headings


# ---------------------------------------------------------------------------
# Page template
# ---------------------------------------------------------------------------
def build_page(doc: dict, body: str, headings: list[tuple[int, str, str]]) -> str:
    title = doc["title_zh"] or doc["parsed_title"]
    # src is relative to the repo root; from site/notes/ it is ../../<src>
    source_rel = "../../" + doc["src"].removeprefix("course/")

    toc_items = [
        (level, text, hid)
        for level, text, hid in headings
        if level in (2, 3)
    ]
    if toc_items:
        toc = "".join(
            f'<a class="toc-level-{level}" href="#{hid}">{inline(text)}</a>'
            for level, text, hid in toc_items
        )
    else:
        toc = '<p class="muted small">本页没有小节。</p>'

    char_count = len(re.sub(r"<[^>]+>", "", body))
    minutes = max(1, round(char_count / 400))

    actions = ""
    if doc.get("key"):
        actions = (
            '<div class="doc-actions">'
            f'<button id="doc-done" class="doc-done" type="button" data-key="{doc["key"]}">标记完成</button>'
            '<span id="doc-done-hint" class="doc-hint"></span>'
            "</div>"
        )

    nav_parts = []
    if doc.get("prev"):
        p = doc["prev"]
        nav_parts.append(
            f'<a class="doc-nav-card" href="{p["out"]}"><span>← 上一篇</span><strong>{html.escape(p["title_zh"] or p["label"])}</strong></a>'
        )
    else:
        nav_parts.append('<span class="doc-nav-card disabled"><span>← 上一篇</span><strong>没有了</strong></span>')
    if doc.get("next"):
        n = doc["next"]
        nav_parts.append(
            f'<a class="doc-nav-card" href="{n["out"]}"><span>下一篇 →</span><strong>{html.escape(n["title_zh"] or n["label"])}</strong></a>'
        )
    else:
        nav_parts.append('<span class="doc-nav-card disabled"><span>下一篇 →</span><strong>没有了</strong></span>')

    return f"""<!doctype html>
<html lang="zh-CN" data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} · DeepInfer 101</title>
  <script>
    try {{
      document.documentElement.setAttribute("data-theme", localStorage.getItem("di101-theme") || "dark");
    }} catch (_) {{}}
  </script>
  <link rel="stylesheet" href="../styles.css">
</head>
<body class="doc-body">
  <header class="doc-topbar">
    <a class="doc-back" href="../../../#/calendar">← 返回周历</a>
    <span class="doc-category">{html.escape(doc["category"])}</span>
    <a class="doc-back" href="../../../#/videos">视频</a>
    <span class="doc-brand">DeepInfer 101</span>
    <button id="doc-theme" class="theme-btn" type="button">切换主题</button>
  </header>

  <main class="doc-main">
    <div class="doc-hero">
      <div class="eyebrow">{html.escape(doc["label"])}</div>
      <h1>{html.escape(title)}</h1>
      <div class="doc-meta">
        <span>纯阅读约 {minutes} 分钟 · 动手另计</span>
        <span>源文件：<a href="{html.escape(source_rel)}">Markdown</a></span>
        <span>本地渲染版 · 无需联网</span>
      </div>
      {actions}
    </div>

    <div class="doc-layout">
      <aside class="doc-toc">
        <h3>本页目录</h3>
        {toc}
      </aside>
      <article class="doc-content">
        {body}
      </article>
    </div>

    <nav class="doc-nav">
      {''.join(nav_parts)}
    </nav>
  </main>

  <script src="../storage.js"></script>
  <script src="../doc.js"></script>
</body>
</html>
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    global CURRENT_SRC_DIR
    for doc in DOCS:
        CURRENT_SRC_DIR = posixpath.dirname(doc["src"])
        md = doc["src_path"].read_text(encoding="utf-8")
        parsed_title, body, headings = parse_markdown(md, doc["label"])
        doc["parsed_title"] = parsed_title
        page = build_page(doc, body, headings)
        target = OUT / doc["out"]
        target.write_text(page, encoding="utf-8")
        print(f"wrote {target.relative_to(ROOT)} ({len(page)} bytes)")
    print(f"done: {len(DOCS)} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
