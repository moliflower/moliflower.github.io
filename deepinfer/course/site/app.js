/* DeepInfer 101 — local course site.
 *
 * No frameworks, no build step, no network. Open index.html directly.
 * All content lives in data.js; this file only renders it.
 */

(function () {
  "use strict";

  const C = window.COURSE;
  const storage = window.CourseStorage;
  const courseBase = document.documentElement.getAttribute("data-course-base") || "";
  if (!C) throw new Error("data.js did not load");

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const state = {
    progress: loadProgress(),
    theme: loadTheme(),
  };

  const PAGE_LABELS = {
    home: "首页",
    calendar: "周历",
    practice: "练习中心",
    syllabus: "课程大纲",
    textbook: "教材",
    resources: "资源",
    videos: "视频",
    sections: "讨论课",
    career: "Infra 秋招",
    staff: "关于",
    contact: "联系",
  };

  // ------------------------------------------------------------------
  // Storage (file:// friendly: localStorage can throw in some browsers)
  // ------------------------------------------------------------------
  function loadProgress() {
    return storage.progress();
  }

  function loadTheme() {
    return storage.theme();
  }

  function saveTheme(theme) {
    storage.theme(theme);
  }

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------
  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function escapeAttr(value) {
    return escapeHtml(value).replaceAll("`", "&#96;");
  }

  function resolveCourseLinks(markup) {
    if (!courseBase) return markup;
    return markup.replace(/(href|src)="([^"]*)"/g, (match, attribute, url) => {
      if (!url || /^(?:[a-z][a-z0-9+.-]*:|\/|#)/i.test(url)) return match;
      return `${attribute}="${escapeAttr(courseBase)}${url}"`;
    });
  }

  function getRoute() {
    const raw = (location.hash || "").replace(/^#\/?/, "").trim();
    if (!raw) return "home";
    return PAGE_LABELS[raw] ? raw : "home";
  }

  function currentWeek() {
    const start = new Date(C.meta.startDate + "T00:00:00");
    const now = new Date();
    const week = Math.floor((now - start) / (7 * 24 * 3600 * 1000)) + 1;
    return Math.max(0, Math.min(15, week));
  }

  function allProgressKeys() {
    const keys = [];
    for (const week of C.weeks) {
      for (const item of week.items) {
        if (item.key && !item.soon) keys.push(item.key);
      }
    }
    return [...new Set(keys)];
  }

  function updateProgress() {
    const keys = allProgressKeys();
    const done = keys.filter((key) => state.progress[key]).length;
    const pct = keys.length ? Math.round((done / keys.length) * 100) : 0;
    const text = $("#progress-text");
    const fill = $("#progress-fill");
    if (text) text.textContent = `${pct}% (${done}/${keys.length})`;
    if (fill) fill.style.width = `${pct}%`;
    $$(".item[data-key-row]").forEach((row) => {
      const key = row.getAttribute("data-key-row");
      row.classList.toggle("done", !!(key && state.progress[key]));
    });
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    const btn = $("#theme-toggle");
    if (btn) btn.textContent = theme === "dark" ? "☀️ 亮色" : "🌙 暗色";
  }

  // ------------------------------------------------------------------
  // Rendering: shared pieces
  // ------------------------------------------------------------------
  function renderNav() {
    const nav = $("#nav");
    if (!nav) return;
    nav.innerHTML = C.nav
      .map(
        (item) =>
          `<a href="#/${item.id}" data-nav="${item.id}"><span class="nav-dot"></span>${escapeHtml(
            item.label
          )}</a>`
      )
      .join("");
  }

  function badge(type, label) {
    return `<span class="badge ${escapeAttr(type)}">${escapeHtml(label)}</span>`;
  }

  function renderItem(item) {
    const key = item.key || "";
    const checkbox = key
      ? `<input class="item-check" type="checkbox" data-key="${escapeAttr(
          key
        )}" aria-label="标记完成：${escapeAttr(item.title)}">`
      : `<span></span>`;
    const label = item.label || item.type;
    const soon = item.soon ? ` ${badge("soon", "待发布")}` : "";
    const body = item.link
      ? `<a class="item-link" href="${escapeAttr(item.link)}">${escapeHtml(item.title)}</a>${soon}`
      : `<span class="item-link">${escapeHtml(item.title)}</span>${soon}`;
    return `<div class="item" data-key-row="${escapeAttr(key)}">${checkbox}${badge(
      item.type,
      label
    )}<div>${body}</div></div>`;
  }

  function renderSide(item) {
    const label = item.type === "video" ? "视频" : "阅读";
    return `<div class="side-item">${badge(item.type, label)}<a href="${escapeAttr(
      item.link
    )}">${escapeHtml(item.title)}</a></div>`;
  }

  function renderWeek(week) {
    const isCurrent = week.n === currentWeek();
    const items = week.items.map(renderItem).join("");
    const right = (week.right || []).map(renderSide).join("");
    const search = [
      week.theme,
      ...week.items.map((item) => `${item.label || ""} ${item.title}`),
      ...(week.right || []).map((item) => item.title),
    ]
      .join(" ")
      .toLowerCase();
    return `<section class="week-card${isCurrent ? " current" : ""}" data-week="${
      week.n
    }" data-search="${escapeAttr(search)}">
      <div class="week-head">
        <div>
          <div class="week-kicker">${week.n === 0 ? "入门预备 · 自选节奏" : `第 ${week.n} 周${isCurrent ? " · 当前" : ""}`}</div>
          <h2>${escapeHtml(week.theme)}</h2>
        </div>
        <div class="week-dates">${escapeHtml(week.dates)}</div>
      </div>
      <div class="week-grid">
        <div class="week-main">${items}</div>
        <aside class="week-side">
          <div class="side-label">阅读 & 视频</div>
          ${right || '<div class="week-empty">材料待发布。</div>'}
        </aside>
      </div>
    </section>`;
  }

  // ------------------------------------------------------------------
  // Pages
  // ------------------------------------------------------------------
  function renderHome() {
    const week = currentWeek();
    const current = C.weeks.find((item) => item.n === week);
    const announcements = C.announcements
      .map(
        (item) => `<div class="announcement">
          <div class="date">${escapeHtml(item.date)}</div>
          <div class="body"><strong>${escapeHtml(item.title)}</strong> ${escapeHtml(
          item.body
        )} <a href="${escapeAttr(item.link)}">→</a></div>
        </div>`
      )
      .join("");

    return `
      <section class="hero">
        <div class="eyebrow">${escapeHtml(C.meta.term)}</div>
        <h1>${escapeHtml(C.meta.title)}</h1>
        <p class="lede">${escapeHtml(C.meta.description)}</p>
        <div class="hero-actions">
          <a class="btn primary" href="notes/foundations.html">从零基础开始 →</a>
          <a class="btn" href="notes/lec01.html">已会调试？进入第一讲 →</a>
          <a class="btn ghost" href="#/textbook">阅读教材</a>
        </div>
        <div class="hero-meta">
          <span>${escapeHtml(C.meta.release || "起步 + Unit1 已发布")}</span>
          <span>每周 12–15 小时</span>
          <span>第一单元不需要 GPU</span>
          <span>CS61A 风格 Lab + 项目</span>
        </div>
      </section>

      <div class="section-title"><h2>先走通入门路径</h2><a href="notes/foundations.html">入门指南 →</a></div>
      <div class="callout"><p><strong>当前建议：</strong>Python 起步 0–3 → 调试预备 0–3 → 第一讲核心 → Lab0 Q0–Q6。C++、CUDA、Tensor Core 暂不作为入门要求；先修复一个失败测试，再增加新知识。日历是路线，不是赶进度的截止日期。</p></div>
      ${C.weeks.find((item) => item.n === 0) ? renderWeek(C.weeks.find((item) => item.n === 0)) : ""}

      <div class="section-title"><h2>公告</h2><a href="#/calendar">周历 →</a></div>
      <div class="card">${announcements}</div>

      <div class="section-title"><h2>${week === 0 ? "开课前：按自己的节奏准备" : `当前周：第 ${week} 周 · ${escapeHtml(current ? current.theme : "")}`}</h2><a href="#/calendar">全部周次 →</a></div>
      ${current && week > 0 ? renderWeek(current) : ""}

      <div class="section-title"><h2>你会做出什么</h2><a href="#/syllabus">评分 →</a></div>
      <div class="grid-3">
        <div class="card"><div class="card-kicker">项目 1 · 第 1–4 周</div><h3>Kernel Zoo</h3>
          <p>Vector add、reduction、softmax、RMSNorm、tiled matmul；正确性 + roofline + 报告。</p>
          <p><a href="notes/project1.html">打开项目 →</a></p></div>
        <div class="card"><div class="card-kicker">项目 2 · Phase1 已发布</div><h3>Mini 推理引擎</h3>
          <p>先完成 FIFO 准入、等待、释放与状态快照；逐 token 调度与模型执行后续发布。</p>
          <p><a href="notes/project2.html">开始 Phase1 →</a></p></div>
        <div class="card"><div class="card-kicker">项目 3 · 待发布</div><h3>EPLB 模拟器</h3>
          <p>合成专家负载、greedy 重排、通信成本模型、imbalance 报告。</p>
          <p><a href="#/calendar">大纲 →</a></p></div>
      </div>

      <div class="section-title"><h2>课程怎么运作</h2><a href="#/syllabus">完整大纲 →</a></div>
      <div class="grid-2">
        <div class="card"><h3>讲义</h3><p>30–45 分钟阅读 + 视频。先建立心智模型，不要背 API。</p></div>
         <div class="card"><h3>实验</h3><p>从示例、填空走到独立实现和修错。按题测试、逐层提示，可分多次完成。</p></div>
        <div class="card"><h3>作业</h3><p>3–5 小时的数学 + 小函数。“为什么”在这里。</p></div>
        <div class="card"><h3>项目</h3><p>2–3 周，分 phase + checkpoint。把多个概念组合成一个能展示的作品。</p></div>
      </div>

      <div class="callout"><p><a class="btn primary" href="../../deepinfer-101-course.zip" download>下载课程代码（含参考答案）</a></p><p>解压后在文件夹内运行下方命令。网页用于阅读和记录进度，Python 练习在电脑本地完成。进度保存在当前浏览器，不会跨设备自动同步。</p></div><div class="section-title"><h2>快速开始</h2><a href="../../START_HERE.md">从这里开始 →</a></div>
      <div class="card">
        <pre><code># 在 DeepInfer 根目录运行；WSL 若无 python，使用 python3
python course/foundations/demo.py
python course/tools/grade.py foundations --question q0

# 开始修第一题；失败时只看这一题
python course/tools/grade.py foundations --question q1 --failfast
python course/tools/grade.py foundations --question q1 --hint 1
</code></pre>
        <p class="small muted">第一单元不需要安装任何包。如果你有 GPU，Project 1 有可选的 TileLang 轨道。</p>
      </div>

      <div class="callout">
        <p><strong>为什么不需要 GPU？</strong> 核心轨道是纯 Python 的 SIMT 模拟器 + cost model。没有 B200，也能学 coalescing、tiling、roofline、KV cache 和调度。等你有 GPU，再把 kernel 移植到 TileLang——理论是相通的。</p>
      </div>
    `;
  }

  function renderCalendar() {
    const legend = [
      ["lecture", "讲义"],
      ["lab", "实验"],
      ["homework", "作业"],
      ["project", "项目"],
      ["quiz", "小测"],
      ["reading", "阅读"],
      ["video", "视频"],
    ]
      .map(([type, label]) => badge(type, label))
      .join(" ");
    return `
      <div class="page-head">
        <div class="eyebrow">15 周计划</div>
        <h1>周历</h1>
        <p>零基础先做 Python 起步，再做调试预备，按知识依赖推进。HW1/2 调到先修齐备的第 4 周；第一周只做第一讲与 Lab0 核心。完成情况保存在本地进度里（不会上传）。</p>
        <div class="calendar-tools">
          <input id="calendar-search" class="search" type="search" aria-label="搜索周历" placeholder="过滤：attention、KV、MoE、roofline...">
          <button id="jump-current" class="btn small">跳到当前周</button>
        </div>
        <div class="small muted">${legend}</div>
      </div>
      <div id="weeks">${C.weeks.map(renderWeek).join("")}</div>
      <p id="search-empty" class="callout hidden" role="status">没有匹配的周次，请换个关键词。</p>
    `;
  }

  function renderPractice() {
    const groups = C.weeks.map((week) => {
      const items = week.items.filter((item) => {
        return !item.soon && item.link && ["lab", "homework", "project", "quiz"].includes(item.type);
      });
      return items.length ? `<section class="card"><h2>${week.n === 0 ? "起步与预备" : `第 ${week.n} 周`} · ${escapeHtml(week.theme)}</h2>${items.map(renderItem).join("")}</section>` : "";
    }).join("");
    return `<div class="page-head"><div class="eyebrow">Lab · HW · Project</div><h1>练习中心</h1><p>按先修顺序找到已发布练习。打开题面查看实现文件、单题命令和验收要求。</p></div>
      <div class="callout"><p>完成标记是自我记录，不是自动评分。每题按「预测 → 实现 → 测试 → 解释」推进；项目预览完成不代表整个项目完成。</p><p><a href="notes/assessment.html">查看评分与验收</a> · <a href="#/textbook">查看先修教材</a></p></div>
      <div class="practice-list">${groups}</div>`;
  }

  function renderSyllabus() {
    const sections = (C.syllabus || [])
      .map((section) => `<h2>${escapeHtml(section.title)}</h2>${section.html}`)
      .join("");
    return `
      <div class="page-head">
        <div class="eyebrow">课程政策</div>
        <h1>课程大纲</h1>
        <p>课程形式、评分、协作、AI 使用、提交方式。先读这一页，再开始实验 0。</p>
      </div>
      <div class="prose">${sections}</div>
    `;
  }

  function renderTextbook() {
    const toc = C.textbook
      .map(
        (chapter) =>
          `<button data-scroll="${escapeAttr(chapter.id)}">${escapeHtml(
            chapter.num
          )} · ${escapeHtml(chapter.title)}</button>`
      )
      .join("");
    const chapters = C.textbook
      .map((chapter) => {
        const points = (chapter.keyPoints || [])
          .map((point) => `<li>${escapeHtml(point)}</li>`)
          .join("");
        const links = (chapter.links || [])
          .map(
            (link) =>
              `<a class="chip" href="${escapeAttr(link.url)}">${escapeHtml(
                link.label
              )} ↗</a>`
          )
          .join("");
        return `<article class="chapter" id="${escapeAttr(chapter.id)}">
          <div class="chapter-head"><span class="chapter-num">${escapeHtml(
            chapter.num
          )}</span><h2>${escapeHtml(chapter.title)}</h2></div>
          <p class="chapter-intro">${escapeHtml(chapter.intro)}</p>
          <div class="understanding"><h3>本章要解决的问题</h3><p>${escapeHtml(
            chapter.understanding
          )}</p></div>
          ${points ? `<ul class="key-points">${points}</ul>` : ""}
          ${chapter.formula ? `<div class="formula">${escapeHtml(chapter.formula)}</div>` : ""}
          <div class="chapter-links">${links}</div>
        </article>`;
      })
      .join("");
    return `
      <div class="page-head">
        <div class="eyebrow">教材</div>
        <h1>教材：从第一行 Python 到推理算子</h1>
        <p>按先修阅读本地正文，先手算、再编码、再解释。每章直接连接练习；外部资源留作扩展阅读。<a href="notes/textbook.html">完整教材路线与术语表 →</a></p>
      </div>
      <div class="textbook-layout">
        <aside class="toc"><h3>章节目录</h3>${toc}</aside>
        <div>${chapters}</div>
      </div>
    `;
  }

  function renderResources() {
    const groups = C.resources
      .map(
        (group) => `<section class="resource-group">
          <h2>${escapeHtml(group.title)}</h2>
          <p>${escapeHtml(group.intro)}</p>
          <div class="resource-grid">
            ${group.items
              .map(
                (item) => `<a class="resource" href="${escapeAttr(item.url)}">
                  <span class="r-tag">${escapeHtml(item.tag)}</span>
                  <span class="r-title">${escapeHtml(item.title)}</span>
                  <span class="r-desc">${escapeHtml(item.desc)}</span>
                </a>`
              )
              .join("")}
          </div>
        </section>`
      )
      .join("");
    return `
      <div class="page-head">
        <div class="eyebrow">资料库</div>
        <h1>资源</h1>
        <p>课程、博客、论文、仓库、工具、B站搜索。原则：一个主题最多看两个来源，剩下时间去写代码。</p>
      </div>
      ${groups}
    `;
  }

  function renderVideos() {
    const groups = C.videos
      .map(
        (group) => `<section class="video-group">
          <h2>${escapeHtml(group.title)}</h2>
          <p>${escapeHtml(group.intro)}</p>
          <div class="video-grid">
            ${group.items
              .map(
                (item) => `<a class="video-card" href="${escapeAttr(
                  item.url
                )}" target="_blank" rel="noopener">
                  <div class="video-top">
                    <span class="badge video">B站</span>
                    <span class="video-level">${escapeHtml(item.level)}</span>
                  </div>
                  <h3>${escapeHtml(item.title)}</h3>
                  <div class="video-meta">${escapeHtml(item.source)} · ${escapeHtml(
                  item.length
                )}</div>
                  <p>${escapeHtml(item.why)}</p>
                  <div class="video-watch"><strong>看哪几段：</strong>${escapeHtml(
                    item.watch
                  )}</div>
                </a>`
              )
              .join("")}
          </div>
        </section>`
      )
      .join("");
    return `
      <div class="page-head">
        <div class="eyebrow">视频</div>
        <h1>视频</h1>
        <p>精选 B 站视频，按主题分组。每个视频写了“为什么看”和“看哪几段”，避免收藏了 100 个视频却一个都没看完。</p>
      </div>
      <div class="callout">
        <p><strong>观看原则：</strong>1.5× 速看第一遍 → 回到 Lab 写代码 → 需要时再 1× 看关键段。一个主题最多看两个来源；看 20 分钟就停下来写 10 行代码。</p>
      </div>
      <div class="video-path">
        <span>建议顺序</span>
        <strong>CUDA 入门 → Attention → vLLM → MoE → 投机解码 → Profiling → TileLang → NCCL</strong>
      </div>
      ${groups}
    `;
  }

  function renderSections() {
    const cards = C.sections
      .map(
        (section) => `<article class="section-card">
          <h2>${escapeHtml(section.title)}</h2>
          <p class="goals">${escapeHtml(section.goals)}</p>
          <ol>${section.questions.map((q) => `<li>${escapeHtml(q)}</li>`).join("")}</ol>
        </article>`
      )
      .join("");
    return `
      <div class="page-head">
        <div class="eyebrow">讨论课</div>
        <h1>讨论课</h1>
        <p>没有强制讨论课；这里给你每周 60 分钟的讨论提纲。建议找一个学习伙伴，互相问、互相讲。</p>
      </div>
      <div class="callout">
        <p><strong>60 分钟流程：</strong>10 分钟回顾上周 · 20 分钟互相讲 checkpoint · 20 分钟 pair debug · 10 分钟下周计划。</p>
      </div>
      ${cards}
    `;
  }

  function renderCareer() {
    const c = C.career;
    const timeline = c.timeline
      .map(
        (row) => `<tr><td>${escapeHtml(row.period)}</td><td>${escapeHtml(
          row.action
        )}</td><td>${escapeHtml(row.output)}</td></tr>`
      )
      .join("");
    const roles = c.roles
      .map(
        (role) => `<div class="card">
          <div class="card-kicker">${escapeHtml(role.project)}</div>
          <h3>${escapeHtml(role.title)}</h3>
          <p>${escapeHtml(role.focus)}</p>
          <p class="small"><strong>Keywords:</strong> ${escapeHtml(role.keywords)}</p>
        </div>`
      )
      .join("");
    const projectMap = c.projectMap
      .map(
        (row) => `<tr><td>${escapeHtml(row.project)}</td><td>${escapeHtml(
          row.roles
        )}</td><td>${escapeHtml(row.evidence)}</td></tr>`
      )
      .join("");
    const bullets = c.bullets
      .map((bullet) => `<li>${escapeHtml(bullet)}</li>`)
      .join("");
    const questions = c.questions
      .map(
        (group) => `<details class="question-group">
          <summary>${escapeHtml(group.topic)} <span class="small muted">(${
          group.items.length
        })</span></summary>
          <ol>${group.items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ol>
        </details>`
      )
      .join("");
    const companies = c.companies
      .map(
        (group) => `<div class="card">
          <div class="card-kicker">${escapeHtml(group.group)}</div>
          <p>${group.names.map((name) => `<span class="chip">${escapeHtml(name)}</span>`).join(" ")}</p>
        </div>`
      )
      .join("");
    const sprint = c.sprint
      .map(
        (row) => `<tr><td>第 ${escapeHtml(row.week)} 周</td><td>${escapeHtml(
          row.focus
        )}</td><td>${escapeHtml(row.deliverable)}</td></tr>`
      )
      .join("");
    const mock = c.mock.map((item) => `<li>${escapeHtml(item)}</li>`).join("");

    return `
      <div class="page-head">
        <div class="eyebrow">求职</div>
        <h1>Infra 秋招</h1>
        <p>${escapeHtml(c.intro)}</p>
      </div>

      <div class="section-title"><h2>时间线</h2><a href="#/calendar">周历 →</a></div>
      <div class="prose"><table>
        <thead><tr><th>时间</th><th>动作</th><th>产出</th></tr></thead>
        <tbody>${timeline}</tbody>
      </table></div>

      <div class="section-title"><h2>岗位地图</h2><a href="#/textbook">教材 →</a></div>
      <div class="grid-3">${roles}</div>

      <div class="section-title"><h2>项目 → 岗位</h2><a href="#/calendar">项目 →</a></div>
      <div class="prose"><table>
        <thead><tr><th>项目</th><th>对口岗位</th><th>面试证据</th></tr></thead>
        <tbody>${projectMap}</tbody>
      </table></div>

      <div class="section-title"><h2>简历 bullet</h2><a href="#/syllabus">评分 →</a></div>
      <div class="card"><ul>${bullets}</ul>
        <p class="small muted">把 X/Y/Z 换成你测出来的数字。没有数字的 bullet 不要写。</p>
      </div>

      <div class="section-title"><h2>面试题库</h2><a href="#/resources">资源 →</a></div>
      <div class="card">${questions}</div>

      <div class="section-title"><h2>目标公司</h2><span class="small muted">参考，不构成投递建议</span></div>
      <div class="grid-3">${companies}</div>

      <div class="section-title"><h2>8 周冲刺</h2><a href="#/calendar">15 周计划 →</a></div>
      <div class="prose"><table>
        <thead><tr><th>周</th><th>重点</th><th>产出</th></tr></thead>
        <tbody>${sprint}</tbody>
      </table></div>

      <div class="section-title"><h2>模拟面试</h2><a href="#/sections">讨论课 →</a></div>
      <div class="card"><ul>${mock}</ul></div>
    `;
  }

  function renderStaff() {
    const cards = C.staff
      .map(
        (person) => `<article class="staff-card">
          <div class="avatar">${escapeHtml(person.avatar)}</div>
          <h3>${escapeHtml(person.name)}</h3>
          <div class="role">${escapeHtml(person.role)}</div>
          <p>${escapeHtml(person.bio)}</p>
        </article>`
      )
      .join("");
    return `
      <div class="page-head">
        <div class="eyebrow">课程角色</div>
        <h1>关于</h1>
        <p>这门课没有正式的教学团队。你是讲师，也是学生；AI 是助教；最好再找一个学习伙伴。</p>
      </div>
      <div class="staff-grid">${cards}</div>
      <div class="callout">
        <p><strong>Checkoff 规则：</strong>如果你不能把本周的 3 个问题讲给一个朋友听，就不算完成。讲不出来，回去重做，不要赶下一个实验。</p>
      </div>
    `;
  }

  function renderContact() {
    return `
      <div class="page-head">
        <div class="eyebrow">帮助</div>
        <h1>联系</h1>
        <p>本地课程没有 Piazza / Ed。用“最小复现 + 期望/实际”提问，用 checkoff 自测，用 issue/PR 留下痕迹。</p>
      </div>
      <div class="prose">${C.contact}</div>
    `;
  }

  const PAGES = {
    home: renderHome,
    calendar: renderCalendar,
    practice: renderPractice,
    syllabus: renderSyllabus,
    textbook: renderTextbook,
    resources: renderResources,
    videos: renderVideos,
    sections: renderSections,
    career: renderCareer,
    staff: renderStaff,
    contact: renderContact,
  };

  // ------------------------------------------------------------------
  // Post-render wiring
  // ------------------------------------------------------------------
  function wireCalendar() {
    const search = $("#calendar-search");
    if (search) {
      search.addEventListener("input", () => {
        const query = search.value.trim().toLowerCase();
        $$(".week-card").forEach((card) => {
          const haystack = card.getAttribute("data-search") || "";
          card.classList.toggle("hidden", !!query && !haystack.includes(query));
        });
        $("#search-empty").classList.toggle("hidden", $$(".week-card:not(.hidden)").length > 0);
      });
    }
    const jump = $("#jump-current");
    if (jump) {
      jump.addEventListener("click", () => {
        search.value = "";
        search.dispatchEvent(new Event("input"));
        const current = $(".week-card.current");
        if (current) current.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    }
  }

  function wireProgress() {
    $$(".item-check").forEach((checkbox) => {
      const key = checkbox.getAttribute("data-key");
      if (!key) return;
      checkbox.checked = !!state.progress[key];
      checkbox.addEventListener("change", () => {
        const saved = storage.set(key, checkbox.checked);
        state.progress = loadProgress();
        $("#storage-status").textContent = saved ? "" : "浏览器禁止保存进度，仅当前页面暂存。";
        updateProgress();
      });
    });
    updateProgress();
  }

  function wireTextbook() {
    $$(".toc button[data-scroll]").forEach((button) => {
      button.addEventListener("click", () => {
        const target = document.getElementById(button.getAttribute("data-scroll"));
        if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    });
  }

  function wirePage() {
    wireCalendar();
    wireProgress();
    wireTextbook();
  }

  // ------------------------------------------------------------------
  // Router
  // ------------------------------------------------------------------
  function render() {
    const route = getRoute();
    const page = PAGES[route] || PAGES.home;
    const label = PAGE_LABELS[route] || "首页";

    const crumbs = $("#crumbs");
    if (crumbs) crumbs.textContent = label;
    $$("#nav a").forEach((link) => {
      link.classList.toggle("active", link.getAttribute("data-nav") === route);
    });

    const content = $("#content");
    if (content) content.innerHTML = resolveCourseLinks(page());
    document.title = `${label} · DeepInfer 101`;
    wirePage();
    window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
  }

  // ------------------------------------------------------------------
  // Init
  // ------------------------------------------------------------------
  function init() {
    applyTheme(state.theme);
    renderNav();
    render();

    const themeBtn = $("#theme-toggle");
    if (themeBtn) {
      themeBtn.addEventListener("click", () => {
        state.theme = state.theme === "dark" ? "light" : "dark";
        saveTheme(state.theme);
        applyTheme(state.theme);
      });
    }

    const reset = $("#reset-progress");
    if (reset) {
      reset.addEventListener("click", () => {
        if (!confirm("重置全部进度？")) return;
        const saved = storage.reset();
        state.progress = loadProgress();
        $("#storage-status").textContent = saved ? "进度已重置" : "仅重置当前页面暂存进度";
        render();
      });
    }

    const menu = $("#menu-toggle");
    if (menu) {
      menu.addEventListener("click", () => {
        menu.setAttribute("aria-expanded", String($(".sidebar").classList.toggle("open")));
      });
    }

    $$("#nav a").forEach((link) => {
      link.addEventListener("click", () => {
        $(".sidebar").classList.remove("open");
        menu.setAttribute("aria-expanded", "false");
      });
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && $(".sidebar").classList.contains("open")) {
        $(".sidebar").classList.remove("open");
        menu.setAttribute("aria-expanded", "false");
        menu.focus();
      }
    });
    for (const event of ["storage", "pageshow", "focus"]) window.addEventListener(event, () => {
      state.progress = loadProgress();
      state.theme = loadTheme();
      applyTheme(state.theme);
      $$(".item-check").forEach((checkbox) => { checkbox.checked = !!state.progress[checkbox.dataset.key]; });
      updateProgress();
    });

    window.addEventListener("hashchange", render);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
