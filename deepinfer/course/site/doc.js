(function () {
  "use strict";
  const storage = window.CourseStorage;
  const themeButton = document.getElementById("doc-theme");
  const done = document.getElementById("doc-done");
  const hint = document.getElementById("doc-done-hint");
  function sync() {
    const theme = storage.theme();
    document.documentElement.setAttribute("data-theme", theme);
    if (themeButton) themeButton.textContent = theme === "dark" ? "☀️ 亮色" : "🌙 暗色";
    if (done) {
      const isDone = !!storage.progress()[done.dataset.key];
      done.textContent = isDone ? "✓ 已完成" : "标记完成";
      done.classList.toggle("done", isDone);
      done.setAttribute("aria-pressed", String(isDone));
    }
  }
  if (themeButton) themeButton.addEventListener("click", () => {
    storage.theme(storage.theme() === "dark" ? "light" : "dark");
    sync();
  });
  if (done) done.addEventListener("click", () => {
    const saved = storage.set(done.dataset.key, !storage.progress()[done.dataset.key]);
    sync();
    if (hint) hint.textContent = saved ? "已保存；完成标记不代表测试通过" : "浏览器禁止保存，仅本页暂存";
  });
  for (const event of ["storage", "pageshow", "focus"]) window.addEventListener(event, sync);
  sync();
})();
