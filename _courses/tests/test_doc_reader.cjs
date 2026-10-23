const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const script = fs.readFileSync(path.join(__dirname, "../deepinfer/course/site/doc.js"), "utf8");

for (const mobile of [false, true]) {
  const events = new Map();
  const progress = {};
  let theme = "light";
  const elements = new Map();
  for (const id of ["doc-theme", "doc-done", "doc-done-hint", "doc-toc"]) {
    const attributes = new Map();
    const listeners = new Map();
    elements.set(id, {
      open: true,
      dataset: { key: "w6-l11" },
      classList: { toggle() {} },
      textContent: "",
      attributes,
      listeners,
      setAttribute(name, value) {
        attributes.set(name, value);
      },
      addEventListener(name, listener) {
        listeners.set(name, listener);
      },
    });
  }
  const documentAttributes = new Map();
  const context = {
    document: {
      getElementById: (id) => elements.get(id),
      documentElement: {
        setAttribute: (name, value) => documentAttributes.set(name, value),
      },
    },
    window: {
      matchMedia: () => ({ matches: mobile }),
      addEventListener: (name, listener) => events.set(name, listener),
      CourseStorage: {
        theme(value) {
          if (value) theme = value;
          return theme;
        },
        progress: () => progress,
        set(key, value) {
          progress[key] = value;
          return true;
        },
      },
    },
  };
  vm.runInNewContext(script, context, { filename: "doc.js" });
  assert.equal(elements.get("doc-toc").open, !mobile);
  elements.get("doc-done").listeners.get("click")();
  assert.equal(progress["w6-l11"], true);
  assert.equal(elements.get("doc-done").attributes.get("aria-pressed"), "true");
  assert.ok(elements.get("doc-done-hint").textContent.includes("不代表测试通过"));
  elements.get("doc-theme").listeners.get("click")();
  assert.equal(documentAttributes.get("data-theme"), "dark");
  elements.get("doc-toc").open = mobile;
  events.get("pageshow")();
  assert.equal(elements.get("doc-toc").open, mobile, "sync must not override a learner's table-of-contents choice");
}
console.log("PASS: desktop/mobile contents, progress, theme, and learner-controlled expansion");
