const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const site = process.argv[2] ? path.resolve(process.argv[2]) : path.resolve(__dirname, "../deepinfer/course/site");
const scripts = Object.fromEntries(["data", "storage", "app"].map((name) => [name, fs.readFileSync(path.join(site, `${name}.js`), "utf8")]));
let links = 0;
const checked = new Set();
function checkLinks(html, file) {
  for (const match of html.matchAll(/(?:href|src)="([^"]+)"/g)) {
    const url = match[1].replaceAll("&amp;", "&");
    if (/^(https?:|mailto:|#\/)/.test(url)) continue;
    const [name, fragment] = url.split("#");
    const target = name ? path.resolve(path.dirname(file), decodeURIComponent(name)) : file;
    assert.ok(fs.existsSync(target), `${file}: missing ${url}`);
    if (fragment && !fragment.startsWith("/") && target.endsWith(".html")) {
      const content = fs.readFileSync(target, "utf8");
      assert.ok(content.includes(`id="${fragment}"`), `${file}: missing fragment ${url}`);
    }
    links++;
  }
}
function render(route) {
  const elements = new Map();
  function element(selector) {
    if (!elements.has(selector))
      elements.set(selector, {
        innerHTML: "",
        textContent: "",
        style: {},
        classList: {
          toggle() {},
          remove() {},
          contains() {
            return false;
          },
        },
        addEventListener() {},
        setAttribute() {},
        focus() {},
      });
    return elements.get(selector);
  }
  const context = {
    window: { addEventListener() {}, scrollTo() {} },
    document: { readyState: "complete", documentElement: element("html"), querySelector: element, querySelectorAll: () => [], addEventListener() {} },
    location: { hash: `#/${route}` },
    localStorage: { getItem: () => "null", setItem() {} },
  };
  for (const name of ["data", "storage", "app"]) vm.runInNewContext(scripts[name], context, { filename: `${name}.js` });
  const html = elements.get("#content").innerHTML;
  assert.ok(html.includes("<h1>"), `route ${route} did not render`);
  assert.ok(!html.includes("undefined"), `route ${route} has missing data`);
  checkLinks(html, path.join(site, "index.html"));
  return context.window.COURSE;
}
const course = render("home");
for (const week of course.weeks)
  for (const item of week.items) {
    if (item.soon) assert.ok(!item.title.includes("已发布"), `contradictory release state: ${item.title}`);
  }
for (const item of course.nav) {
  render(item.id);
  checked.add(item.id);
}
assert.equal(checked.size, course.nav.length);
const progressKeys = new Set(course.weeks.flatMap((week) => week.items.filter((item) => !item.soon && item.key).map((item) => item.key)));
const notes = fs.readdirSync(path.join(site, "notes")).filter((name) => name.endsWith(".html"));
for (const name of notes) {
  const file = path.join(site, "notes", name);
  const html = fs.readFileSync(file, "utf8");
  checkLinks(html, file);
  const key = html.match(/id="doc-done"[^>]+data-key="([^"]+)"/);
  if (key) assert.ok(progressKeys.has(key[1]), `orphan progress key ${key[1]}`);
}
checkLinks(fs.readFileSync(path.join(site, "index.html"), "utf8"), path.join(site, "index.html"));
console.log(
  `PASS: ${checked.size} routes render, ${notes.length} notes, ${links} local references, progress keys valid (structural checks; no browser layout)`
);
