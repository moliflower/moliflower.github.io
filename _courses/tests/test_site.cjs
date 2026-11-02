const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const site = process.argv[2] ? path.resolve(process.argv[2]) : path.resolve(__dirname, "../deepinfer/course/site");
const rootEntry = path.resolve(site, "../../index.html");
const rootHtml = fs.readFileSync(rootEntry, "utf8");
assert.ok(rootHtml.includes('id="content"'), "root entry must contain the course application");
assert.ok(rootHtml.includes('data-course-base="course/site/"'), "root entry must declare its asset prefix");
assert.match(rootHtml, /course\/site\/app\.js\?v=[a-f0-9]{12}/, "root entry must invalidate outdated script caches");
assert.ok(!/http-equiv\s*=\s*["']refresh|<iframe|<base\s/i.test(rootHtml), "root entry must not redirect or wrap another page");
const scripts = Object.fromEntries(["data", "storage", "app"].map((name) => [name, fs.readFileSync(path.join(site, `${name}.js`), "utf8")]));
let links = 0;
const checked = new Set();
function checkLinks(html, file) {
  for (const match of html.matchAll(/(?:href|src)="([^"]+)"/g)) {
    const url = match[1].replaceAll("&amp;", "&");
    if (/^(https?:|mailto:|#\/)/.test(url)) continue;
    const [resource, fragment] = url.split("#");
    const name = resource.split("?")[0];
    const target = name ? path.resolve(path.dirname(file), decodeURIComponent(name)) : file;
    assert.ok(fs.existsSync(target), `${file}: missing ${url}`);
    if (fragment && !fragment.startsWith("/") && target.endsWith(".html")) {
      const content = fs.readFileSync(target, "utf8");
      assert.ok(content.includes(`id="${fragment}"`), `${file}: missing fragment ${url}`);
    }
    links++;
  }
}
function render(route, entry = path.join(site, "index.html")) {
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
        getAttribute(name) {
          return selector === "html" && name === "data-course-base" && entry === rootEntry ? "course/site/" : null;
        },
        focus() {},
      });
    return elements.get(selector);
  }
  const context = {
    window: { addEventListener() {}, scrollTo() {} },
    document: {
      readyState: "complete",
      documentElement: element("html"),
      querySelector: element,
      querySelectorAll: () => [],
      addEventListener() {},
    },
    location: { hash: `#/${route}` },
    localStorage: { getItem: () => "null", setItem() {} },
  };
  for (const name of ["data", "storage", "app"]) vm.runInNewContext(scripts[name], context, { filename: `${name}.js` });
  const html = elements.get("#content").innerHTML;
  assert.ok(html.includes("<h1>"), `route ${route} did not render`);
  assert.ok(!html.includes("undefined"), `route ${route} has missing data`);
  checkLinks(html, entry);
  const navigation = elements.get("#nav").innerHTML;
  assert.ok(
    [...navigation.matchAll(/href="([^"]+)"/g)].every((match) => match[1].startsWith("#/")),
    "navigation must stay on the current entry"
  );
  return { course: context.window.COURSE, html };
}
const { course } = render("home");
const weekSix = course.weeks.find((week) => week.n === 6);
assert.equal(weekSix.items.length, 5);
for (const item of weekSix.items) {
  assert.ok(!item.soon && item.key && item.link, `Week 6 item is not published: ${item.title}`);
}
const practiceHtml = render("practice").html;
const introItems = course.weeks.find((week) => week.n === 0).items;
for (const route of ["home", "calendar"]) {
  const introCard = render(route).html.match(/<section[^>]*data-week="0"[\s\S]*?<\/section>/)[0];
  assert.equal((introCard.match(/data-key-row=/g) || []).length, 2);
  for (const item of introItems) {
    const row = `data-key-row="${item.key}"`;
    assert.equal(introCard.includes(row), item.type === "reading");
    if (item.type === "lab") assert.ok(practiceHtml.includes(row));
  }
}
for (const note of ["lab05.html", "hw06.html", "project2-scheduler.html"]) {
  assert.ok(practiceHtml.includes(`href="notes/${note}"`), `practice center omits ${note}`);
}
for (const week of course.weeks)
  for (const item of week.items) {
    if (item.soon) assert.ok(!item.title.includes("已发布"), `contradictory release state: ${item.title}`);
  }
for (const item of course.nav) {
  const legacy = render(item.id);
  const root = render(item.id, rootEntry);
  const legacyLinks = [...legacy.html.matchAll(/(?:href|src)="([^"]+)"/g)].map((match) => match[1]);
  const rootLinks = [...root.html.matchAll(/(?:href|src)="([^"]+)"/g)].map((match) => match[1]);
  assert.equal(rootLinks.length, legacyLinks.length);
  rootLinks.forEach((link, index) => {
    const actual = new URL(link.replaceAll("&amp;", "&"), "https://example.com/deepinfer/");
    if (legacyLinks[index].startsWith("#")) {
      assert.equal(actual.pathname, "/deepinfer/", "hash navigation must retain the short address");
      assert.equal(actual.hash, legacyLinks[index]);
    } else {
      const expected = new URL(legacyLinks[index].replaceAll("&amp;", "&"), "https://example.com/deepinfer/course/site/index.html");
      assert.equal(actual.href, expected.href, `${item.id}: root entry changed link target`);
    }
  });
  checked.add(item.id);
}
assert.equal(checked.size, course.nav.length);
const progressKeys = new Set(course.weeks.flatMap((week) => week.items.filter((item) => !item.soon && item.key).map((item) => item.key)));
const notes = fs.readdirSync(path.join(site, "notes")).filter((name) => name.endsWith(".html"));
for (const name of notes) {
  const file = path.join(site, "notes", name);
  const html = fs.readFileSync(file, "utf8");
  assert.match(html, /\.\.\/doc\.js\?v=[a-f0-9]{12}/, `${name}: reader code must be cache-versioned`);
  assert.ok(html.includes('id="doc-toc"'), `${name}: accessible chapter contents missing`);
  checkLinks(html, file);
  const returnPath = process.argv.includes("--offline") ? "../../../index.html" : "../../../";
  assert.ok(html.includes(`href="${returnPath}#/calendar"`), `${name}: return link must use the root entry`);
  const key = html.match(/id="doc-done"[^>]+data-key="([^"]+)"/);
  if (key) assert.ok(progressKeys.has(key[1]), `orphan progress key ${key[1]}`);
}
checkLinks(fs.readFileSync(path.join(site, "index.html"), "utf8"), path.join(site, "index.html"));
checkLinks(rootHtml, rootEntry);
console.log(
  `PASS: ${checked.size} routes at both entries, no redirect, equivalent link targets, ${notes.length} notes, ${links} local references, progress keys valid (structural checks; no browser layout)`
);
