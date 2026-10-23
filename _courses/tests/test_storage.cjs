const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const site = process.argv[2] ? path.resolve(process.argv[2]) : path.resolve(__dirname, "../deepinfer/course/site");
const source = fs.readFileSync(path.join(site, "storage.js"), "utf8");
const values = new Map();
const localStorage = { getItem: (key) => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) };
function client(storage = localStorage) {
  const context = { window: {}, localStorage: storage };
  vm.runInNewContext(source, context);
  return context.window.CourseStorage;
}
const first = client();
const second = client();
for (const value of ["null", "[]", "42", '"bad"', "{broken", '{"q1":"false"}']) {
  values.set("di101-progress", value);
  assert.equal(Object.keys(first.progress()).length, 0);
}
first.set("q1", true);
second.set("q2", true);
assert.equal(first.progress().q2, true);
assert.equal(second.progress().q1, true);
first.reset();
assert.equal(Object.keys(second.progress()).length, 0);
values.set("di101-theme", "invalid");
assert.equal(first.theme(), "dark");
second.theme("light");
assert.equal(first.theme(), "light");
const denied = client({
  getItem() {
    throw Error("denied");
  },
  setItem() {
    throw Error("denied");
  },
});
assert.equal(denied.set("q1", true), false);
assert.equal(denied.progress().q1, true);
denied.reset();
assert.equal(Object.keys(denied.progress()).length, 0);
console.log("PASS: malformed data, two clients, reset, theme and denied storage");
