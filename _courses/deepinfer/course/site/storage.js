(function () {
  "use strict";
  let memory = {};
  let memoryTheme = "dark";
  function clean(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) return {};
    return Object.fromEntries(Object.entries(value).filter(([key, done]) =>
      key !== "__proto__" && key !== "constructor" && typeof done === "boolean"));
  }
  function progress() {
    try { memory = clean(JSON.parse(localStorage.getItem("di101-progress") || "{}")); }
    catch (_) {}
    return { ...memory };
  }
  function save(value) {
    memory = clean(value);
    try {
      localStorage.setItem("di101-progress", JSON.stringify(memory));
      return true;
    } catch (_) { return false; }
  }
  function set(key, done) {
    const latest = progress();
    latest[key] = !!done;
    return save(latest);
  }
  function theme(next) {
    if (next) {
      memoryTheme = next === "light" ? "light" : "dark";
      try { localStorage.setItem("di101-theme", memoryTheme); } catch (_) {}
    } else {
      try { memoryTheme = localStorage.getItem("di101-theme") === "light" ? "light" : "dark"; }
      catch (_) {}
    }
    return memoryTheme;
  }
  window.CourseStorage = { progress, set, reset: () => save({}), theme };
})();
