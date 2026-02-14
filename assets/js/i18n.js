(function () {
  "use strict";

  var translations = {
    en: {
      about: "about",
      blog: "blog",
      publications: "publications",
      projects: "projects",
      repositories: "repositories",
      cv: "cv",
      news: "news",
      latest_posts: "latest posts",
      selected_publications: "selected publications",
      no_news: "No news so far...",
      no_posts: "No posts so far...",
      copyright: "Copyright",
      last_updated: "Last updated",
      search: "Search",
      toggle_navigation: "Toggle navigation",
    },
    zh: {
      about: "\u5173\u4e8e",
      blog: "\u535a\u5ba2",
      publications: "\u8bba\u6587\u53d1\u8868",
      projects: "\u9879\u76ee",
      repositories: "\u4ee3\u7801\u4ed3\u5e93",
      cv: "\u7b80\u5386",
      news: "\u65b0\u95fb\u52a8\u6001",
      latest_posts: "\u6700\u65b0\u6587\u7ae0",
      selected_publications: "\u7cbe\u9009\u8bba\u6587",
      no_news: "\u6682\u65e0\u65b0\u95fb...",
      no_posts: "\u6682\u65e0\u6587\u7ae0...",
      copyright: "\u7248\u6743\u6240\u6709",
      last_updated: "\u6700\u540e\u66f4\u65b0",
      search: "\u641c\u7d22",
      toggle_navigation: "\u5207\u6362\u5bfc\u822a",
    },
  };

  function setLanguage(lang) {
    var elements = document.querySelectorAll("[data-i18n]");
    for (var i = 0; i < elements.length; i++) {
      var key = elements[i].getAttribute("data-i18n");
      if (translations[lang] && translations[lang][key]) {
        elements[i].textContent = translations[lang][key];
      }
    }
    document.documentElement.setAttribute("lang", lang);
    localStorage.setItem("site-lang", lang);

    var toggleBtn = document.getElementById("lang-toggle");
    if (toggleBtn) {
      toggleBtn.textContent = lang === "zh" ? "EN" : "\u4e2d";
    }
  }

  function toggleLanguage() {
    var current = localStorage.getItem("site-lang") || "en";
    var next = current === "en" ? "zh" : "en";
    setLanguage(next);
  }

  function initLanguage() {
    var saved = localStorage.getItem("site-lang");
    if (saved && saved !== "en") {
      setLanguage(saved);
    }
  }

  // Expose globally
  window.i18n = {
    setLanguage: setLanguage,
    toggleLanguage: toggleLanguage,
    initLanguage: initLanguage,
  };

  // Auto-init when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initLanguage);
  } else {
    initLanguage();
  }
})();
