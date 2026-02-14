(function () {
  "use strict";

  function initScrollAnimations() {
    var targets = document.querySelectorAll(".scroll-animate");
    if (!targets.length) return;

    // Respect reduced motion preference
    var prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    );
    if (prefersReducedMotion.matches) {
      for (var i = 0; i < targets.length; i++) {
        targets[i].style.opacity = "1";
      }
      return;
    }

    var observer = new IntersectionObserver(
      function (entries) {
        for (var j = 0; j < entries.length; j++) {
          if (entries[j].isIntersecting) {
            entries[j].target.classList.add("animate-fade-in-up");
            observer.unobserve(entries[j].target);
          }
        }
      },
      { threshold: 0.1, rootMargin: "0px 0px -50px 0px" },
    );

    for (var k = 0; k < targets.length; k++) {
      observer.observe(targets[k]);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initScrollAnimations);
  } else {
    initScrollAnimations();
  }
})();
