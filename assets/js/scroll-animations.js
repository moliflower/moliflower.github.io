(function () {
  "use strict";

  function initScrollAnimations() {
    var targets = document.querySelectorAll(".scroll-animate");
    if (!targets.length || !window.matchMedia || !("IntersectionObserver" in window)) return;

    // Respect reduced motion preference
    var prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (prefersReducedMotion.matches) return;

    var observer = new IntersectionObserver(
      function (entries) {
        for (var j = 0; j < entries.length; j++) {
          if (entries[j].isIntersecting) {
            if (entries[j].target.classList.contains("scroll-reveal-pending")) {
              entries[j].target.classList.remove("scroll-reveal-pending");
              entries[j].target.classList.add("animate-fade-in-up");
            }
            observer.unobserve(entries[j].target);
          }
        }
      },
      { threshold: 0.01, rootMargin: "0px 0px -24px 0px" }
    );

    for (var k = 0; k < targets.length; k++) {
      // Leave the first screen visible; enhance only content below the fold.
      if (targets[k].getBoundingClientRect().top >= window.innerHeight) {
        observer.observe(targets[k]);
        targets[k].classList.add("scroll-reveal-pending");
        targets[k].addEventListener("focusin", function () {
          this.classList.remove("scroll-reveal-pending", "animate-fade-in-up");
          observer.unobserve(this);
        });
      }
    }

    function updateMotionPreference() {
      if (!prefersReducedMotion.matches) return;
      observer.disconnect();
      for (var i = 0; i < targets.length; i++) {
        targets[i].classList.remove("scroll-reveal-pending", "animate-fade-in-up");
      }
    }

    if (prefersReducedMotion.addEventListener) {
      prefersReducedMotion.addEventListener("change", updateMotionPreference);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initScrollAnimations);
  } else {
    initScrollAnimations();
  }
})();
