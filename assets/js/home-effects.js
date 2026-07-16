(function () {
  "use strict";

  var hero = document.querySelector(".home-hero");
  if (!hero || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    return;
  }

  if (window.matchMedia("(pointer: fine)").matches) {
    hero.addEventListener("pointermove", function (event) {
      var bounds = hero.getBoundingClientRect();
      var x = ((event.clientX - bounds.left) / bounds.width) * 100;
      var y = ((event.clientY - bounds.top) / bounds.height) * 100;

      hero.style.setProperty("--pointer-x", x.toFixed(1) + "%");
      hero.style.setProperty("--pointer-y", y.toFixed(1) + "%");
    });
  }
})();
