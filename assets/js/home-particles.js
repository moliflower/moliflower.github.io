/**
 * Particle halo adapted from React Bits Particles by David Haz.
 * Source revision: e12c22192ac7b4b452c5e2feb9b586d74765d696
 * https://github.com/DavidHDev/react-bits/blob/e12c22192ac7b4b452c5e2feb9b586d74765d696/src/content/Backgrounds/Particles/Particles.jsx
 * MIT + Commons Clause: licenses/react-bits-LICENSE.txt.
 *
 * The upstream glow fragment shader and seeded random generator are retained.
 * Torus geometry, vertex projection, palettes, and the homepage controller are
 * customized here; this is not an unchanged React Bits component.
 */
(function () {
  "use strict";

  const PARTICLE_VERTEX = `#version 300 es
precision highp float;
layout(location = 0) in vec2 aCorner;
layout(location = 1) in vec3 aPosition;
layout(location = 2) in vec4 aRandom;
layout(location = 3) in vec3 aColor;
uniform mat3 uRotation;
uniform float uTime;
uniform float uCamera;
uniform float uAspect;
uniform vec2 uView;
uniform float uRatio;
out vec2 vLocal;
out float vRadius;
out float vBlur;
out float vAlpha;
out float vHalo;
out float vSeed;
out vec3 vColor;

void main() {
  float dust = aRandom.w < 0.0 ? 1.0 : 0.0;
  vec3 position = aPosition;
  float angle = atan(position.y, position.x);
  float ripple = sin(angle * 3.0 + uTime * 0.7) * 0.018;
  position.xy *= 1.0 + ripple * (1.0 - dust);
  position.z += sin(angle * 4.0 - uTime * 0.6) * 0.035 * (1.0 - dust);
  vec3 world = mix(uRotation * position, position, dust);
  float ahead = max(1.0, uCamera - world.z);
  vec2 ndc = vec2(world.x / uAspect, world.y) * 2.5 / ahead;
  vec2 screen = (ndc * 0.5 + 0.5) * uView;
  float depth = clamp((world.z + 2.3) / 4.6, 0.0, 1.0);
  float radius = (0.58 + aRandom.x * 0.82) * 6.8 / ahead;
  radius = mix(radius, 0.45 + aRandom.x * 0.55, dust);
  float halo = radius * 2.5 + 1.0;
  float extent = radius + halo + 1.5;
  float twinkle = 0.94 + 0.06 * sin(uTime * 0.55 + aRandom.y * 6.28);
  vLocal = aCorner * extent;
  vRadius = radius;
  vBlur = 0.0;
  vHalo = halo;
  vSeed = aRandom.y;
  vAlpha = mix(0.32 + depth * 0.58, 0.12 + aRandom.z * 0.16, dust) * twinkle;
  vColor = aColor * mix(0.55 + depth * 0.45, 0.7, dust);
  gl_Position = vec4((screen + aCorner * extent) / uView * 2.0 - 1.0, 0.0, 1.0);
}`;

  const PARTICLE_FRAGMENT = `#version 300 es
precision highp float;
precision highp int;
uniform int uShape;
uniform float uGlow;
uniform float uRatio;
uniform float uTime;
uniform float uSoft;
uniform float uShimmer;
uniform float uOpacity;
in vec2 vLocal;
in float vRadius;
in float vBlur;
in float vAlpha;
in float vHalo;
in float vSeed;
in vec3 vColor;
out vec4 outColor;

float ray(vec2 q, float aa) {
  float width = 0.14 * max(0.0, 1.0 - q.x);
  return (1.0 - smoothstep(width - aa, width + aa, q.y)) * step(q.x, 1.0);
}

void main() {
  float soft = vBlur + 0.7 / uRatio + vRadius * uSoft;
  float d = length(vLocal);
  float shape;
  if (uShape == 1) {
    float box = max(abs(vLocal.x), abs(vLocal.y));
    shape = 1.0 - smoothstep(vRadius * 0.86 - soft, vRadius * 0.86 + soft, box);
  } else if (uShape == 2) {
    float edge = max(vRadius * 0.22, 0.6 / uRatio);
    shape = 1.0 - smoothstep(edge - soft, edge + soft, abs(d - vRadius * 0.8));
  } else if (uShape == 3) {
    float reach = vRadius * 1.6;
    vec2 q = abs(vLocal) / max(reach, 0.0001);
    float aa = soft / max(reach, 0.0001);
    shape = max(max(ray(q, aa), ray(q.yx, aa)), 1.0 - smoothstep(vRadius * 0.32 - soft, vRadius * 0.32 + soft, d));
  } else {
    shape = 1.0 - smoothstep(vRadius - soft, vRadius + soft, d);
  }
  float halo = vHalo > 0.0 ? exp(-d / (vHalo * 0.35)) * uGlow * 0.4 : 0.0;
  float alpha = clamp(shape + halo * (1.0 - shape), 0.0, 1.0) * vAlpha * uOpacity * (1.0 - uSoft);
  if (alpha < 0.002) discard;
  vec2 uv = vLocal / max(vRadius * 2.0, 0.0001) + 0.5;
  vec3 color = clamp(vColor + uShimmer * 0.2 * sin(uv.yxx + uTime + vSeed * 6.28), 0.0, 1.0);
  outColor = vec4(color * alpha, alpha);
}`;

  const seeded = (seed) => {
    let state = seed >>> 0;
    return () => {
      state = (state + 0x6d2b79f5) >>> 0;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };

  const rgb = (hex) => [1, 3, 5].map((offset) => parseInt(hex.slice(offset, offset + 2), 16) / 255);
  const LIGHT = ["#0f766e", "#287c80", "#569c91", "#88b7ab"].map(rgb);
  const DARK = ["#5eead4", "#73bdbe", "#a4d3c9", "#d7f5e9"].map(rgb);
  const MAX_DPR = 1.5;
  const FRAME_INTERVAL = 1000 / 30;

  // The geometry is authored for this homepage, not copied from the source cloud.
  function makeHalo(compact) {
    const next = seeded(14837);
    const ringCount = compact ? 2200 : 4200;
    const dustCount = compact ? 70 : 130;
    const count = ringCount + dustCount;
    const positions = new Float32Array(count * 3);
    const randoms = new Float32Array(count * 4);
    const colors = { light: new Float32Array(count * 3), dark: new Float32Array(count * 3) };
    for (let i = 0; i < count; i++) {
      const dust = i >= ringCount;
      const u = (i / ringCount) * Math.PI * 2 + next() * 0.012;
      // Thin helical filaments give the dense halo a coherent flowing structure.
      const v = ((i % 17) / 17) * Math.PI * 2 + u * 3 + (next() - 0.5) * 0.28;
      const tube = 0.49 * (i % 5 === 0 ? 0.45 + next() * 0.55 : 0.86 + next() * 0.18);
      const radius = 1.8 + Math.cos(v) * tube;
      positions.set(
        dust ? [(next() - 0.5) * 7, (next() - 0.5) * 5.4, (next() - 0.5) * 4] : [radius * Math.cos(u), radius * Math.sin(u), Math.sin(v) * tube],
        i * 3
      );
      randoms.set([next(), next(), next(), dust ? -1 : 1], i * 4);
      // The same particles take on ink or luminous mint as the page theme changes.
      const selection = next() > 0.96 ? 3 : Math.floor(((u / (Math.PI * 2)) % 1) * 3);
      colors.light.set(LIGHT[selection], i * 3);
      colors.dark.set(DARK[selection], i * 3);
    }
    return { count, positions, randoms, colors };
  }

  function initialize() {
    const canvas = document.querySelector("canvas[data-home-particles]");
    const container = canvas?.closest(".home-particles");
    const hero = canvas?.closest(".home-hero");
    if (!canvas || !container || !hero) return;
    const toggle = hero.querySelector("button[data-particles-toggle]");
    const toggleLabel = toggle?.querySelector("[data-particles-label]") || toggle;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const cleanups = [];
    const buffers = [];
    let gl;
    let program;
    let vao;
    let raf = 0;
    let lastFrame = null;
    let visible = true;
    let failed = false;
    let dirty = true;
    // Motion without an available pause control must remain static.
    let paused = reduceMotion.matches || !toggle;
    let clock = 0;
    const pointer = { x: 0, y: 0, targetX: 0, targetY: 0 };
    const view = { width: 1, height: 1, ratio: 1 };

    function listen(target, type, callback, options) {
      target.addEventListener(type, callback, options);
      cleanups.push(() => target.removeEventListener(type, callback, options));
    }

    function cancelFrame() {
      if (raf) window.cancelAnimationFrame(raf);
      raf = 0;
      lastFrame = null;
    }

    function fallback() {
      if (failed) return;
      failed = true;
      cancelFrame();
      cleanups.forEach((cleanup) => cleanup());
      container.classList.remove("is-ready");
      if (toggle) toggle.hidden = true;
      if (gl) {
        buffers.forEach((buffer) => gl.deleteBuffer(buffer));
        if (vao) gl.deleteVertexArray(vao);
        if (program) gl.deleteProgram(program);
      }
    }

    function updateControl() {
      if (!toggle) return;
      toggle.hidden = reduceMotion.matches;
      toggle.setAttribute("aria-label", paused ? "播放粒子动画" : "暂停粒子动画");
      if (toggleLabel) toggleLabel.textContent = paused ? "播放粒子" : "暂停粒子";
      container.dataset.particlesPaused = String(paused);
    }

    function canAnimate() {
      return !failed && visible && !document.hidden && !reduceMotion.matches && !paused;
    }

    function compile(type, source) {
      const shader = gl.createShader(type);
      if (!shader) throw new Error("Particles shader unavailable");
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        gl.deleteShader(shader);
        throw new Error("Particles shader compilation failed");
      }
      return shader;
    }

    try {
      gl = canvas.getContext("webgl2", {
        alpha: true,
        premultipliedAlpha: true,
        antialias: false,
        depth: false,
        stencil: false,
        powerPreference: "low-power",
      });
      if (!gl) return;
      const shaders = [];
      try {
        shaders.push(compile(gl.VERTEX_SHADER, PARTICLE_VERTEX));
        shaders.push(compile(gl.FRAGMENT_SHADER, PARTICLE_FRAGMENT));
        program = gl.createProgram();
        if (!program) throw new Error("Particles program unavailable");
        shaders.forEach((shader) => gl.attachShader(program, shader));
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error("Particles program linking failed");
      } finally {
        shaders.forEach((shader) => gl.deleteShader(shader));
      }

      const uniforms = {};
      ["uRotation", "uTime", "uCamera", "uAspect", "uView", "uRatio", "uShape", "uGlow", "uSoft", "uShimmer", "uOpacity"].forEach((name) => {
        uniforms[name] = gl.getUniformLocation(program, name);
      });

      vao = gl.createVertexArray();
      if (!vao) throw new Error("Particles geometry unavailable");
      gl.bindVertexArray(vao);
      function attribute(location, width, data, divisor) {
        const buffer = gl.createBuffer();
        if (!buffer) throw new Error("Particles buffer unavailable");
        buffers.push(buffer);
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(location);
        gl.vertexAttribPointer(location, width, gl.FLOAT, false, 0, 0);
        gl.vertexAttribDivisor(location, divisor);
        return buffer;
      }
      const halo = makeHalo(container.clientWidth < 560);
      attribute(0, 2, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), 0);
      attribute(1, 3, halo.positions, 1);
      attribute(2, 4, halo.randoms, 1);
      const theme = () => (document.documentElement.dataset.theme === "dark" ? "dark" : "light");
      let currentTheme = theme();
      const colorBuffer = attribute(3, 3, halo.colors[currentTheme], 1);
      gl.enable(gl.BLEND);
      function setBlend() {
        gl.blendFuncSeparate(gl.ONE, currentTheme === "dark" ? gl.ONE : gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      }
      setBlend();
      gl.clearColor(0, 0, 0, 0);

      function rotation() {
        const x = 0.96 + Math.sin(clock * 0.2) * 0.035 + pointer.y * 0.12;
        const y = 0.15 + Math.sin(clock * 0.15) * 0.12 + pointer.x * 0.16;
        const z = -0.32 + clock * 0.026;
        const [sx, cx, sy, cy, sz, cz] = [Math.sin(x), Math.cos(x), Math.sin(y), Math.cos(y), Math.sin(z), Math.cos(z)];
        // Column-major Euler rotation, following the source component's layout.
        return new Float32Array([
          cy * cz + sy * sx * sz,
          cx * sz,
          -sy * cz + cy * sx * sz,
          -cy * sz + sy * sx * cz,
          cx * cz,
          sy * sz + cy * sx * cz,
          sy * cx,
          -sx,
          cy * cx,
        ]);
      }

      function render() {
        if (failed || gl.isContextLost()) return;
        try {
          if (currentTheme !== theme()) {
            currentTheme = theme();
            gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
            gl.bufferSubData(gl.ARRAY_BUFFER, 0, halo.colors[currentTheme]);
            setBlend();
          }
          const aspect = view.width / view.height;
          gl.viewport(0, 0, canvas.width, canvas.height);
          gl.clear(gl.COLOR_BUFFER_BIT);
          gl.useProgram(program);
          gl.bindVertexArray(vao);
          gl.uniformMatrix3fv(uniforms.uRotation, false, rotation());
          gl.uniform1f(uniforms.uTime, clock);
          gl.uniform1f(uniforms.uCamera, 6.8 / Math.min(1, aspect));
          gl.uniform1f(uniforms.uAspect, aspect);
          gl.uniform2f(uniforms.uView, view.width, view.height);
          gl.uniform1f(uniforms.uRatio, view.ratio);
          gl.uniform1i(uniforms.uShape, 0);
          gl.uniform1f(uniforms.uGlow, 0.9);
          gl.uniform1f(uniforms.uSoft, 0.06);
          gl.uniform1f(uniforms.uShimmer, 0.08);
          gl.uniform1f(uniforms.uOpacity, 0.9);
          gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, halo.count);
          dirty = false;
        } catch {
          fallback();
        }
      }

      function frame(now) {
        raf = 0;
        if (!canAnimate()) {
          lastFrame = null;
          return;
        }
        if (lastFrame === null) lastFrame = now;
        const elapsed = now - lastFrame;
        if (elapsed >= FRAME_INTERVAL) {
          const dt = Math.min(elapsed / 1000, 0.1);
          lastFrame = now;
          clock += dt;
          const follow = 1 - Math.exp(-dt * 4);
          pointer.x += (pointer.targetX - pointer.x) * follow;
          pointer.y += (pointer.targetY - pointer.y) * follow;
          render();
        }
        if (canAnimate()) raf = window.requestAnimationFrame(frame);
      }

      function wake() {
        if (failed || !visible || document.hidden) return;
        if (dirty) render();
        if (canAnimate() && !raf) raf = window.requestAnimationFrame(frame);
      }

      function resize() {
        view.width = Math.max(1, container.clientWidth);
        view.height = Math.max(1, container.clientHeight);
        view.ratio = Math.min(window.devicePixelRatio || 1, MAX_DPR);
        const width = Math.max(1, Math.round(view.width * view.ratio));
        const height = Math.max(1, Math.round(view.height * view.ratio));
        if (canvas.width !== width || canvas.height !== height) {
          canvas.width = width;
          canvas.height = height;
          dirty = true;
        }
        wake();
      }

      listen(canvas, "webglcontextlost", fallback);
      listen(document, "visibilitychange", () => {
        if (document.hidden) cancelFrame();
        else wake();
      });
      listen(window, "resize", resize);
      listen(window, "pagehide", cancelFrame);
      listen(window, "pageshow", wake);
      const themeObserver = new MutationObserver(() => {
        dirty = true;
        wake();
      });
      themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
      cleanups.push(() => themeObserver.disconnect());
      listen(reduceMotion, "change", () => {
        if (reduceMotion.matches) {
          paused = true;
          cancelFrame();
        }
        updateControl();
        dirty = true;
        wake();
      });
      listen(
        hero,
        "pointermove",
        (event) => {
          if (!canAnimate() || event.pointerType === "touch") return;
          const rect = canvas.getBoundingClientRect();
          pointer.targetX = Math.max(-1, Math.min(1, ((event.clientX - rect.left) / rect.width) * 2 - 1));
          pointer.targetY = Math.max(-1, Math.min(1, ((event.clientY - rect.top) / rect.height) * 2 - 1));
        },
        { passive: true }
      );
      listen(hero, "pointerleave", () => {
        pointer.targetX = 0;
        pointer.targetY = 0;
      });
      if (toggle) {
        listen(toggle, "click", () => {
          if (reduceMotion.matches || failed) return;
          paused = !paused;
          if (paused) cancelFrame();
          updateControl();
          wake();
        });
      }

      if ("ResizeObserver" in window) {
        const observer = new ResizeObserver(resize);
        observer.observe(container);
        cleanups.push(() => observer.disconnect());
      }
      if ("IntersectionObserver" in window) {
        const observer = new IntersectionObserver((entries) => {
          visible = entries.some((entry) => entry.isIntersecting);
          if (!visible) cancelFrame();
          else wake();
        });
        observer.observe(container);
        cleanups.push(() => observer.disconnect());
      }

      resize();
      if (!failed && !gl.isContextLost()) {
        container.classList.add("is-ready");
        updateControl();
        wake();
      }
    } catch {
      fallback();
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
