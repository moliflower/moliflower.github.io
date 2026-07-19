import { spawn } from "node:child_process";
import { mkdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";

const chrome = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const root = path.resolve(".omo/evidence/blog-illustrations");
const profile = "C:\\Users\\YUHAND~1\\AppData\\Local\\Temp\\opencode\\chrome-cdp-blog-illustrations";
const port = 9337;

await rm(profile, { recursive: true, force: true });
await mkdir(profile, { recursive: true });

const browser = spawn(
  chrome,
  [
    "--headless=new",
    "--disable-gpu",
    "--hide-scrollbars",
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${profile}`,
    "about:blank",
  ],
  { stdio: "ignore" },
);

async function waitForEndpoint() {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try {
      const response = await fetch(`http://127.0.0.1:${port}/json/version`);
      if (response.ok) return;
    } catch (error) {
      if (!(error instanceof TypeError)) throw error;
    }
    await new Promise((resolve) => setTimeout(resolve, 125));
  }
  throw new Error("Chrome DevTools endpoint did not become ready");
}

await waitForEndpoint();

const targetResponse = await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, {
  method: "PUT",
});
const target = await targetResponse.json();
const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener("open", resolve, { once: true });
  socket.addEventListener("error", reject, { once: true });
});

let nextId = 0;
const pending = new Map();
socket.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  if (!message.id) return;
  const waiter = pending.get(message.id);
  if (!waiter) return;
  pending.delete(message.id);
  if (message.error) waiter.reject(new Error(message.error.message));
  else waiter.resolve(message.result);
});

function send(method, params = {}) {
  nextId += 1;
  const id = nextId;
  socket.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}

const posts = [
  ["position", "http://localhost:8080/blog/2026/%E4%BB%8E-vit-patch-%E5%88%B0-llm-rope-%E7%90%86%E8%A7%A3%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81/"],
  ["ffn", "http://localhost:8080/blog/2026/%E5%AF%B9%E4%BA%8E%E6%A8%A1%E5%9E%8B%E4%B8%AD%E5%8D%87%E7%BB%B4%E5%92%8C%E9%99%8D%E7%BB%B4%E7%9A%84%E7%90%86%E8%A7%A3/"],
  ["ssh", "http://localhost:8080/blog/2025/%E8%80%81%E5%B8%88%E5%8F%AA%E7%BB%99%E6%88%91%E4%B8%80%E4%B8%AAip-%E7%A7%91%E7%A0%94%E5%B0%8F%E7%99%BD%E7%9A%84ssh%E5%85%A5%E9%97%A8/"],
];
const requestedName = process.argv[2];
const selectedPosts = requestedName ? posts.filter(([name]) => name === requestedName) : posts;
if (selectedPosts.length === 0) throw new Error(`Unknown post capture filter: ${requestedName}`);

const cases = [];
for (const [name, url] of selectedPosts) {
  for (const scheme of ["light", "dark"]) {
    for (const width of [375, 1280]) {
      cases.push({ name, url, scheme, width, height: width === 375 ? 812 : 900 });
    }
  }
}

await send("Page.enable");
for (const item of cases) {
  await send("Emulation.setDeviceMetricsOverride", {
    width: item.width,
    height: item.height,
    deviceScaleFactor: 1,
    mobile: item.width === 375,
  });
  await send("Emulation.setEmulatedMedia", {
    media: "screen",
    features: [{ name: "prefers-color-scheme", value: item.scheme }],
  });
  const separator = item.url.includes("?") ? "&" : "?";
  await send("Page.navigate", { url: `${item.url}${separator}qa=${Date.now()}` });
  for (let attempt = 0; attempt < 30; attempt += 1) {
    const state = await send("Runtime.evaluate", {
      expression: "document.readyState === 'complete' && document.body.scrollHeight > 1200",
      returnByValue: true,
    });
    if (state.result.value === true) break;
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  await send("Runtime.evaluate", {
    expression: `
      document.querySelectorAll('img[loading=lazy]').forEach((img) => img.loading = 'eager');
      const style = document.createElement('style');
      style.textContent = 'footer.fixed-bottom { display: none !important; }';
      document.head.append(style);
    `,
  });
  await new Promise((resolve) => setTimeout(resolve, 900));
  const panStates = item.width === 375 ? [["", 0], ["-pan-mid", 0.5], ["-pan-end", 1]] : [["", 0]];
  for (const [panSuffix, ratio] of panStates) {
    await send("Runtime.evaluate", {
      expression: `
        document.querySelector('footer.fixed-bottom')?.remove();
        document.querySelectorAll('.figure-scroll').forEach((figure) => {
          figure.scrollLeft = (figure.scrollWidth - figure.clientWidth) * ${ratio};
        });
      `,
    });
    await new Promise((resolve) => setTimeout(resolve, 250));
    const metrics = await send("Page.getLayoutMetrics");
    const width = item.width;
    const height = Math.ceil(metrics.cssContentSize.height);
    const screenshot = await send("Page.captureScreenshot", {
      format: "png",
      fromSurface: true,
      captureBeyondViewport: true,
      clip: { x: 0, y: 0, width, height, scale: 1 },
    });
    const suffix = item.width === 375 ? "mobile" : "desktop";
    const output = path.join(root, `${item.name}-${suffix}-${item.scheme}${panSuffix}.png`);
    await writeFile(output, Buffer.from(screenshot.data, "base64"));
    console.log(`${path.basename(output)} ${width}x${height}`);
  }
}

socket.close();
browser.kill();
