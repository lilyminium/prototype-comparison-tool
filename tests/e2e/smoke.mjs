// End-to-end smoke test of the site in real browsers (Playwright). Fails on any console error.
// Usage: pixi run node tests/e2e/smoke.mjs [chromium|firefox|webkit ...] [--url https://<deployed>/]
// Writes screenshots and a JSON report to logs/e2e/.
import { mkdirSync, writeFileSync } from "node:fs";
import { chromium, firefox, webkit } from "playwright";
import { serve } from "./serve.mjs";

const args = process.argv.slice(2);
const urlArg = args.includes("--url") ? args[args.indexOf("--url") + 1] : null;
const browsers = args.filter((a) => ["chromium", "firefox", "webkit"].includes(a));
const engines = { chromium, firefox, webkit };
mkdirSync("logs/e2e", { recursive: true });

const local = urlArg ? null : await serve();
const base = urlArg || local.url;
const report = { base, browsers: {} };
let failed = false;

for (const name of browsers.length ? browsers : ["chromium"]) {
  const browser = await engines[name].launch();
  const r = (report.browsers[name] = { steps: {}, errors: [] });
  for (const scheme of ["light", "dark"]) {
    const context = await browser.newContext({ viewport: { width: 1200, height: 900 }, colorScheme: scheme });
    const page = await context.newPage();
    let bytes = 0;
    page.on("console", (m) => m.type() === "error" && r.errors.push(`[${scheme}] console: ${m.text()}`));
    page.on("pageerror", (e) => r.errors.push(`[${scheme}] pageerror: ${e.message}`));
    page.on("response", async (res) => {
      try {
        const h = await res.allHeaders();
        bytes += Number(h["content-length"] || 0) || (await res.body()).length;
      } catch {}
    });
    const step = async (label, fn) => {
      const t0 = Date.now();
      try {
        await fn();
        r.steps[`${scheme}:${label}`] = { ms: Date.now() - t0, bytes };
      } catch (e) {
        r.errors.push(`[${scheme}] ${label}: ${e.message.split("\n")[0]}`);
      }
      await page.screenshot({ path: `logs/e2e/${name}-${scheme}-${label}.png`, fullPage: true });
    };
    await step("load", async () => {
      await page.goto(base);
      await page.waitForSelector("table tbody tr", { timeout: 30000 });
    });
    if (scheme === "dark") {
      await context.close();
      continue;
    }
    await step("bond-param", async () => {
      await page.goto(base + "#param/b1");
      await page.waitForSelector("svg.chart rect", { timeout: 30000 });
      await page.waitForSelector(".mol svg", { timeout: 60000 });
    });
    await step("proper-param", async () => {
      await page.goto(base + "#param/t17");
      await page.waitForSelector("svg.chart path.line", { timeout: 30000 });
      await page.check("text=TorsionDrives (");
      await page.uncheck("text=Optimizations (");
      await page.waitForSelector("text=3D scan + energy", { timeout: 60000 });
    });
    await step("td-3d-animation", async () => {
      await page.click("text=3D scan + energy");
      await page.waitForSelector("dialog.viewer3d[open] canvas", { timeout: 60000 });
      await page.waitForSelector("dialog.viewer3d .viewer-chart svg circle.current", { timeout: 30000 });
      const first = await page.textContent("dialog.viewer3d .viewer-readout");
      await page.waitForTimeout(1200);
      const later = await page.textContent("dialog.viewer3d .viewer-readout");
      if (first === later) throw new Error(`animation did not advance: ${first}`);
    });
    await step("close-3d", async () => {
      await page.click("dialog.viewer3d .viewer-close");
    });
    await step("opt-3d", async () => {
      await page.goto(base + "#param/b1");
      await page.waitForSelector("text=3D structure", { timeout: 60000 });
      await page.click("text=3D structure");
      await page.waitForSelector("dialog.viewer3d[open] canvas", { timeout: 60000 });
      await page.waitForFunction(() => document.querySelector("dialog.viewer3d .viewer-note")?.textContent.includes("QM-optimized"), null, { timeout: 30000 });
      await page.click("dialog.viewer3d .viewer-close");
    });
    await step("td-only-param", async () => {
      // parameter with observations only in TorsionDrive records must not throw
      await page.goto(base + "#param/b47");
      await page.waitForSelector("text=Training molecules with this parameter", { timeout: 30000 });
    });
    await step("improper-param", async () => {
      await page.goto(base + "#param/i1");
      await page.waitForSelector("svg.chart rect", { timeout: 30000 });
    });
    await step("smarts-bond", async () => {
      await page.goto(base + "#smarts/" + encodeURIComponent("[#6X4:1]-[#6X4:2]"));
      await page.waitForSelector("text=Sage bond parameters on these atoms", { timeout: 120000 });
    });
    await step("smarts-torsion", async () => {
      await page.goto(base + "#smarts/" + encodeURIComponent("[*:1]-[#6X4:2]-[#6X4:3]-[*:4]"));
      await page.waitForSelector("text=Sage proper torsion parameters on these atoms", { timeout: 180000 });
    });
    await step("smarts-broad-angle", async () => {
      await page.goto(base + "#smarts/" + encodeURIComponent("[*:1]~[*:2]~[*:3]"));
      await page.waitForSelector("text=Sage angle parameters on these atoms", { timeout: 180000 });
    });
    await step("smarts-all-atoms", async () => {
      await page.goto(base + "#smarts/" + encodeURIComponent("[*:1]"));
      await page.waitForSelector(".mol svg", { timeout: 120000 });
    });
    await step("smarts-duplicate-map", async () => {
      await page.goto(base + "#smarts/" + encodeURIComponent("[C:1][C:1][O:2]"));
      await page.waitForSelector(".error", { timeout: 60000 });
    });
    await step("smarts-bad", async () => {
      await page.goto(base + "#smarts/" + encodeURIComponent("[C:1]-[C:3]"));
      await page.waitForSelector(".error", { timeout: 60000 });
    });
    await step("molecule", async () => {
      await page.goto(base + "#molecule/" + encodeURIComponent("CC(=O)Nc1ccc(O)cc1"));
      await page.waitForSelector("text=Most similar training molecules", { timeout: 120000 });
      await page.waitForSelector(".mol svg", { timeout: 60000 });
    });
    await step("opt-conformers-3d", async () => {
      // a molecule with three optimization records: conformer checkboxes, no playback controls
      await page.goto(base + "#molecule/" + encodeURIComponent('BrCCBr'));
      await page.waitForSelector("text=3D structure", { timeout: 120000 });
      const text = await page.textContent("main");
      if (/\b(null|undefined|NaN)\b/.test(text)) throw new Error("page shows a literal null/undefined/NaN");
      await page.click("text=3D structure");
      await page.waitForSelector("dialog.viewer3d[open] .viewer-conformers input[type=checkbox] >> nth=2", { timeout: 60000 });
      if (await page.isVisible("dialog.viewer3d .viewer-play")) throw new Error("play button visible for an optimization record");
      await page.uncheck("dialog.viewer3d .viewer-conformers input[type=checkbox] >> nth=1");
      await page.click("dialog.viewer3d .viewer-close");
    });
    await step("about", async () => {
      await page.goto(base + "#about");
      await page.waitForSelector("text=Validation");
    });
    await context.close();
  }
  // Phone width
  const phone = await browser.newContext({ viewport: { width: 390, height: 844 } });
  const pp = await phone.newPage();
  await pp.goto(base + "#param/a1");
  await pp.waitForSelector("svg.chart rect", { timeout: 30000 });
  const overflow = await pp.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
  if (overflow) r.errors.push("horizontal page scroll at 390px width");
  await pp.screenshot({ path: `logs/e2e/${name}-phone-a1.png`, fullPage: true });
  await phone.close();
  await browser.close();
  if (r.errors.length) failed = true;
}
writeFileSync("logs/e2e/report.json", JSON.stringify(report, null, 1));
console.log(JSON.stringify(report, null, 1));
local?.server.close();
process.exit(failed ? 1 : 0);
