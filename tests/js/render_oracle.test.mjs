// Rendering oracle (PLAN_precompute.md U1, oracle B): the browser renders PRECOMPUTED depictions without
// laying them out. For each sampled record/variant/highlight set, RDKit.js (worker-lib renderDepiction, the
// exact code the worker runs) must produce the same SVG as Python RDKit from the same molblock.
// Negative controls: a shifted layout and a different highlight set must be rejected.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { test } from "node:test";
import { loadRDKit } from "../../scripts/js/rdkit.mjs";

const require = createRequire(import.meta.url);
const WorkerLib = require("../../docs/js/classic/worker-lib.js");
const DATA = new URL("../../docs/data/", import.meta.url);
const fx = JSON.parse(readFileSync(new URL("render_fixtures.json", import.meta.url), "utf8"));
const shards = new Map();
const depiction = (t) => {
  const k = Math.floor(t / 500);
  if (!shards.has(k)) shards.set(k, JSON.parse(readFileSync(new URL(`depictions/${k}.json`, DATA), "utf8")));
  const s = shards.get(k);
  return { h: s.molblock_h[t - s.first], heavy: s.molblock_heavy[t - s.first] };
};
const norm = (s) => s.replace(/<\?xml[^>]*>/, "").replace(/\s+/g, " ").trim();

test("RDKit.js renders every sampled precomputed depiction identically to Python RDKit", async () => {
  const { RDKit } = await loadRDKit();
  let n = 0;
  for (const c of fx.cases) {
    const mb = depiction(c.t)[c.variant];
    const svg = WorkerLib.renderDepiction(RDKit, mb, c.atoms, c.bonds, 260, 190);
    assert.equal(norm(svg), norm(c.svg), `record ${c.t} ${c.variant} ${JSON.stringify(c.atoms)}`);
    n++;
  }
  assert.ok(n > 400 && fx.kept_h.length > 0);
});

test("negative controls: shifted layout and wrong highlights are rejected; coordinate-free input refused", async () => {
  const { RDKit } = await loadRDKit();
  for (const c of fx.cases.filter((x) => x.atoms.length).slice(0, 60)) {
    const mb = depiction(c.t)[c.variant];
    const lines = mb.split("\n");
    lines[4] = (parseFloat(lines[4].slice(0, 10)) + 0.7).toFixed(4).padStart(10) + lines[4].slice(10);
    assert.notEqual(norm(WorkerLib.renderDepiction(RDKit, lines.join("\n"), c.atoms, c.bonds, 260, 190)), norm(c.svg));
    assert.notEqual(norm(WorkerLib.renderDepiction(RDKit, mb, [c.atoms[0]], [], 260, 190)), norm(c.svg));
  }
  // a molblock whose atoms all sit at the origin has no usable layout: RDKit.js reports no coordinates
  const flat = depiction(0).h.split("\n").map((l, i) => (i >= 4 && /^\s+-?\d+\.\d{4}\s+-?\d+\.\d{4}/.test(l) ? "    0.0000    0.0000    0.0000" + l.slice(30) : l)).join("\n");
  assert.throws(() => WorkerLib.renderDepiction(RDKit, flat, [], [], 260, 190), /no 2D coordinates/);
  // a coordinate-free input (SMILES) is refused instead of being laid out
  assert.throws(() => WorkerLib.renderDepiction(RDKit, "CCO", [], [], 260, 190), /no 2D coordinates/);
});
