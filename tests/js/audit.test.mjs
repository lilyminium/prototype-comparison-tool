// Static audit (PLAN_precompute.md R6/S1): the page's own JavaScript must not compute anything that can be
// precomputed. Behavioural checks complement this (render_oracle.test.mjs, e2e); this test stops
// geometry/alignment/layout code from creeping back in. Narrow allowlist, documented per rule.
import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../docs/js");
const files = [];
(function walk(dir) {
  for (const f of readdirSync(dir)) {
    const p = path.join(dir, f);
    if (statSync(p).isDirectory()) walk(p);
    else if (p.endsWith(".js")) files.push(p);
  }
})(ROOT);
const src = Object.fromEntries(files.map((f) => [path.relative(ROOT, f), readFileSync(f, "utf8")]));

/** Occurrences of `pattern` in each file, optionally restricted to text outside an allowed function body. */
function occurrences(pattern) {
  const out = [];
  for (const [file, text] of Object.entries(src)) {
    for (const m of text.matchAll(pattern)) out.push({ file, index: m.index, line: text.slice(0, m.index).split("\n").length });
  }
  return out;
}
function insideFunction(file, index, name) {
  const text = src[file];
  const start = text.indexOf(`function ${name}(`);
  if (start < 0 || index < start) return false;
  // the body ends at the first closing brace at the function's own indentation
  const lineStart = text.lastIndexOf("\n", start) + 1;
  const indent = text.slice(lineStart, start).match(/^\s*/)[0];
  const end = text.indexOf(`\n${indent}}\n`, start);
  return end > 0 && index < end;
}

test("no geometry, alignment or coordinate generation in docs/js", () => {
  const forbidden = [
    /\bsuperpose\b/, /\bkabsch\b/i, /\bcross\(/, /\bdihedral\(/, /\bangle\(/, /\bdistance\(/, /\bmeasure(Topology)?\(/,
    /\bfourier(Energy|Profile)\(/, /\bobservations\(/, /Math\.acos/,
    /set_new_coords/, /generate_aligned_coords/, /get_new_coords/, /straighten_depiction/, /normalize_depiction/,
    /get_molblock\(/, /get_v2Kmolblock/,
  ];
  for (const re of forbidden) {
    const hits = occurrences(new RegExp(re.source, re.flags.includes("g") ? re.flags : re.flags + "g"));
    assert.deepEqual(hits, [], `forbidden ${re}: ${JSON.stringify(hits)}`);
  }
});

test("V3000 molblocks are only read from the SMARTS query (atom-map numbers), never generated for data", () => {
  for (const h of occurrences(/get_v3Kmolblock\(/g)) {
    const line = src[h.file].split("\n")[h.line - 1];
    assert.match(line, /qmol\.get_v3Kmolblock\(\)/, `${h.file}:${h.line}`);
  }
});

test("trigonometry only inside weightedCircularStats (statistics of looked-up dihedrals)", () => {
  for (const re of [/Math\.atan2/g, /Math\.hypot/g, /Math\.(cos|sin)\(/g]) {
    for (const h of occurrences(re)) assert.ok(h.file === "stats.js" && insideFunction("stats.js", h.index, "weightedCircularStats"), `${re} at ${h.file}:${h.line}`);
  }
});

test("RDKit.js drawing: only precomputed depictions, plus the user's own query molecule", () => {
  const svg = occurrences(/\.get_svg\(/g);
  assert.equal(svg.length, 1, JSON.stringify(svg));
  assert.ok(svg[0].file === "classic/worker.js" && /the query itself/.test(src["classic/worker.js"].split("\n")[svg[0].line - 1]));
  const hl = occurrences(/get_svg_with_highlights/g);
  assert.equal(hl.length, 1);
  assert.equal(hl[0].file, "classic/worker-lib.js");
  assert.ok(insideFunction("classic/worker-lib.js", hl[0].index, "renderDepiction"));
  assert.match(src["classic/worker-lib.js"], /has_coords\(\)/, "render path must require precomputed coordinates");
});

test("removed modules stay removed", () => {
  assert.ok(!("align.js" in src));
  assert.ok(!/topologyMolblock/.test(Object.values(src).join("\n")));
});
