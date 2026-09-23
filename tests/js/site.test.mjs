// Tests of the site's own modules (docs/js/) against the Python pipeline and the shipped data files.
// Run: pixi run python scripts/10_test_fixtures.py && pixi run node --test tests/js/
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { test } from "node:test";

import { conformerCoords, GEOMETRY_KINDS, parseAssignments, parseGeometry } from "../../docs/js/data.js";
import { withCoordinates } from "../../docs/js/molblock.js";
import { classifyMeasurement, extent, histogram, tanimotoAll, universeKey, weightedCircularStats, weightedLinearStats } from "../../docs/js/stats.js";
import { loadRDKit } from "../../scripts/js/rdkit.mjs";

const require = createRequire(import.meta.url);
const WorkerLib = require("../../docs/js/classic/worker-lib.js");

const DATA = new URL("../../docs/data/", import.meta.url);
const readJSON = (name) => JSON.parse(readFileSync(new URL(name, DATA), "utf8"));
const readBuf = (name) => {
  const b = readFileSync(new URL(name, DATA));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
const fixtures = JSON.parse(readFileSync(new URL("fixtures.json", import.meta.url), "utf8"));
const params = readJSON("params.json");
const topologies = readJSON("topologies.json");

test("geom_opt.bin decodes every sampled universe row exactly (atoms, values, validity, flags, parameter)", () => {
  const G = parseGeometry(readBuf("geom_opt.bin"));
  assert.equal(G.nTop, topologies.source.length);
  for (const r of fixtures.geometry) {
    const row = G.lookup(r.kind, r.topology_idx).get(r.atoms.join(","));
    assert.ok(row, `${r.kind} ${r.atoms} in ${r.topology_idx}`);
    assert.equal(row.values.length, r.kind === "improper" ? 3 : 1);
    row.values.forEach((v, k) => assert.ok(Number.isNaN(v) ? Number.isNaN(r.values[k]) : v === Math.fround(r.values[k])));
    assert.equal(row.valid, r.valid);
    assert.equal(row.frozen, r.frozen);
    assert.equal(row.param, r.param_idx);
  }
  for (const t of fixtures.opt_first_last) for (const [kind] of GEOMETRY_KINDS) assert.equal(G.lookup(kind, t).size, fixtures.geometry_counts[`${t}:${kind}`] ?? 0, `${t} ${kind}`);
  const td = topologies.source.indexOf("td");
  for (const [kind] of GEOMETRY_KINDS) assert.equal(G.lookup(kind, td).size, 0);
  const handler = { bond: "Bonds", angle: "Angles", proper: "ProperTorsions", improper: "ImproperTorsions" };
  for (const r of fixtures.geometry) if (r.param_idx >= 0) assert.equal(params[r.param_idx].handler, handler[r.kind]);
});

test("universe keys: reversed chains and every outer-atom permutation of an improper give the same key", () => {
  assert.equal(universeKey("bond", [7, 3]), universeKey("bond", [3, 7]));
  assert.equal(universeKey("angle", [9, 2, 4]), "4,2,9");
  assert.equal(universeKey("proper", [8, 1, 2, 3]), universeKey("proper", [3, 2, 1, 8]));
  assert.equal(universeKey("proper", [3, 1, 2, 3]), universeKey("proper", [3, 2, 1, 3]));
  const outer = [[0, 2, 3], [0, 3, 2], [2, 0, 3], [2, 3, 0], [3, 0, 2], [3, 2, 0]];
  assert.deepEqual([...new Set(outer.map(([a, b, d]) => universeKey("improper", [a, 1, b, d])))], ["1,0,2,3"]);
});

test("every SMARTS-measurable tuple of sampled records exists in the universe (no geometry computed in JS)", async () => {
  const { RDKit } = await loadRDKit();
  const G = parseGeometry(readBuf("geom_opt.bin"));
  const queries = { bond: "[*:1]~[*:2]", angle: "[*:1]~[*:2]~[*:3]", proper: "[*:1]~[*:2]~[*:3]~[*:4]", improper: "[*:1]~[*:2](~[*:3])~[*:4]" };
  const optTops = topologies.source.flatMap((s, t) => (s === "opt" ? [t] : [])).filter((_, i) => i % 97 === 0);
  for (const [kind, smarts] of Object.entries(queries)) {
    const q = RDKit.get_qmol(smarts);
    const info = WorkerLib.queryMapInfo(q.get_v3Kmolblock());
    for (const t of optTops) {
      const mol = RDKit.get_mol(topologies.mdl_smiles[t], WorkerLib.MDL_MOL_OPTIONS);
      const table = G.lookup(kind, t);
      const keys = new Set(WorkerLib.matchTuples(mol, q, info.tagIndex, topologies.mdl_order[t]).map((tup) => universeKey(kind, tup)));
      for (const k of keys) assert.ok(table.has(k), `${kind} ${k} missing in ${t}`);
      assert.equal(keys.size, table.size, `${kind} count in ${t}`);
      mol.delete();
    }
    q.delete();
  }
});

test("assignments.bin round-trips for sampled topologies", () => {
  const asg = parseAssignments(readBuf("assignments.bin"));
  const byTop = new Map();
  for (const r of fixtures.rows) {
    if (!byTop.has(r.topology_idx)) byTop.set(r.topology_idx, []);
    byTop.get(r.topology_idx).push(r);
  }
  for (const [t, rows] of byTop) {
    const list = asg.forTopology(t).map((a) => `${params[a.paramIndex].param_id}:${a.atoms.join(",")}`);
    for (const r of rows) assert.ok(list.includes(`${r.param_id}:${r.atoms.join(",")}`));
  }
});

test("overview statistics (params.json) equal the parameter-page statistics (param/<id>.json)", () => {
  let checked = 0;
  for (const p of params) {
    if (!p.detail) continue;
    const w = JSON.parse(readFileSync(new URL(p.detail, DATA), "utf8")).weightings.instances;
    if (!w.n_obs) continue;
    if (p.handler === "ProperTorsions" || p.handler === "ImproperTorsions") {
      assert.ok(Math.abs(p.opt_stats.resultant_length - w.stats.resultantLength) < 1e-9, p.param_id);
    } else {
      for (const k of ["mean", "std", "median", "min", "max"]) assert.ok(Math.abs(p.opt_stats[k] - w.stats[k]) < 1e-9 * Math.max(1, Math.abs(w.stats[k])), `${p.param_id} ${k}`);
    }
    checked++;
  }
  assert.ok(checked > 300);
});

test("RDKit.js query map parsing (V3000) equals OpenFF's tag indices for every SMIRKS", async () => {
  const { RDKit } = await loadRDKit();
  for (const s of fixtures.smirks) {
    const q = RDKit.get_qmol(s.smirks);
    const info = WorkerLib.queryMapInfo(q.get_v3Kmolblock());
    assert.deepEqual(info.tagIndex, s.tag_index, s.smirks);
    assert.ok(info.contiguous);
    q.delete();
  }
});

test("atom maps: duplicates and gaps are rejected; out-of-order maps resolve by number", async () => {
  const { RDKit } = await loadRDKit();
  const info = (s) => WorkerLib.queryMapInfo(RDKit.get_qmol(s).get_v3Kmolblock());
  assert.equal(info("[C:1][C:1][O:2]").contiguous, false);
  assert.equal(info("[C:1][C:3]").contiguous, false);
  const ooo = info("[O:2][C:1]");
  assert.ok(ooo.contiguous);
  assert.deepEqual(ooo.tagIndex, [1, 0]);
  assert.equal(info("CCO").nTags, 0);
});

test("worker matchTuples reproduces label_molecules assignments for sampled rows", async () => {
  const { RDKit } = await loadRDKit();
  const smirksById = new Map(fixtures.smirks.filter((s) => s.param_id).map((s) => [s.param_id, s]));
  for (const r of fixtures.rows.slice(0, 300)) {
    const s = smirksById.get(r.param_id);
    const mol = RDKit.get_mol(topologies.mdl_smiles[r.topology_idx], WorkerLib.MDL_MOL_OPTIONS);
    const q = RDKit.get_qmol(s.smirks);
    const canon = (t) =>
      r.handler === "ImproperTorsions" ? [t[1], ...[t[0], t[2], t[3]].sort((a, b) => a - b)].join(",") : [t, [...t].reverse()].map((x) => x.join(",")).sort()[0];
    const tuples = WorkerLib.matchTuples(mol, q, s.tag_index, topologies.mdl_order[r.topology_idx]).map(canon);
    assert.ok(tuples.includes(canon(r.atoms)), `${r.param_id} ${r.atoms}`);
    mol.delete();
    q.delete();
  }
});

test("query-dependent aggregation helpers", () => {
  const big = new Float64Array(2_000_000).map((_, i) => i % 1000);
  assert.deepEqual(extent(big), [0, 999]);
  const c = weightedCircularStats([179, -179], [1, 1]);
  assert.ok(Math.abs(Math.abs(c.circMean) - 180) < 1e-9);
  const s = weightedLinearStats([1, 2, 3, 4], [1, 1, 1, 1]);
  assert.equal(s.mean, 2.5);
  assert.equal(s.median, 2);
  assert.deepEqual(histogram([0, 0.5, 1], [1, 1, 1], 0, 1, 2).counts, [1, 2]);
  assert.equal(classifyMeasurement(4, [[1, 2], [2, 3], [2, 4]]).kind, "improper");
  assert.ok(classifyMeasurement(3, [[1, 2]]).error);
});

test("Tanimoto with precomputed popcounts equals a direct computation", () => {
  const fps = new Uint8Array(readBuf("fp_morgan.bin"));
  const pop = new Uint16Array(readBuf("fp_popcount.bin"));
  const q = fps.slice(256 * 17, 256 * 18);
  const sim = tanimotoAll(q, fps, pop);
  assert.equal(sim[17], 1);
  const bits = (a) => a.reduce((s, v) => s + [...v.toString(2)].filter((c) => c === "1").length, 0);
  for (const m of [0, 100, 5000]) {
    const row = fps.slice(256 * m, 256 * m + 256);
    const inter = bits(row.map((v, i) => v & q[i]));
    assert.ok(Math.abs(sim[m] - inter / (bits(q) + bits(row) - inter)) < 1e-6);
  }
});

test("3D molblock: precomputed depiction + QM coordinates keeps topology atom order and bonds", () => {
  const asg = parseAssignments(readBuf("assignments.bin"));
  const coords = { opt: new Float32Array(readBuf("coords_opt.bin")), td: new Float32Array(readBuf("coords_td.bin")) };
  for (const t of [0, 17, 4700, 6000]) {
    const shard = JSON.parse(readFileSync(new URL(`depictions/${Math.floor(t / 500)}.json`, DATA), "utf8"));
    const i = t - shard.first;
    const x = conformerCoords(coords[topologies.source[t]], topologies, t, 0);
    const out = withCoordinates(shard.molblock_h[i], x).split("\n");
    const n = topologies.n_atoms[t];
    for (let a = 0; a < n; a++) assert.ok(Math.abs(parseFloat(out[4 + a].slice(0, 10)) - x[3 * a]) < 1e-4);
    const molBonds = new Set(shard.bonds_h[i].map(([a, b]) => [a, b].sort((p, q) => p - q).join(",")));
    const ffBonds = new Set(asg.forTopology(t).filter((a) => params[a.paramIndex].handler === "Bonds").map((a) => [...a.atoms].sort((p, q) => p - q).join(",")));
    assert.deepEqual([...molBonds].sort(), [...ffBonds].sort(), `topology ${t}`);
  }
});
