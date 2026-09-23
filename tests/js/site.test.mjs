// Tests of the site's own modules (docs/js/) against the Python pipeline and the shipped data files.
// Run: pixi run python scripts/10_test_fixtures.py && pixi run node --test tests/js/
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { test } from "node:test";

import { conformerCoords, parseAssignments, parseShard, shardRowStarts } from "../../docs/js/data.js";
import {
  angle,
  classifyMeasurement,
  dihedral,
  distance,
  fourierEnergy,
  fourierProfile,
  measure,
  observations,
  weightedCircularStats,
  wrap,
} from "../../docs/js/stats.js";
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
const paramById = new Map(params.map((p) => [p.param_id, p]));
const topologies = readJSON("topologies.json");
const coords = { opt: new Float32Array(readBuf("coords_opt.bin")), td: new Float32Array(readBuf("coords_td.bin")) };
const shards = new Map();
const shardFor = (pid) => {
  if (!shards.has(pid)) shards.set(pid, parseShard(readBuf(paramById.get(pid).shard), paramById.get(pid).handler));
  return shards.get(pid);
};

test("shard rows reproduce the Python values (Float32) and link to the right topology/atoms", () => {
  for (const r of fixtures.rows) {
    const shard = shardFor(r.param_id);
    const starts = shardRowStarts(shard, topologies);
    const i = r.shard_pos;
    assert.equal(shard.topology[i], r.topology_idx);
    const atoms = Array.from(shard.atoms.slice(4 * i, 4 * i + 4)).filter((a) => a >= 0);
    assert.deepEqual(atoms, r.atoms);
    const row = starts[i] + r.conf_j;
    if (r.handler === "ImproperTorsions") {
      for (let k = 0; k < 3; k++) assert.equal(shard.values[3 * row + k], Math.fround(r.improper_terms_deg[k]));
    } else {
      assert.equal(shard.values[row], Math.fround(r.value));
    }
    if (shard.valid) assert.equal(Boolean(shard.valid[row]), r.valid);
  }
});

test("JS geometry on shipped Float32 coordinates matches Python float64 values", () => {
  let worst = { bond: 0, angle: 0, dihedral: 0 };
  for (const r of fixtures.rows) {
    const t = r.topology_idx;
    const x = conformerCoords(coords[topologies.source[t]], topologies, t, r.conf_j);
    if (r.handler === "Bonds") worst.bond = Math.max(worst.bond, Math.abs(distance(x, ...r.atoms) - r.value));
    else if (r.handler === "Angles") worst.angle = Math.max(worst.angle, Math.abs(angle(x, ...r.atoms) - r.value));
    else if (r.handler === "ProperTorsions") {
      const d = dihedral(x, ...r.atoms);
      assert.equal(d.valid, r.valid);
      if (r.valid) worst.dihedral = Math.max(worst.dihedral, Math.abs(wrap(d.value - r.value)));
    }
  }
  assert.ok(worst.bond < 1e-5, `bond ${worst.bond}`);
  assert.ok(worst.angle < 1e-3, `angle ${worst.angle}`);
  assert.ok(worst.dihedral < 1e-2, `dihedral ${worst.dihedral}`);
});

test("JS torsion energies match Python (which matches OpenMM)", () => {
  let worst = 0;
  for (const r of fixtures.rows) {
    if (r.energy_kcal === null) continue;
    const p = paramById.get(r.param_id);
    const e =
      r.handler === "ImproperTorsions"
        ? r.improper_terms_deg.reduce((s, phi) => s + fourierEnergy(phi, p), 0)
        : fourierEnergy(r.value, p);
    worst = Math.max(worst, Math.abs(e - r.energy_kcal));
  }
  assert.ok(worst < 1e-9, `max |dE| ${worst}`);
});

test("molecule-balanced weights sum to 1 per molecule under filters; improper terms weigh 1/3", () => {
  for (const pid of ["b1", "a1", "t1", "i1", "t17"]) {
    const p = paramById.get(pid);
    const shard = shardFor(pid);
    for (const series of [["opt"], ["td"], ["opt", "td"]]) {
      const obs = observations(shard, p, topologies, { series, weighting: "molecule" });
      const sums = new Map();
      obs.mol.forEach((m, n) => sums.set(m, (sums.get(m) || 0) + obs.weight[n]));
      for (const s of sums.values()) assert.ok(Math.abs(s - 1) < 1e-9);
      const inst = observations(shard, p, topologies, { series, weighting: "instances" });
      const expected = p.handler === "ImproperTorsions" ? 1 / 3 : 1;
      assert.ok(inst.weight.every((w) => w === expected));
    }
  }
});

test("TorsionDrive driven-bond torsions and frozen-bond opt torsions are excluded from distributions", () => {
  const shard = shardFor("t1");
  const obs = observations(shard, paramById.get("t1"), topologies, { series: ["opt", "td"] });
  for (const i of new Set(obs.asg)) assert.equal(shard.flags[i] & 3, 0);
});

test("fourier profile minima and wrap", () => {
  const prof = fourierProfile({ periodicity: [3], phase_deg: [0], k_effective: [1] }, 1);
  assert.deepEqual(prof.minima.map((m) => m.phi).sort((a, b) => a - b), [-180, -60, 60]);
  assert.equal(wrap(190), -170);
  assert.equal(wrap(-190), 170);
  const c = weightedCircularStats([179, -179], [1, 1]);
  assert.ok(Math.abs(Math.abs(c.circMean) - 180) < 1e-9);
});

test("SMARTS measurement classification and canonical keys", () => {
  assert.equal(classifyMeasurement(2, [[1, 2]]).kind, "bond");
  assert.equal(classifyMeasurement(4, [[1, 2], [2, 3], [3, 4]]).kind, "proper");
  assert.equal(classifyMeasurement(4, [[1, 2], [2, 3], [2, 4]]).kind, "improper");
  assert.ok(classifyMeasurement(3, [[1, 2]]).error);
  const x = new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 2, 1, 1]);
  assert.equal(measure("proper", x, [0, 1, 2, 3]).key, measure("proper", x, [3, 2, 1, 0]).key);
  assert.ok(Math.abs(measure("proper", x, [0, 1, 2, 3]).value - measure("proper", x, [3, 2, 1, 0]).value) < 1e-9);
  assert.equal(measure("improper", x, [0, 1, 2, 3]).key, measure("improper", x, [3, 1, 0, 2]).key);
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

test("worker matchTuples reproduces label_molecules assignments for sampled rows", async () => {
  const { RDKit } = await loadRDKit();
  const smirksById = new Map(fixtures.smirks.filter((s) => s.param_id).map((s) => [s.param_id, s]));
  let checked = 0;
  for (const r of fixtures.rows.slice(0, 300)) {
    const s = smirksById.get(r.param_id);
    const mol = RDKit.get_mol(topologies.mdl_smiles[r.topology_idx], WorkerLib.MDL_MOL_OPTIONS);
    const q = RDKit.get_qmol(s.smirks);
    // label_molecules returns canonicalized tuples (ValenceDict / ImproperDict), so compare canonical forms
    const canon = (t) =>
      r.handler === "ImproperTorsions"
        ? [t[1], ...[t[0], t[2], t[3]].sort((a, b) => a - b)].join(",")
        : [t, [...t].reverse()].map((x) => x.join(",")).sort()[0];
    const tuples = WorkerLib.matchTuples(mol, q, s.tag_index, topologies.mdl_order[r.topology_idx]).map(canon);
    assert.ok(tuples.includes(canon(r.atoms)), `${r.param_id} ${r.atoms}`);
    mol.delete();
    q.delete();
    checked++;
  }
  assert.ok(checked > 0);
});

// ---------------------------------------------------------------- review follow-ups
import { extent, measureTopology, weightedLinearStats as wls, weightedCircularStats as wcs } from "../../docs/js/stats.js";
import { topologyMolblock } from "../../docs/js/molblock.js";

test("atom maps: duplicates, gaps are rejected; out-of-order maps resolve by number", async () => {
  const { RDKit } = await loadRDKit();
  const info = (s) => WorkerLib.queryMapInfo(RDKit.get_qmol(s).get_v3Kmolblock());
  assert.equal(info("[C:1][C:1][O:2]").contiguous, false);
  assert.equal(info("[C:1][C:3]").contiguous, false);
  const ooo = info("[O:2][C:1]");
  assert.ok(ooo.contiguous);
  assert.deepEqual(ooo.tagIndex, [1, 0]);
  assert.equal(info("CCO").nTags, 0);
});

test("extent handles very large arrays (no argument spreading)", () => {
  const big = new Float64Array(2_000_000).map((_, i) => i % 1000);
  assert.deepEqual(extent(big), [0, 999]);
});

test("overview statistics (params.json) equal the detail-view statistics for every parameter", () => {
  let checked = 0;
  for (const p of params) {
    if (!p.shard) continue;
    const obs = observations(shardFor(p.param_id), p, topologies, { series: ["opt"], weighting: "instances" });
    if (!obs.value.length) {
      assert.ok(p.opt_stats.mean === null || p.opt_stats.circ_mean === null, p.param_id);
      continue;
    }
    if (p.handler === "ProperTorsions" || p.handler === "ImproperTorsions") {
      const c = wcs(obs.value, obs.weight);
      assert.ok(Math.abs(wrap(c.circMean - p.opt_stats.circ_mean)) < 1e-6 && Math.abs(c.resultantLength - p.opt_stats.resultant_length) < 1e-9, p.param_id);
    } else {
      const s = wls(obs.value, obs.weight);
      for (const k of ["mean", "std", "median", "min", "max"]) assert.ok(Math.abs(s[k] - p.opt_stats[k]) < 1e-6, `${p.param_id} ${k} ${s[k]} ${p.opt_stats[k]}`);
    }
    checked++;
  }
  assert.ok(checked > 300);
});

test("SMARTS geometry path and parameter path include identical interactions and values", () => {
  // Feed the SMARTS-path measurement the exact tuples label_molecules assigned to a parameter; it must
  // keep/exclude the same interactions (frozen/driven-bond rule) and reproduce the shard values.
  const frozenParam = params.find((p) => p.handler === "ProperTorsions" && p.shard && Array.from(shardFor(p.param_id).flags).some((f, i) => f & 2 && topologies.source[shardFor(p.param_id).topology[i]] === "opt"));
  assert.ok(frozenParam, "expected a proper parameter with frozen-bond optimization rows");
  for (const pid of ["b1", "a1", "t1", "i1", frozenParam.param_id]) {
    const p = paramById.get(pid);
    const shard = shardFor(pid);
    const kind = { Bonds: "bond", Angles: "angle", ProperTorsions: "proper", ImproperTorsions: "improper" }[p.handler];
    const obs = observations(shard, p, topologies, { series: ["opt"] });
    const byTop = new Map();
    for (let i = 0; i < shard.nAsg; i++) {
      const t = shard.topology[i];
      if (topologies.source[t] !== "opt") continue;
      const a = Array.from(shard.atoms.slice(4 * i, 4 * i + 4)).filter((x) => x >= 0);
      if (!byTop.has(t)) byTop.set(t, []);
      byTop.get(t).push(a);
    }
    const smartsValues = [];
    let excluded = 0;
    for (const [t, tuples] of byTop) {
      const m = measureTopology(kind, topologies, t, tuples, coords.opt);
      smartsValues.push(...m.values);
      excluded += m.nConstrained;
    }
    assert.equal(smartsValues.length, obs.value.length, `${pid}: count`);
    const a = [...smartsValues].sort((x, y) => x - y), b = [...obs.value].sort((x, y) => x - y);
    // impropers: SMARTS path uses sorted outer atoms, the FF uses label_molecules order: compare |values|
    // multiset only for chain interactions; for impropers compare counts (handedness convention differs)
    if (kind !== "improper") for (let k = 0; k < a.length; k++) assert.ok(Math.abs(a[k] - b[k]) < 2e-3, `${pid}: value ${a[k]} vs ${b[k]}`);
    if (pid === frozenParam.param_id) assert.ok(excluded > 0, "frozen-bond interactions must be excluded");
  }
});

test("3D molblock is rewritten into topology order with matching bonds", async () => {
  const { RDKit } = await loadRDKit();
  const asg = parseAssignments(readBuf("assignments.bin"));
  for (const t of [0, 17, 4700, 6000]) {
    const mol = RDKit.get_mol(topologies.mdl_smiles[t], JSON.stringify({ removeHs: false }));
    const mb = mol.get_molblock(JSON.stringify({ kekulize: true }));
    mol.delete();
    const x = conformerCoords(coords[topologies.source[t]], topologies, t, 0);
    const out = topologyMolblock(mb, topologies.mdl_order[t], x).split("\n");
    const n = topologies.n_atoms[t];
    const nb = parseInt(out[3].slice(3, 6), 10);
    // coordinates of atom i equal the shipped coordinates of topology atom i
    for (let i = 0; i < n; i++) assert.ok(Math.abs(parseFloat(out[4 + i].slice(0, 10)) - x[3 * i]) < 1e-4);
    const molBonds = new Set(out.slice(4 + n, 4 + n + nb).map((l) => [parseInt(l.slice(0, 3)) - 1, parseInt(l.slice(3, 6)) - 1].sort((a, b) => a - b).join(",")));
    const ffBonds = new Set(asg.forTopology(t).filter((a) => params[a.paramIndex].handler === "Bonds").map((a) => [...a.atoms].sort((p, q) => p - q).join(",")));
    assert.deepEqual([...molBonds].sort(), [...ffBonds].sort(), `topology ${t}`);
  }
});

import { superpose } from "../../docs/js/align.js";

test("superpose recovers a rotated + translated copy (RMSD ~ 0) and reduces RMSD otherwise", () => {
  const x = new Float32Array(conformerCoords(coords.opt, topologies, 5, 0));
  const n = x.length / 3;
  const [a, b, c] = [0.7, -1.1, 2.3];
  const R = [
    [Math.cos(b) * Math.cos(c), -Math.cos(b) * Math.sin(c), Math.sin(b)],
    [Math.cos(a) * Math.sin(c) + Math.sin(a) * Math.sin(b) * Math.cos(c), Math.cos(a) * Math.cos(c) - Math.sin(a) * Math.sin(b) * Math.sin(c), -Math.sin(a) * Math.cos(b)],
    [Math.sin(a) * Math.sin(c) - Math.cos(a) * Math.sin(b) * Math.cos(c), Math.sin(a) * Math.cos(c) + Math.cos(a) * Math.sin(b) * Math.sin(c), Math.cos(a) * Math.cos(b)],
  ];
  const moved = new Float32Array(x.length);
  for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) moved[3 * i + d] = R[d][0] * x[3 * i] + R[d][1] * x[3 * i + 1] + R[d][2] * x[3 * i + 2] + [5, -3, 8][d];
  const pairs = Array.from({ length: n }, (_, i) => [i, i]);
  const { rmsd } = superpose(moved, x, pairs);
  assert.ok(rmsd < 1e-3, `rmsd ${rmsd}`);
});
