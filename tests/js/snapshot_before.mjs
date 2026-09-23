// Migration oracle (PLAN_precompute.md, S4/T1): run the CURRENT browser code paths in node and record
// what the page displays, BEFORE those paths are replaced by Python-precomputed data.
// Output: tests/js/snapshot_before.json (parameter pages) and tests/js/snapshot_smarts_before.json
// (SMARTS geometry for every tuple of the optimization geometry universe is added in a later step).
import { readFileSync, writeFileSync } from "node:fs";
import { parseShard } from "./legacy_data.mjs";
import {
  drivenScans,
  fourierProfile,
  histogram,
  observations,
  weightedCircularStats,
  weightedLinearStats,
} from "./legacy_stats.mjs";

const DATA = new URL("../../docs/data/", import.meta.url);
const readJSON = (n) => JSON.parse(readFileSync(new URL(n, DATA), "utf8"));
const readBuf = (n) => {
  const b = readFileSync(new URL(n, DATA));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
const params = readJSON("params.json");
const topologies = readJSON("topologies.json");
const conformers = readJSON("conformers.json");

// Same binning rules as docs/js/app.js renderParam (optimization series only)
function bins(p, values) {
  if (p.handler === "ProperTorsions" || p.handler === "ImproperTorsions") return [-180, 180, 72];
  let mn = Infinity, mx = -Infinity;
  for (const v of values) (mn = Math.min(mn, v)), (mx = Math.max(mx, v));
  const centre = p.handler === "Bonds" ? p.length_angstrom : p.angle_deg;
  mn = Math.min(mn, centre);
  mx = Math.max(mx, centre);
  const pad = (mx - mn) * 0.05 || (p.handler === "Bonds" ? 0.01 : 1);
  return [mn - pad, mx + pad, 60];
}

const out = {};
for (const p of params) {
  const rec = { handler: p.handler };
  if (!p.shard) {
    out[p.param_id] = { ...rec, empty: true };
    continue;
  }
  const shard = parseShard(readBuf(p.shard), p.handler);
  const isTorsion = shard.valid !== null;
  for (const weighting of ["instances", "molecule"]) {
    const obs = observations(shard, p, topologies, { series: ["opt"], weighting });
    if (!obs.value.length) {
      rec[weighting] = { n_obs: 0 };
      continue;
    }
    const [lo, hi, nb] = bins(p, obs.value);
    const hist = histogram(obs.value, obs.weight, lo, hi, nb);
    const stats = isTorsion ? weightedCircularStats(obs.value, obs.weight) : weightedLinearStats(obs.value, obs.weight);
    rec[weighting] = { n_obs: obs.value.length, n_mol: new Set(obs.mol).size, lo, hi, nb, counts: hist.counts, stats };
  }
  if (isTorsion) {
    const prof = fourierProfile(p, 1);
    rec.profile = { y: prof.y, minima: prof.minima };
  }
  // examples: first assignment per record, split by source (renderExamples)
  const first = new Map();
  for (let i = 0; i < shard.nAsg; i++) if (!first.has(shard.topology[i])) first.set(shard.topology[i], i);
  rec.examples = [...first].map(([t, i]) => [t, Array.from(shard.atoms.slice(4 * i, 4 * i + 4)).filter((a) => a >= 0), topologies.source[t]]);
  // driven scans with frames sorted by grid angle (renderScans / viewer3d)
  if (isTorsion) {
    rec.scans = drivenScans(shard, topologies).map((s) => {
      const t = s.topology;
      const frames = [];
      for (let j = 0; j < topologies.n_conf[t]; j++) {
        const c = topologies.conf_start[t] + j;
        frames.push([conformers.grid_deg[c], conformers.rel_energy_kcal[c]]);
      }
      frames.sort((a, b) => a[0] - b[0]);
      return { t, atoms: Array.from(shard.atoms.slice(4 * s.asg, 4 * s.asg + 4)), frames };
    });
  }
  out[p.param_id] = rec;
}
writeFileSync(new URL("snapshot_before.json", import.meta.url), JSON.stringify(out));
console.log(`snapshot of ${Object.keys(out).length} parameters written`);
