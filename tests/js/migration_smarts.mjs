// Migration oracle T1 (PLAN_precompute.md): run the CURRENT browser SMARTS-geometry code (measure,
// constrainedCentralBonds, parameter lookup via measure keys) on every tuple of the Python geometry universe
// and compare keys, values, validity, frozen-bond exclusion and parameter references.
import { readFileSync, writeFileSync } from "node:fs";
import { conformerCoords, parseAssignments } from "../../docs/js/data.js";
import { constrainedCentralBonds, measure } from "./legacy_stats.mjs";

const DATA = new URL("../../docs/data/", import.meta.url);
const readJSON = (n) => JSON.parse(readFileSync(new URL(n, DATA), "utf8"));
const readBuf = (n) => {
  const b = readFileSync(new URL(n, DATA));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
const T = readJSON("topologies.json");
const params = readJSON("params.json");
const coords = new Float32Array(readBuf("coords_opt.bin"));
const asg = parseAssignments(readBuf("assignments.bin"));
const U = JSON.parse(readFileSync("data/processed/geometry_universe.json", "utf8"));
const HANDLER = { bond: "Bonds", angle: "Angles", proper: "ProperTorsions", improper: "ImproperTorsions" };
const wrap = (d) => ((((d + 180) % 360) + 360) % 360) - 180;

const report = { n: U.kind.length, byKind: {}, differences: [] };
const oldAssigned = new Map(); // topology -> Map(kind:key -> param index), built exactly as app.js does
function assignedFor(t, x) {
  if (!oldAssigned.has(t)) {
    const m = new Map();
    for (const a of asg.forTopology(t)) {
      const kind = Object.keys(HANDLER).find((k) => HANDLER[k] === params[a.paramIndex].handler);
      m.set(`${kind}:${measure(kind, x, a.atoms).key}`, a.paramIndex);
    }
    oldAssigned.set(t, m);
  }
  return oldAssigned.get(t);
}
const bump = (kind, field) => {
  report.byKind[kind] ??= { n: 0, key: 0, value: 0, valid: 0, frozen: 0, param: 0, maxAbs: 0 };
  report.byKind[kind][field]++;
};
for (let r = 0; r < U.kind.length; r++) {
  const kind = U.kind[r], t = U.topology_idx[r], tup = U.atoms[r];
  const x = conformerCoords(coords, T, t, 0);
  bump(kind, "n");
  // the old SMARTS path receives tag order (:1, :2 central, :3, :4); the universe stores central first
  const m = measure(kind, x, kind === "improper" ? [tup[1], tup[0], tup[2], tup[3]] : tup);
  const oldVals = Array.isArray(m.value) ? m.value : [m.value];
  const newKey = tup.join(",");
  if (m.key !== newKey) (bump(kind, "key"), report.differences.length < 20 && report.differences.push({ r, kind, field: "key", old: m.key, new: newKey }));
  if (m.valid !== U.valid[r]) (bump(kind, "valid"), report.differences.length < 20 && report.differences.push({ r, kind, field: "valid", old: m.valid, new: U.valid[r] }));
  if (U.valid[r]) {
    for (let k = 0; k < oldVals.length; k++) {
      const d = kind === "bond" || kind === "angle" ? Math.abs(oldVals[k] - U.values[r][k]) : Math.abs(wrap(oldVals[k] - U.values[r][k]));
      report.byKind[kind].maxAbs = Math.max(report.byKind[kind].maxAbs, d);
      if (d > (kind === "bond" ? 1e-4 : 1e-3)) (bump(kind, "value"), report.differences.length < 20 && report.differences.push({ r, kind, field: "value", old: oldVals[k], new: U.values[r][k] }));
    }
  }
  if (kind === "proper") {
    const frozenOld = constrainedCentralBonds(T, t).has([tup[1], tup[2]].sort((a, b) => a - b).join(","));
    if (frozenOld !== U.on_frozen_bond[r]) bump(kind, "frozen");
  }
  const pOld = assignedFor(t, x).get(`${kind}:${m.key}`) ?? -1;
  if (kind !== "improper" && pOld !== U.param_idx[r]) (bump(kind, "param"), report.differences.length < 20 && report.differences.push({ r, kind, field: "param", old: pOld, new: U.param_idx[r] }));
}
writeFileSync("logs/migration_smarts_report.json", JSON.stringify(report, null, 1));
console.log(JSON.stringify(report.byKind, null, 1), "\nfirst differences:", JSON.stringify(report.differences.slice(0, 8)));
