// PRE-MIGRATION browser computations, kept only as the oracle for tests/js/migration*.mjs.
// Not served by the site (PLAN_precompute.md).
// Geometry, torsion energies, weighting and summary statistics. No DOM; tested in tests/js/.
// Conventions match the Python pipeline (scripts/05_geometry.py), which is validated against OpenMM.
import { FLAG_IS_DRIVEN_TORSION, FLAG_ON_DRIVEN_BOND, FLAG_ON_FROZEN_BOND, shardRowStarts } from "./legacy_data.mjs";

export const SIN_THRESHOLD = 0.02; // dihedral undefined when either bond angle is within ~1.15° of 0/180
const DEG = 180 / Math.PI;

const sub = (x, i, j) => [x[3 * i] - x[3 * j], x[3 * i + 1] - x[3 * j + 1], x[3 * i + 2] - x[3 * j + 2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => Math.sqrt(dot(a, a));

export function distance(x, i, j) {
  return norm(sub(x, i, j));
}

export function angle(x, i, j, k) {
  const a = sub(x, i, j), b = sub(x, k, j);
  return Math.acos(Math.max(-1, Math.min(1, dot(a, b) / (norm(a) * norm(b))))) * DEG;
}

/** Signed dihedral (IUPAC/OpenMM) and validity (not near-linear). */
export function dihedral(x, i, j, k, l) {
  const b0 = sub(x, j, i), b1 = sub(x, k, j), b2 = sub(x, l, k);
  const n1 = cross(b0, b1), n2 = cross(b1, b2);
  const nb1 = norm(b1);
  const y = dot(cross(n1, n2), [b1[0] / nb1, b1[1] / nb1, b1[2] / nb1]);
  const sin1 = norm(n1) / (norm(b0) * nb1), sin2 = norm(n2) / (nb1 * norm(b2));
  return { value: Math.atan2(y, dot(n1, n2)) * DEG, valid: Math.min(sin1, sin2) >= SIN_THRESHOLD };
}

/** sum_n k_eff (1 + cos(n phi - phase)); kcal/mol. */
export function fourierEnergy(phiDeg, param) {
  let e = 0;
  for (let n = 0; n < param.periodicity.length; n++) {
    e += param.k_effective[n] * (1 + Math.cos((param.periodicity[n] * phiDeg - param.phase_deg[n]) / DEG));
  }
  return e;
}

/** Profile of a torsion parameter on a grid, with numerically located minima. */
export function fourierProfile(param, step = 1) {
  const x = [], y = [];
  for (let phi = -180; phi <= 180 + 1e-9; phi += step) {
    x.push(phi);
    y.push(fourierEnergy(phi, param));
  }
  const minima = [];
  const n = x.length - 1; // periodic: last point duplicates the first
  for (let i = 0; i < n; i++) {
    const prev = y[(i - 1 + n) % n], next = y[(i + 1) % n];
    if (y[i] < prev && y[i] <= next) minima.push({ phi: x[i], energy: y[i] });
  }
  return { x, y, minima };
}

export const wrap = (d) => ((((d + 180) % 360) + 360) % 360) - 180;

/**
 * Observations of one parameter as flat arrays, filtered and weighted.
 * options.series: which series to include: "opt" (optimization minima, frozen-bond rows excluded),
 *   "td" (grid-conditioned TorsionDrive observations, driven-bond torsions excluded).
 * options.weighting: "instances" (each observation weight 1; impropers 1/3 per term) or
 *   "molecule" (each molecule's included observations sum to 1).
 * Returns {value, weight, series, asg, row, mol} with one entry per plotted value (impropers: 3 per row).
 */
export function observations(shard, param, topologies, options) {
  const { series = ["opt"], weighting = "instances" } = options;
  const includeOpt = series.includes("opt"), includeTd = series.includes("td");
  const starts = shardRowStarts(shard, topologies);
  const isTorsion = shard.valid !== null;
  const width = shard.width;
  const out = { value: [], weight: [], series: [], asg: [], row: [], mol: [] };
  for (let i = 0; i < shard.nAsg; i++) {
    const t = shard.topology[i];
    const src = topologies.source[t];
    const f = shard.flags[i];
    if (src === "opt" && (!includeOpt || f & FLAG_ON_FROZEN_BOND)) continue;
    if (src === "td" && (!includeTd || f & FLAG_ON_DRIVEN_BOND)) continue;
    for (let r = starts[i]; r < starts[i + 1]; r++) {
      if (isTorsion && !shard.valid[r]) continue;
      for (let w = 0; w < width; w++) {
        out.value.push(shard.values[r * width + w]);
        out.weight.push(1 / width);
        out.series.push(src);
        out.asg.push(i);
        out.row.push(r);
        out.mol.push(topologies.mol_idx[t]);
      }
    }
  }
  if (weighting === "molecule") {
    const total = new Map();
    out.mol.forEach((m, n) => total.set(m, (total.get(m) || 0) + out.weight[n]));
    out.weight = out.weight.map((w, n) => w / total.get(out.mol[n]));
  }
  out.isTorsion = isTorsion;
  return out;
}

/** Rows of a torsion shard that are the driven dihedral itself (TorsionDrive scan panel). */
export function drivenScans(shard, topologies) {
  const starts = shardRowStarts(shard, topologies);
  const scans = [];
  for (let i = 0; i < shard.nAsg; i++) {
    if (!(shard.flags[i] & FLAG_IS_DRIVEN_TORSION)) continue;
    const rows = [];
    for (let r = starts[i]; r < starts[i + 1]; r++) rows.push(r);
    scans.push({ asg: i, topology: shard.topology[i], rows });
  }
  return scans;
}

export function histogram(values, weights, lo, hi, nBins) {
  const counts = new Float64Array(nBins);
  const width = (hi - lo) / nBins;
  for (let n = 0; n < values.length; n++) {
    let b = Math.floor((values[n] - lo) / width);
    if (b === nBins && values[n] === hi) b = nBins - 1;
    if (b >= 0 && b < nBins) counts[b] += weights[n];
  }
  return { lo, hi, width, counts: Array.from(counts) };
}

export function weightedLinearStats(values, weights) {
  let sw = 0, s = 0;
  for (let n = 0; n < values.length; n++) {
    sw += weights[n];
    s += weights[n] * values[n];
  }
  if (!sw) return null;
  const mean = s / sw;
  let v = 0;
  for (let n = 0; n < values.length; n++) v += weights[n] * (values[n] - mean) ** 2;
  const idx = values.map((_, n) => n).sort((a, b) => values[a] - values[b]);
  let acc = 0, median = values[idx[0]];
  for (const n of idx) {
    acc += weights[n];
    if (acc >= sw / 2) {
      median = values[n];
      break;
    }
  }
  return { mean, std: Math.sqrt(v / sw), median, min: values[idx[0]], max: values[idx[idx.length - 1]] };
}

export function weightedCircularStats(valuesDeg, weights) {
  let c = 0, s = 0, sw = 0;
  for (let n = 0; n < valuesDeg.length; n++) {
    c += weights[n] * Math.cos(valuesDeg[n] / DEG);
    s += weights[n] * Math.sin(valuesDeg[n] / DEG);
    sw += weights[n];
  }
  if (!sw) return null;
  const r = Math.min(1, Math.hypot(c, s) / sw);
  return { circMean: Math.atan2(s, c) * DEG, circStd: r > 0 ? Math.sqrt(-2 * Math.log(r)) * DEG : null, resultantLength: r };
}

/** Morgan Tanimoto similarity of a query (Uint8Array, 256 bytes) against all rows. */
const POPCOUNT = new Uint8Array(256).map((_, i) => {
  let c = 0;
  for (let v = i; v; v >>= 1) c += v & 1;
  return c;
});
export function tanimotoAll(query, fps, nBytes = 256) {
  const n = fps.length / nBytes;
  const out = new Float32Array(n);
  let qc = 0;
  for (let b = 0; b < nBytes; b++) qc += POPCOUNT[query[b]];
  for (let m = 0; m < n; m++) {
    let inter = 0, cnt = 0;
    const base = m * nBytes;
    for (let b = 0; b < nBytes; b++) {
      const v = fps[base + b];
      inter += POPCOUNT[v & query[b]];
      cnt += POPCOUNT[v];
    }
    const union = qc + cnt - inter;
    out[m] = union ? inter / union : 0;
  }
  return out;
}

/**
 * Classify a mapped SMARTS query (tags 1..n, n = 2..4) as a geometric measurement, given which
 * tag pairs are bonded in the query. Returns "bond" | "angle" | "proper" | "improper" or an error.
 */
export function classifyMeasurement(nTags, bondedPairs) {
  const has = (a, b) => bondedPairs.some(([x, y]) => (x === a && y === b) || (x === b && y === a));
  if (nTags === 2) return has(1, 2) ? { kind: "bond" } : { error: "tags :1 and :2 must be bonded" };
  if (nTags === 3) return has(1, 2) && has(2, 3) ? { kind: "angle" } : { error: "angle needs :1-:2-:3 bonded in a chain" };
  if (nTags === 4) {
    if (has(1, 2) && has(2, 3) && has(3, 4)) return { kind: "proper" };
    if (has(1, 2) && has(2, 3) && has(2, 4)) return { kind: "improper" };
    return { error: "four tags must form a chain :1-:2-:3-:4 (proper) or have :2 bonded to :1, :3, :4 (improper)" };
  }
  return { error: "map 2 to 4 atoms, numbered :1..:n" };
}

/**
 * Measure a matched tagged tuple. Reverse duplicates are one observation (the caller dedups on
 * the canonical key returned here). Impropers: central :2, outer set defines identity; `value` is
 * an array of the three trefoil dihedrals (see below), each to be weighted 1/3.
 */
export function measure(kind, x, tup) {
  if (kind === "bond") return { key: [Math.min(...tup), Math.max(...tup)].join(","), value: distance(x, tup[0], tup[1]), valid: true };
  if (kind === "angle") {
    const key = (tup[0] < tup[2] ? tup : [...tup].reverse()).join(",");
    return { key, value: angle(x, tup[0], tup[1], tup[2]), valid: true };
  }
  if (kind === "proper") {
    const key = (tup[0] < tup[3] || (tup[0] === tup[3] && tup[1] < tup[2]) ? tup : [...tup].reverse()).join(",");
    const d = dihedral(x, ...tup);
    return { key, value: d.value, valid: d.valid };
  }
  // Improper: identity = central + outer set. Handedness depends on outer order, so use a fixed
  // order (outer atoms sorted a<b<d) and report the three cyclic terms in Interchange's
  // central-first form: (c,a,b,d), (c,b,d,a), (c,d,a,b).
  const c = tup[1];
  const [a, b, d] = [tup[0], tup[2], tup[3]].sort((p, q) => p - q);
  const terms = [[c, a, b, d], [c, b, d, a], [c, d, a, b]].map((q) => dihedral(x, ...q));
  return { key: [c, a, b, d].join(","), value: terms.map((t) => t.value), valid: terms.every((t) => t.valid) };
}

/** Central bonds ("a,b", sorted) of dihedrals that are constrained in topology t (frozen or driven). */
export function constrainedCentralBonds(topologies, t) {
  const out = new Set();
  const add = (d) => d && out.add([d[1], d[2]].sort((a, b) => a - b).join(","));
  add(topologies.driven[t]);
  for (const d of topologies.frozen[t]) add(d);
  return out;
}

/**
 * SMARTS geometry for one topology: de-duplicate matched tuples (reverse / permuted matches of the
 * same interaction), drop proper torsions about a constrained bond (same rule as the parameter
 * pages' FLAG_ON_DRIVEN_BOND / FLAG_ON_FROZEN_BOND), then measure every conformer of `coords`.
 * Returns {keys, values, weights, nInvalid, nConstrained}; impropers give 3 values of weight 1/3.
 */
export function measureTopology(kind, topologies, t, tuples, coords) {
  const constrained = kind === "proper" ? constrainedCentralBonds(topologies, t) : new Set();
  const unique = new Map();
  const x0 = conformerCoordsOf(coords, topologies, t, 0);
  for (const tup of tuples) {
    const key = measure(kind, x0, tup).key;
    if (!unique.has(key)) unique.set(key, tup);
  }
  const out = { keys: [], values: [], weights: [], nInvalid: 0, nConstrained: 0 };
  for (const [key, tup] of unique) {
    if (constrained.has([tup[1], tup[2]].sort((a, b) => a - b).join(","))) {
      out.nConstrained++;
      continue;
    }
    out.keys.push(key);
    for (let j = 0; j < topologies.n_conf[t]; j++) {
      const m = measure(kind, conformerCoordsOf(coords, topologies, t, j), tup);
      if (!m.valid) {
        out.nInvalid++;
        continue;
      }
      const vals = Array.isArray(m.value) ? m.value : [m.value];
      for (const v of vals) (out.values.push(v), out.weights.push(1 / vals.length));
    }
  }
  return out;
}

function conformerCoordsOf(coords, topologies, t, j) {
  const n = topologies.n_atoms[t];
  const start = 3 * (topologies.coord_offset[t] + j * n);
  return coords.subarray(start, start + 3 * n);
}

/** Min and max without spreading (safe for arrays of any length). */
export function extent(values) {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return [lo, hi];
}
