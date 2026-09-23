// Query-dependent aggregation only. Everything input-independent (parameter histograms and statistics,
// torsion profiles, geometry of every bond/angle/torsion/improper, alignments, depictions) is precomputed
// in Python (scripts/, PLAN_precompute.md). This module never computes geometry from coordinates.

const DEG = 180 / Math.PI;

/** Weighted histogram of SMARTS-matched (precomputed) values. */
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

/** Weighted mean, population SD, lower weighted median, min, max (same definitions as scripts/sitelib.py). */
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

/** Weighted circular mean, SD sqrt(-2 ln R) and R, of precomputed dihedral values (degrees). */
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

/** Min and max without spreading (safe for arrays of any length). */
export function extent(values) {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return [lo, hi];
}

/** Tanimoto of a query fingerprint against all dataset fingerprints; dataset popcounts are precomputed. */
const POPCOUNT = new Uint8Array(256).map((_, i) => {
  let c = 0;
  for (let v = i; v; v >>= 1) c += v & 1;
  return c;
});
export function tanimotoAll(query, fps, popcounts, nBytes = 256) {
  const n = popcounts.length;
  const out = new Float32Array(n);
  let qc = 0;
  for (let b = 0; b < nBytes; b++) qc += POPCOUNT[query[b]];
  for (let m = 0; m < n; m++) {
    let inter = 0;
    const base = m * nBytes;
    for (let b = 0; b < nBytes; b++) inter += POPCOUNT[fps[base + b] & query[b]];
    const union = qc + popcounts[m] - inter;
    out[m] = union ? inter / union : 0;
  }
  return out;
}

/**
 * Classify a mapped SMARTS query (tags 1..n, n = 2..4) as a measurement, from which tag pairs are bonded.
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
 * Canonical key of a matched tagged tuple in the precomputed geometry universe (no geometry):
 * bond (i<j); angle (min, j, max); proper min(t, reversed t); improper (central :2, outer atoms sorted).
 */
export function universeKey(kind, tup) {
  if (kind === "bond") return [Math.min(...tup), Math.max(...tup)].join(",");
  if (kind === "angle") return (tup[0] < tup[2] ? tup : [tup[2], tup[1], tup[0]]).join(",");
  if (kind === "proper") {
    const rev = [...tup].reverse();
    for (let i = 0; i < 4; i++) if (tup[i] !== rev[i]) return (tup[i] < rev[i] ? tup : rev).join(",");
    return tup.join(",");
  }
  return [tup[1], ...[tup[0], tup[2], tup[3]].sort((a, b) => a - b)].join(",");
}
