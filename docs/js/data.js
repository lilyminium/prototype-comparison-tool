// Data access for the static site. No DOM; also imported by the node tests (tests/js/).
// Binary layouts are documented in docs/data/SCHEMA.md and written by scripts/09_build_site_data.py.
// This module only decodes and looks up precomputed data; it performs no chemistry or geometry.

const align4 = (n) => n + ((4 - (n % 4)) % 4);

/** Decode assignments.bin: per-topology list of (param index, atoms). */
export function parseAssignments(buffer) {
  const [schema, nTop, nRows] = new Uint32Array(buffer, 0, 3);
  if (schema !== 1) throw new Error(`unsupported assignments schema ${schema}`);
  let off = 12;
  const rowStart = new Int32Array(buffer, off, nTop + 1);
  off += 4 * (nTop + 1);
  const param = new Uint16Array(buffer, off, nRows);
  off = align4(off + 2 * nRows);
  const atoms = new Int16Array(buffer, off, 4 * nRows);
  return {
    nTop,
    nRows,
    rowStart,
    param,
    atoms,
    forTopology(t) {
      const out = [];
      for (let r = rowStart[t]; r < rowStart[t + 1]; r++) {
        const a = [];
        for (let k = 0; k < 4; k++) if (atoms[4 * r + k] >= 0) a.push(atoms[4 * r + k]);
        out.push({ paramIndex: param[r], atoms: a });
      }
      return out;
    },
  };
}

export const GEOMETRY_KINDS = [
  ["bond", 2, 1],
  ["angle", 3, 1],
  ["proper", 4, 1],
  ["improper", 4, 3],
];
export const GEOM_FLAG_VALID = 1;
export const GEOM_FLAG_FROZEN = 2;

/**
 * Decode geom_opt.bin (schema 2): every bond, angle, proper chain and improper star of every optimization
 * record with its precomputed value(s), validity, frozen-bond flag and Sage parameter (-1 = none).
 * `lookup(kind, t)` returns a Map from the canonical key (stats.universeKey) to a row accessor.
 */
export function parseGeometry(buffer) {
  const header = new Uint32Array(buffer, 0, 6);
  if (header[0] !== 2) throw new Error(`unsupported geometry schema ${header[0]}`);
  const nTop = header[1];
  let off = 24;
  const kinds = {};
  GEOMETRY_KINDS.forEach(([kind, k, width], i) => {
    const n = header[2 + i];
    const rowStart = new Int32Array(buffer, off, nTop + 1);
    off += 4 * (nTop + 1);
    const atoms = new Int16Array(buffer, off, n * k);
    off += 2 * n * k;
    const param = new Int16Array(buffer, off, n);
    off += 2 * n;
    const flags = new Uint8Array(buffer, off, n);
    off = align4(off + n);
    const values = new Float32Array(buffer, off, n * width);
    off += 4 * n * width;
    kinds[kind] = { n, k, width, rowStart, atoms, param, flags, values };
  });
  const cache = new Map();
  return {
    nTop,
    kinds,
    lookup(kind, t) {
      const ck = `${kind}:${t}`;
      if (!cache.has(ck)) {
        const K = kinds[kind];
        const m = new Map();
        for (let r = K.rowStart[t]; r < K.rowStart[t + 1]; r++) {
          m.set(Array.from(K.atoms.subarray(r * K.k, r * K.k + K.k)).join(","), {
            values: Array.from(K.values.subarray(r * K.width, r * K.width + K.width)),
            valid: (K.flags[r] & GEOM_FLAG_VALID) !== 0,
            frozen: (K.flags[r] & GEOM_FLAG_FROZEN) !== 0,
            param: K.param[r],
          });
        }
        cache.set(ck, m);
      }
      return cache.get(ck);
    },
  };
}

/** Coordinates (Å) of conformer `j` of topology `t` as a Float32Array view (slicing only). */
export function conformerCoords(coords, topologies, t, j) {
  const n = topologies.n_atoms[t];
  const start = 3 * (topologies.coord_offset[t] + j * n);
  return coords.subarray(start, start + 3 * n);
}

/** Fetch helpers with a small cache (browser only). */
const cache = new Map();
export function fetchJSON(url) {
  if (!cache.has(url)) cache.set(url, fetch(url).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${url}: ${r.status}`)))));
  return cache.get(url);
}
export function fetchBuffer(url) {
  if (!cache.has(url)) cache.set(url, fetch(url).then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`${url}: ${r.status}`)))));
  return cache.get(url);
}

export const SHARD_SIZE = 500;
/** Precomputed 2D depiction of topology t (molblocks, heavy-atom index map, bond lists). */
export async function depiction(base, t) {
  const shard = await fetchJSON(`${base}depictions/${Math.floor(t / SHARD_SIZE)}.json`);
  const i = t - shard.first;
  return { molblock_h: shard.molblock_h[i], molblock_heavy: shard.molblock_heavy[i], heavy_index: shard.heavy_index[i], bonds_h: shard.bonds_h[i], bonds_heavy: shard.bonds_heavy[i] };
}
/** Pre-rendered SVG of molecule m. */
export async function moleculeSvg(base, m) {
  const shard = await fetchJSON(`${base}molecule_svgs/${Math.floor(m / SHARD_SIZE)}.json`);
  return shard.svg[m - shard.first];
}
