// Data access for the static site. No DOM; also imported by the node tests (tests/js/).
// Binary layouts are documented in docs/data/SCHEMA.md and written by scripts/09_build_site_data.py.

export const FLAG_ON_DRIVEN_BOND = 1;
export const FLAG_ON_FROZEN_BOND = 2;
export const FLAG_IS_DRIVEN_TORSION = 4;
export const TORSION_HANDLERS = new Set(["ProperTorsions", "ImproperTorsions"]);

const align4 = (n) => n + ((4 - (n % 4)) % 4);

/** Parse a per-parameter shard (scripts/09_build_site_data.py, "per-parameter shards"). */
export function parseShard(buffer, handler) {
  const header = new Uint32Array(buffer, 0, 4);
  const [schema, nAsg, nRows, width] = header;
  if (schema !== 1) throw new Error(`unsupported shard schema ${schema}`);
  let off = 16;
  const topology = new Int32Array(buffer, off, nAsg);
  off += 4 * nAsg;
  const atoms = new Int16Array(buffer, off, 4 * nAsg);
  off += 8 * nAsg;
  const flags = new Uint8Array(buffer, off, nAsg);
  off = align4(off + nAsg);
  const values = new Float32Array(buffer, off, nRows * width);
  off += 4 * nRows * width;
  const valid = TORSION_HANDLERS.has(handler) ? new Uint8Array(buffer, off, nRows) : null;
  return { nAsg, nRows, width, topology, atoms, flags, values, valid };
}

/** Parse assignments.bin: per-topology list of (param index, atoms). */
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

/** Coordinates (Å) of conformer `j` (0-based within the topology) as a Float32Array view. */
export function conformerCoords(coords, topologies, t, j) {
  const n = topologies.n_atoms[t];
  const start = 3 * (topologies.coord_offset[t] + j * n);
  return coords.subarray(start, start + 3 * n);
}

/** Row offset of each assignment in a shard (rows = assignment x topology conformers). */
export function shardRowStarts(shard, topologies) {
  const starts = new Int32Array(shard.nAsg + 1);
  for (let i = 0; i < shard.nAsg; i++) starts[i + 1] = starts[i] + topologies.n_conf[shard.topology[i]];
  if (starts[shard.nAsg] !== shard.nRows) throw new Error("shard rows inconsistent with topologies.n_conf");
  return starts;
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
