// PRE-MIGRATION shard parsers (per-parameter shards now live in data/processed/shards and are not served).
// Used only by the migration oracles and data tests.
const align4 = (n) => n + ((4 - (n % 4)) % 4);
const TORSION_HANDLERS = new Set(["ProperTorsions", "ImproperTorsions"]);
export const FLAG_ON_DRIVEN_BOND = 1;
export const FLAG_ON_FROZEN_BOND = 2;
export const FLAG_IS_DRIVEN_TORSION = 4;

export function parseShard(buffer, handler) {
  const [schema, nAsg, nRows, width] = new Uint32Array(buffer, 0, 4);
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

export function shardRowStarts(shard, topologies) {
  const starts = new Int32Array(shard.nAsg + 1);
  for (let i = 0; i < shard.nAsg; i++) starts[i + 1] = starts[i] + topologies.n_conf[shard.topology[i]];
  if (starts[shard.nAsg] !== shard.nRows) throw new Error("shard rows inconsistent with topologies.n_conf");
  return starts;
}
