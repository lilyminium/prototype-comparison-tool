// RDKit.js Web Worker. Only query-dependent work happens here: SMARTS matching, parsing and fingerprinting
// the user's molecule, substructure search, and drawing (precomputed dataset depictions are rendered, never
// laid out). Messages: {id, type, ...} -> {id, ok, result} | {id, ok: false, error}
/* global importScripts, initRDKitModule, WorkerLib */
importScripts("../../vendor/rdkit/RDKit_minimal.js", "worker-lib.js");

let RDKit = null;
let topologies = null; // {mdl_smiles, mdl_order}
let molecules = null; // {display_smiles}
const mdlMols = new Map(); // topology -> RDKit mol (OpenFF MDL form), created lazily for matching
let displayMols = null; // mol_idx -> RDKit mol, created lazily for substructure search

const ready = initRDKitModule({ locateFile: (f) => `../../vendor/rdkit/${f}` }).then((m) => {
  RDKit = m;
  return m.version();
});

function mdlMol(t) {
  let m = mdlMols.get(t);
  if (!m) {
    m = RDKit.get_mol(topologies.mdl_smiles[t], WorkerLib.MDL_MOL_OPTIONS);
    mdlMols.set(t, m);
  }
  return m;
}

function parseQuery(smarts) {
  const qmol = RDKit.get_qmol(smarts);
  if (!qmol || !qmol.is_valid()) throw new Error("RDKit could not parse this SMARTS");
  const info = WorkerLib.queryMapInfo(qmol.get_v3Kmolblock());
  if (info.nTags && !info.contiguous) throw new Error("atom map numbers must be unique and run :1, :2, ... without gaps");
  return { qmol, info };
}

const handlers = {
  async init({ topologies: t, molecules: m }) {
    const version = await ready;
    topologies = t;
    molecules = m;
    return { version };
  },

  // Long-running; the page cancels by terminating this worker and starting a new one.
  async smarts({ smarts }) {
    await ready;
    const { qmol, info } = parseQuery(smarts);
    const hitTopologies = [];
    const offsets = [0];
    const flat = [];
    const n = topologies.mdl_smiles.length;
    for (let t = 0; t < n; t++) {
      const tuples = WorkerLib.matchTuples(mdlMol(t), qmol, info.tagIndex, topologies.mdl_order[t]);
      if (!tuples.length) continue;
      hitTopologies.push(t);
      for (const tup of tuples) flat.push(...tup);
      offsets.push(offsets[offsets.length - 1] + tuples.length);
      if (t % 500 === 0) self.postMessage({ progress: { done: t, total: n } });
    }
    qmol.delete();
    const width = info.nTags || info.nAtoms;
    return { info, width, hitTopologies: Int32Array.from(hitTopologies), offsets: Int32Array.from(offsets), tuples: Int32Array.from(flat) };
  },

  // The user's molecule: parse, canonicalize, fingerprint, substructure search. Dataset canonical SMILES and
  // fingerprints are precomputed; exact matching and Tanimoto happen on the page.
  async molecule({ smiles }) {
    await ready;
    const mol = RDKit.get_mol(smiles);
    if (!mol || !mol.is_valid()) throw new Error("RDKit could not parse this molecule");
    if (!displayMols) displayMols = molecules.display_smiles.map((s) => RDKit.get_mol(s));
    const canonical = mol.get_smiles();
    const fp = mol.get_morgan_fp_as_uint8array(WorkerLib.MORGAN_OPTIONS);
    const substructure = [];
    displayMols.forEach((m, i) => m && m.get_substruct_match(mol) !== "{}" && substructure.push(i));
    const svg = mol.get_svg(360, 240); // the query itself: runtime layout is unavoidable
    mol.delete();
    return { canonical, fp, substructure, svg };
  },

  // Precomputed dataset depiction + highlight indices (computed from precomputed maps on the page)
  async render({ molblock, atoms, bonds, width = 260, height = 190 }) {
    await ready;
    return WorkerLib.renderDepiction(RDKit, molblock, atoms, bonds, width, height);
  },
};

self.onmessage = async (event) => {
  const { id, type, ...args } = event.data;
  try {
    const result = await handlers[type](args);
    const transfer = result && result.tuples ? [result.hitTopologies.buffer, result.offsets.buffer, result.tuples.buffer] : [];
    self.postMessage({ id, ok: true, result }, transfer);
  } catch (e) {
    self.postMessage({ id, ok: false, error: e.message || String(e) });
  }
};
