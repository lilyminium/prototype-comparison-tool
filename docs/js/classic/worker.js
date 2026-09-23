// RDKit.js Web Worker: all cheminformatics runs here so the page stays responsive.
// Messages: {id, type, ...} -> {id, ok, result} | {id, ok: false, error}
/* global importScripts, initRDKitModule, WorkerLib */
importScripts("../../vendor/rdkit/RDKit_minimal.js", "worker-lib.js");

let RDKit = null;
let topologies = null; // {mdl_smiles, mdl_order}
let molecules = null; // {display_smiles}
const mdlMols = new Map(); // topology -> RDKit mol (MDL form), created lazily
let displayMols = null; // mol_idx -> RDKit mol (RDKit default aromaticity), created lazily
let displayCanonical = null;

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

function ensureDisplayMols() {
  if (displayMols) return;
  displayMols = molecules.display_smiles.map((s) => RDKit.get_mol(s));
  displayCanonical = displayMols.map((m) => (m ? m.get_smiles() : null));
}

function parseQuery(smarts) {
  const qmol = RDKit.get_qmol(smarts);
  if (!qmol || !qmol.is_valid()) throw new Error("RDKit could not parse this SMARTS");
  const info = WorkerLib.queryMapInfo(qmol.get_v3Kmolblock());
  if (info.nTags && !info.contiguous) throw new Error("atom map numbers must be unique and run :1, :2, ... without gaps");
  return { qmol, info };
}

/**
 * Depiction of a topology with highlighted topology atoms (drawn with RDKit default perception).
 * `highlight`: one interaction (bonds between consecutive atoms, or the improper star).
 * `matches`: several tuples (SMARTS results): all their atoms and every bond between two of them.
 */
function topologySvg(t, highlight, improper, width, height, matches) {
  const order = topologies.mdl_order[t];
  const inv = new Map(order.map((top, s) => [top, s]));
  const mol = RDKit.get_mol(topologies.mdl_smiles[t], JSON.stringify({ removeHs: false }));
  const json = JSON.parse(mol.get_json()).molecules[0];
  const z = json.atoms.map((a) => (a.z === undefined ? 6 : a.z));
  let atoms, bondPairs = [];
  if (matches) {
    atoms = [...new Set(matches.flat())].map((a) => inv.get(a));
    const inSet = new Set(atoms);
    for (const b of json.bonds || []) if (inSet.has(b.atoms[0]) && inSet.has(b.atoms[1])) bondPairs.push(b.atoms);
  } else {
    atoms = highlight.map((a) => inv.get(a));
    for (let i = 0; i + 1 < atoms.length; i++) bondPairs.push([atoms[i], atoms[i + 1]]);
    if (atoms.length === 4 && improper) bondPairs = [[atoms[1], atoms[0]], [atoms[1], atoms[2]], [atoms[1], atoms[3]]];
  }
  const keepH = atoms.some((a) => z[a] === 1);
  if (!keepH) {
    // RemoveHs keeps the relative order of heavy atoms
    const heavyIndex = [];
    let h = 0;
    z.forEach((zz, i) => (heavyIndex[i] = zz === 1 ? -1 : h++));
    atoms = atoms.map((a) => heavyIndex[a]);
    bondPairs = bondPairs.map(([a, b]) => [heavyIndex[a], heavyIndex[b]]);
    mol.remove_hs_in_place();
  }
  const bonds = (JSON.parse(mol.get_json()).molecules[0].bonds || []).map((b) => b.atoms);
  const bondIdx = bondPairs
    .map(([a, b]) => bonds.findIndex(([p, q]) => (p === a && q === b) || (p === b && q === a)))
    .filter((i) => i >= 0);
  const svg = mol.get_svg_with_highlights(
    JSON.stringify({ atoms: atoms.filter((a) => a >= 0), bonds: bondIdx, width, height, clearBackground: false }),
  );
  mol.delete();
  return svg;
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

  async molecule({ smiles }) {
    await ready;
    const mol = RDKit.get_mol(smiles);
    if (!mol || !mol.is_valid()) throw new Error("RDKit could not parse this molecule");
    ensureDisplayMols();
    const canonical = mol.get_smiles();
    const fp = mol.get_morgan_fp_as_uint8array(WorkerLib.MORGAN_OPTIONS);
    const exact = [];
    const substructure = [];
    // Query and targets both use RDKit's default aromaticity (consistent with fingerprints)
    displayMols.forEach((m, i) => {
      if (!m) return;
      if (displayCanonical[i] === canonical) exact.push(i);
      if (m.get_substruct_match(mol) !== "{}") substructure.push(i);
    });
    const svg = mol.get_svg(360, 240);
    mol.delete();
    return { canonical, fp, exact, substructure, svg };
  },

  async topologySvg({ t, highlight = [], improper, matches = null, width = 260, height = 190 }) {
    await ready;
    return topologySvg(t, highlight, improper, width, height, matches);
  },

  // Kekulized V2000 molblock with explicit H, in mdl_smiles atom order (the page reorders it to
  // topology order and inserts QM coordinates; see viewer3d.js)
  async molblock({ t }) {
    await ready;
    const mol = RDKit.get_mol(topologies.mdl_smiles[t], JSON.stringify({ removeHs: false }));
    const molblock = mol.get_molblock(JSON.stringify({ kekulize: true }));
    mol.delete();
    return { molblock, order: topologies.mdl_order[t] };
  },

  async moleculeSvg({ mol_idx, width = 260, height = 190 }) {
    await ready;
    const m = RDKit.get_mol(molecules.display_smiles[mol_idx]);
    const svg = m.get_svg(width, height);
    m.delete();
    return svg;
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
