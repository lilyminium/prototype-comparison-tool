// Browser-side half of the SMARTS parity test (see 08a_smarts_parity.py).
// Loads each training topology exactly as the website does (OpenFF MDL-aromatic explicit-H SMILES,
// no re-aromatization) and matches every force-field SMIRKS with OpenFF's match settings.
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { loadRDKit, MDL_MOL_OPTIONS } from "./js/rdkit.mjs";

// Same settings as openff.toolkit RDKitToolkitWrapper._find_smarts_matches
export const MATCH_OPTIONS = JSON.stringify({ uniquify: false, maxMatches: 4294967295, useChirality: true });

const { RDKit, version } = await loadRDKit();
const input = JSON.parse(readFileSync("data/processed/parity_input.json", "utf8"));

// Query-atom index of each tag :1..:n comes from Python (same RDKit release, OpenFF's own logic);
// RDKit.js's query JSON does not expose atom map numbers. End-to-end parity validates the indexing.
const queries = input.smirks.map(({ smirks, tag_index }) => {
  const qmol = RDKit.get_qmol(smirks);
  if (!qmol) throw new Error(`RDKit.js could not parse ${smirks}`);
  return { qmol, tagIndex: tag_index };
});

const lines = [];
const md5 = (s) => createHash("md5").update(s).digest("hex");
for (const { topology_idx, mdl_smiles, mdl_order } of input.molecules) {
  const mol = RDKit.get_mol(mdl_smiles, MDL_MOL_OPTIONS);
  if (!mol) throw new Error(`RDKit.js could not load ${mdl_smiles}`);
  queries.forEach((q, s_idx) => {
    const res = JSON.parse(mol.get_substruct_matches(q.qmol, MATCH_OPTIONS));
    if (!Array.isArray(res) || res.length === 0) return;
    const set = new Map();
    for (const { atoms } of res) {
      const tuple = q.tagIndex.map((qi) => mdl_order[atoms[qi]]);
      set.set(tuple.join(","), tuple);
    }
    const sorted = [...set.values()].sort((a, b) => {
      for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] - b[i];
      return 0;
    });
    lines.push(`${topology_idx}\t${s_idx}\t${sorted.length}\t${md5(JSON.stringify(sorted))}`);
  });
  mol.delete();
}
writeFileSync("data/processed/parity_js.tsv", lines.join("\n") + "\n");
console.log(`RDKit.js ${version}: ${lines.length} (topology, SMIRKS) pairs with matches`);
