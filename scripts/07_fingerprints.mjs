// Morgan fingerprints (radius 2, 2048 bits) for every unique molecule, computed with the vendored
// RDKit.js so that the site's query fingerprints are produced by the identical code.
// Molecules are parsed from their RDKit-default (display) SMILES with default sanitization, i.e.
// exactly how a user query is parsed in the browser.
//
// Input:  data/processed/fp_input.json   [{mol_idx, smiles}] written by 07a_fp_input.py
// Output: data/processed/fp_morgan.bin   N x 256 bytes, row = mol_idx
//         data/processed/fp_morgan.json  provenance (RDKit.js version, WASM hash, options)
import { readFileSync, writeFileSync } from "node:fs";
import { loadRDKit, MORGAN_OPTIONS } from "./js/rdkit.mjs";

const { RDKit, hashes, version } = await loadRDKit();
const molecules = JSON.parse(readFileSync("data/processed/fp_input.json", "utf8"));
const nBytes = 2048 / 8;
const out = new Uint8Array(molecules.length * nBytes);
const failures = [];
for (const { mol_idx, smiles } of molecules) {
  const mol = RDKit.get_mol(smiles);
  if (!mol) {
    failures.push({ mol_idx, smiles });
    continue;
  }
  out.set(mol.get_morgan_fp_as_uint8array(MORGAN_OPTIONS), mol_idx * nBytes);
  mol.delete();
}
if (failures.length) throw new Error(`RDKit.js failed to parse ${failures.length} molecules: ${JSON.stringify(failures.slice(0, 5))}`);
writeFileSync("data/processed/fp_morgan.bin", out);
writeFileSync(
  "data/processed/fp_morgan.json",
  JSON.stringify({ rdkit_js_version: version, vendored_sha256: hashes, options: JSON.parse(MORGAN_OPTIONS), n: molecules.length }, null, 1) + "\n",
);
console.log(`wrote ${molecules.length} fingerprints with RDKit.js ${version}`);
