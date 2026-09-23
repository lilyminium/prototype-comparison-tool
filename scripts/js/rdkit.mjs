// Load the VENDORED RDKit.js (docs/vendor/rdkit), i.e. the exact bytes the website serves,
// after verifying them against docs/vendor/rdkit/SHA256SUMS. Build scripts must use this loader,
// never node_modules, so that fingerprints and validation results apply to the shipped WASM.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
export const VENDOR_DIR = path.resolve(here, "../../docs/vendor/rdkit");

export function verifyVendoredRDKit() {
  const sums = readFileSync(path.join(VENDOR_DIR, "SHA256SUMS"), "utf8").trim().split("\n");
  const hashes = {};
  for (const line of sums) {
    const [expected, name] = line.trim().split(/\s+/);
    const actual = createHash("sha256").update(readFileSync(path.join(VENDOR_DIR, name))).digest("hex");
    if (actual !== expected) throw new Error(`vendored ${name} sha256 ${actual} != ${expected}`);
    hashes[name] = actual;
  }
  return hashes;
}

export async function loadRDKit() {
  const hashes = verifyVendoredRDKit();
  const require = createRequire(import.meta.url);
  const initRDKitModule = require(path.join(VENDOR_DIR, "RDKit_minimal.js"));
  const RDKit = await initRDKitModule({ locateFile: (file) => path.join(VENDOR_DIR, file) });
  return { RDKit, hashes, version: RDKit.version() };
}

// Options used everywhere a training-set molecule is loaded for SMARTS/SMIRKS matching:
// keep OpenFF's MDL aromatic flags and explicit hydrogens exactly as supplied.
export const MDL_MOL_OPTIONS = JSON.stringify({ setAromaticity: false, kekulize: false, removeHs: false });
// Morgan fingerprint options used for the dataset and for user queries.
export const MORGAN_OPTIONS = JSON.stringify({ radius: 2, nBits: 2048 });
