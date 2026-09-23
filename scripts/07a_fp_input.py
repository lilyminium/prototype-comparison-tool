"""Write the fingerprint input for 07_fingerprints.mjs and, afterwards, cross-check its output.

python scripts/07a_fp_input.py            -> data/processed/fp_input.json
node   scripts/07_fingerprints.mjs        -> data/processed/fp_morgan.bin
python scripts/07a_fp_input.py --check    -> compares every RDKit.js fingerprint bit-for-bit with
                                             Python RDKit (same release) on the same SMILES
"""

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed", type=pathlib.Path, default=pathlib.Path("data/processed")
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    molecules = pd.read_parquet(args.processed / "molecules.parquet")

    if not args.check:
        records = [
            {"mol_idx": int(i), "smiles": s}
            for i, s in zip(molecules.mol_idx, molecules.display_smiles)
        ]
        (args.processed / "fp_input.json").write_text(json.dumps(records))
        print(f"wrote {len(records)} molecules")
        return

    js = np.fromfile(args.processed / "fp_morgan.bin", dtype=np.uint8).reshape(
        len(molecules), 256
    )
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mismatched = []
    for mol_idx, smiles in zip(molecules.mol_idx, molecules.display_smiles):
        bits = np.zeros(2048, dtype=np.uint8)
        bits[list(generator.GetFingerprint(Chem.MolFromSmiles(smiles)).GetOnBits())] = 1
        # RDKit.js packs bit i into byte i // 8, least-significant bit first
        packed = np.packbits(bits, bitorder="little")
        if not np.array_equal(packed, js[mol_idx]):
            mismatched.append(int(mol_idx))
    print(
        json.dumps(
            {
                "n": len(molecules),
                "n_mismatched": len(mismatched),
                "examples": mismatched[:10],
            }
        )
    )
    if mismatched:
        raise SystemExit("RDKit.js and Python fingerprints differ")


if __name__ == "__main__":
    main()
