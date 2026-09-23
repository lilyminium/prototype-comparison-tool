"""Reference SVGs from Python RDKit for the rendering oracle (tests/js/render_oracle.test.mjs).

For sampled records (every 25th, plus all records whose heavy-only drawing keeps a hydrogen), both drawing
variants, with and without highlights, render the PRECOMPUTED depiction molblock with Python RDKit
2026.03.6 using the same options as the site. The browser path (RDKit.js renderDepiction) must produce
identical SVG.
"""

import argparse
import json
import pathlib

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

WIDTH, HEIGHT = 260, 190


def render(molblock: str, atoms: list[int], bonds: list[int]) -> str:
    mol = Chem.MolFromMolBlock(molblock, removeHs=False)
    d = rdMolDraw2D.MolDraw2DSVG(WIDTH, HEIGHT)
    d.drawOptions().clearBackground = False
    d.DrawMolecule(mol, highlightAtoms=atoms, highlightBonds=bonds)
    d.FinishDrawing()
    return d.GetDrawingText()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=pathlib.Path, default=pathlib.Path("data/processed"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("tests/js/render_fixtures.json"))
    args = parser.parse_args()
    dep = pd.read_parquet(args.processed / "depictions.parquet").sort_values("topology_idx")
    kept_h = [
        int(t)
        for t, hi, mb in zip(dep.topology_idx, dep.heavy_index, dep.molblock_heavy)
        if any(Chem.MolFromMolBlock(mb, removeHs=False).GetAtomWithIdx(int(x)).GetAtomicNum() == 1 for x in hi if x >= 0)
    ]
    sample = sorted(set(dep.topology_idx[::25].astype(int)) | set(kept_h))
    by_t = dep.set_index("topology_idx")
    cases = []
    for t in sample:
        r = by_t.loc[t]
        for variant in ("h", "heavy"):
            mb = r.molblock_h if variant == "h" else r.molblock_heavy
            mol = Chem.MolFromMolBlock(mb, removeHs=False)
            n_bonds = mol.GetNumBonds()
            for atoms, bonds in (([], []), ([0, 1] if mol.GetNumAtoms() > 1 else [0], [0] if n_bonds else [])):
                cases.append({"t": int(t), "variant": variant, "atoms": atoms, "bonds": bonds, "svg": render(mb, atoms, bonds)})
    args.out.write_text(json.dumps({"kept_h": kept_h, "cases": cases}))
    print(f"{len(cases)} reference renders for {len(sample)} records ({len(kept_h)} keep an H in the heavy-only drawing)")


if __name__ == "__main__":
    main()
