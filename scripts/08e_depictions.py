"""Precompute 2D depictions (PLAN_precompute.md R4/S2/U1). RDKit.js only renders these; it never lays out.

Per record (topology), in the record's own atom order:
  molblock_h      explicit-H molblock, RDKit native 2D coordinates (deterministic; CoordGen is not), kekulized bonds, wedges from stereo
  molblock_heavy  the same drawing with hydrogens removed (heavy-atom coordinates unchanged)
  heavy_index     topology atom -> atom index in molblock_heavy (-1 for hydrogens)
The explicit-H molblock also serves the 3D viewer: only its coordinate columns are replaced by QM
coordinates (atom order = topology order), so no atom reordering or bond perception happens in the browser.

Per unique molecule: a pre-rendered unhighlighted SVG (similarity / substructure cards), 260 x 190.

Rendering contract, verified by tests/js/render_oracle.mjs: for a fixed molblock, highlight set and size,
the vendored RDKit.js SVG is identical to Python RDKit's (same 2026.03.6 release).

Outputs: data/processed/depictions.parquet, data/processed/molecule_svgs.parquet
"""

import argparse
import multiprocessing
import pathlib

import pandas as pd

WIDTH, HEIGHT = 260, 190


def rdkit_only():
    from openff.toolkit.utils.toolkit_registry import (
        ToolkitRegistry,
        toolkit_registry_manager,
    )
    from openff.toolkit.utils.toolkits import BuiltInToolkitWrapper, RDKitToolkitWrapper

    return toolkit_registry_manager(
        ToolkitRegistry([RDKitToolkitWrapper, BuiltInToolkitWrapper])
    )


def depict_topology(task: tuple[int, str]) -> dict:
    from openff.toolkit import Molecule
    from rdkit import Chem
    from rdkit.Chem import rdDepictor

    topology_idx, mapped_smiles = task
    with rdkit_only():
        rdmol = Molecule.from_mapped_smiles(
            mapped_smiles, allow_undefined_stereo=True
        ).to_rdkit()
    rdmol = Chem.Mol(rdmol)
    rdmol.RemoveAllConformers()
    rdDepictor.SetPreferCoordGen(False)  # CoordGen layouts are not deterministic for strained cages (STATUS.md)
    rdDepictor.Compute2DCoords(rdmol)
    Chem.WedgeMolBonds(rdmol, rdmol.GetConformer())
    molblock_h = Chem.MolToMolBlock(rdmol)  # atom order = topology order
    # Tag atoms with their topology index so the heavy-only drawing maps back exactly, whichever H
    # RemoveHs keeps (e.g. hydrogens needed to define stereochemistry)
    for atom in rdmol.GetAtoms():
        atom.SetIntProp("topology_atom", atom.GetIdx())
    heavy = Chem.RemoveHs(rdmol, updateExplicitCount=True)
    heavy_index = [-1] * rdmol.GetNumAtoms()
    for atom in heavy.GetAtoms():
        heavy_index[atom.GetIntProp("topology_atom")] = atom.GetIdx()
    for atom in rdmol.GetAtoms():
        if heavy_index[atom.GetIdx()] < 0 and atom.GetAtomicNum() != 1:
            raise RuntimeError(f"topology {topology_idx}: heavy atom {atom.GetIdx()} missing from heavy-only drawing")
    return {
        "topology_idx": topology_idx,
        "molblock_h": molblock_h,
        "molblock_heavy": Chem.MolToMolBlock(heavy),
        "heavy_index": heavy_index,
    }


def render_molecule(task: tuple[int, str]) -> dict:
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    mol_idx, smiles = task
    m = Chem.MolFromSmiles(smiles)
    rdDepictor.SetPreferCoordGen(False)  # CoordGen layouts are not deterministic for strained cages (STATUS.md)
    rdDepictor.Compute2DCoords(m)
    d = rdMolDraw2D.MolDraw2DSVG(WIDTH, HEIGHT)
    d.drawOptions().clearBackground = False
    d.DrawMolecule(m)
    d.FinishDrawing()
    return {"mol_idx": mol_idx, "svg": d.GetDrawingText()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed", type=pathlib.Path, default=pathlib.Path("data/processed")
    )
    parser.add_argument(
        "--nproc", type=int, default=max(1, multiprocessing.cpu_count() - 2)
    )
    args = parser.parse_args()
    p = args.processed
    tops = pd.read_parquet(
        p / "topologies.parquet", columns=["topology_idx", "mapped_smiles"]
    )
    mols = pd.read_parquet(
        p / "molecules.parquet", columns=["mol_idx", "display_smiles"]
    )
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(args.nproc) as pool:
        dep = pool.map(
            depict_topology,
            list(zip(tops.topology_idx.astype(int), tops.mapped_smiles)),
            chunksize=32,
        )
        svgs = pool.map(
            render_molecule,
            list(zip(mols.mol_idx.astype(int), mols.display_smiles)),
            chunksize=32,
        )
    pd.DataFrame(dep).to_parquet(p / "depictions.parquet", index=False)
    pd.DataFrame(svgs).to_parquet(p / "molecule_svgs.parquet", index=False)
    print(f"{len(dep)} record depictions, {len(svgs)} molecule SVGs")


if __name__ == "__main__":
    main()
