"""Every measurable internal coordinate of every OPTIMIZATION record, independent of parameter assignment.

This is what the site's SMARTS search looks up instead of computing geometry in the browser
(PLAN_precompute.md, R1/R2/T1). Tuples are enumerated from the molecular graph alone:

  bond      (i, j)             i < j
  angle     (i, j, k)          j central, i < k
  proper    (i, j, k, l)       bonded chain, i != l, canonical orientation min(t, reversed t)
  improper  (c, a, b, d)       every atom c with >= 3 neighbours and every 3-subset a < b < d of them

Values are measured with RDKit rdMolTransforms on the coordinates the site ships (float32-rounded), so the
numbers equal what the browser previously computed from coords_opt.bin:

  bond      length (Å)                      angle   degrees
  proper    signed dihedral (IUPAC/OpenMM)  improper three terms (c,a,b,d), (c,b,d,a), (c,d,a,b) — the
            SMARTS convention (sorted outer atoms), NOT the Interchange order used on parameter pages
  valid     dihedrals only: every term's bond angles have sin >= 0.02 and the value is finite

Each tuple carries the index (into params.json order) of the Sage parameter label_molecules assigned to
it, or -1 (no parameter: the 4 declared unassigned propers and every improper star no Sage improper
matches), and a flag for propers about a frozen bond (constrained optimizations).

Output: data/processed/geometry_universe.parquet
"""

import argparse
import itertools
import multiprocessing
import pathlib

import numpy as np
import pandas as pd

KINDS = ("bond", "angle", "proper", "improper")
SIN_THRESHOLD = 0.02
HANDLER_OF = {
    "bond": "Bonds",
    "angle": "Angles",
    "proper": "ProperTorsions",
    "improper": "ImproperTorsions",
}


def rdkit_only():
    from openff.toolkit.utils.toolkit_registry import (
        ToolkitRegistry,
        toolkit_registry_manager,
    )
    from openff.toolkit.utils.toolkits import BuiltInToolkitWrapper, RDKitToolkitWrapper

    return toolkit_registry_manager(
        ToolkitRegistry([RDKitToolkitWrapper, BuiltInToolkitWrapper])
    )


def enumerate_tuples(
    bonds: list[tuple[int, int]], n_atoms: int
) -> dict[str, list[tuple[int, ...]]]:
    """All bonds, angles, proper chains and improper stars of a molecular graph (canonical forms)."""
    nbr: list[set[int]] = [set() for _ in range(n_atoms)]
    for i, j in bonds:
        nbr[i].add(j)
        nbr[j].add(i)
    out = {k: [] for k in KINDS}
    out["bond"] = sorted(tuple(sorted(b)) for b in bonds)
    for j in range(n_atoms):
        for i, k in itertools.combinations(sorted(nbr[j]), 2):
            out["angle"].append((i, j, k))
    propers = set()
    for j, k in out["bond"]:
        for a, b in ((j, k), (k, j)):
            for i in nbr[a] - {b}:
                for l in nbr[b] - {a}:
                    if i != l:
                        t = (i, a, b, l)
                        propers.add(min(t, t[::-1]))
    out["proper"] = sorted(propers)
    for c in range(n_atoms):
        for a, b, d in itertools.combinations(sorted(nbr[c]), 3):
            out["improper"].append((c, a, b, d))
    return out


def canonical_key(kind: str, atoms: tuple[int, ...]) -> tuple[int, ...]:
    """Identity of a label_molecules tuple in the universe's canonical form."""
    if kind == "bond":
        return tuple(sorted(atoms))
    if kind == "angle":
        i, j, k = atoms
        return (min(i, k), j, max(i, k))
    if kind == "proper":
        return min(tuple(atoms), tuple(atoms)[::-1])
    a, c, b, d = atoms  # label_molecules improper key: central atom second
    return (c, *sorted((a, b, d)))


def measure_topology(task: dict) -> list[dict]:
    from openff.toolkit import Molecule
    from rdkit import Chem
    from rdkit.Chem import rdMolTransforms as T
    from rdkit.Geometry import Point3D

    with rdkit_only():
        mol = Molecule.from_mapped_smiles(
            task["mapped_smiles"], allow_undefined_stereo=True
        )
    bonds = [(b.atom1_index, b.atom2_index) for b in mol.bonds]
    tuples = enumerate_tuples(bonds, mol.n_atoms)
    xyz = np.asarray(task["coords"], dtype=np.float32).reshape(-1, 3).astype(np.float64)
    conf = Chem.Conformer(mol.n_atoms)
    for n, (x, y, z) in enumerate(xyz):
        conf.SetAtomPosition(n, Point3D(float(x), float(y), float(z)))

    def dihedral(i, j, k, l):
        value = T.GetDihedralDeg(conf, i, j, k, l)
        s1 = np.sin(np.radians(T.GetAngleDeg(conf, i, j, k)))
        s2 = np.sin(np.radians(T.GetAngleDeg(conf, j, k, l)))
        return value, bool(min(s1, s2) >= SIN_THRESHOLD and np.isfinite(value))

    assigned = task["assigned"]  # {(kind, key): param_idx}
    frozen = task["frozen_bonds"]
    rows = []
    for kind in KINDS:
        for t in tuples[kind]:
            if kind == "bond":
                values, valid = [T.GetBondLength(conf, *t)], True
            elif kind == "angle":
                values, valid = [T.GetAngleDeg(conf, *t)], True
            elif kind == "proper":
                v, valid = dihedral(*t)
                values = [v]
            else:
                c, a, b, d = t
                terms = [
                    dihedral(c, a, b, d),
                    dihedral(c, b, d, a),
                    dihedral(c, d, a, b),
                ]
                values, valid = [v for v, _ in terms], all(ok for _, ok in terms)
            rows.append(
                {
                    "topology_idx": task["topology_idx"],
                    "kind": kind,
                    "atoms": list(t),
                    "values": values,
                    "valid": valid,
                    "param_idx": assigned.get((kind, t), -1),
                    "on_frozen_bond": kind == "proper"
                    and tuple(sorted(t[1:3])) in frozen,
                }
            )
    # every assigned interaction must exist in the universe (else enumeration and labeling disagree)
    universe = {(r["kind"], tuple(r["atoms"])) for r in rows}
    missing = [k for k in assigned if k not in universe]
    if missing:
        raise RuntimeError(
            f"topology {task['topology_idx']}: assigned tuples not in the enumerated universe: {missing[:3]}"
        )
    return rows


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

    topologies = pd.read_parquet(p / "topologies.parquet")
    opt = topologies[topologies.source == "opt"]
    conformers = pd.read_parquet(
        p / "conformers.parquet", columns=["topology_idx", "coords_angstrom"]
    )
    coords = dict(
        zip(conformers.topology_idx, conformers.coords_angstrom)
    )  # opt: one conformer each
    params = pd.read_parquet(p / "params.parquet")
    param_index = {
        pid: i for i, pid in enumerate(params.param_id)
    }  # params.json keeps this order
    assignments = pd.read_parquet(p / "assignments.parquet")
    constraints = pd.read_parquet(p / "constraints.parquet")
    frozen = constraints[constraints.kind == "frozen"]
    frozen_by_top: dict[int, set] = {}
    for t, a in zip(frozen.topology_idx, frozen.atoms):
        frozen_by_top.setdefault(int(t), set()).add(
            tuple(sorted((int(a[1]), int(a[2]))))
        )
    kind_of = {v: k for k, v in HANDLER_OF.items()}
    assigned_by_top: dict[int, dict] = {}
    for t, h, atoms, pid in zip(
        assignments.topology_idx,
        assignments.handler,
        assignments.atoms,
        assignments.param_id,
    ):
        kind = kind_of[h]
        assigned_by_top.setdefault(int(t), {})[
            (kind, canonical_key(kind, tuple(int(x) for x in atoms)))
        ] = param_index[pid]

    tasks = [
        {
            "topology_idx": int(t),
            "mapped_smiles": s,
            "coords": list(coords[t]),
            "assigned": assigned_by_top.get(int(t), {}),
            "frozen_bonds": frozen_by_top.get(int(t), set()),
        }
        for t, s in zip(opt.topology_idx, opt.mapped_smiles)
    ]
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(args.nproc) as pool:
        results = pool.map(measure_topology, tasks, chunksize=16)
    df = pd.DataFrame([r for rows in results for r in rows])
    df.to_parquet(p / "geometry_universe.parquet", index=False)
    summary = df.groupby("kind").agg(
        n=("kind", "size"),
        unassigned=("param_idx", lambda x: int((x < 0).sum())),
        invalid=("valid", lambda x: int((~x).sum())),
        frozen=("on_frozen_bond", "sum"),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
