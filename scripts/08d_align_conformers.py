"""Superpose the optimization conformers of each molecule (precomputed for the 3D viewer; no alignment in JS).

For every molecule with more than one optimization record and every ORDERED pair (reference r, record u),
u's coordinates are rigidly superposed onto r (Kabsch: SVD with determinant correction, so a proper
rotation, never a reflection) using heavy atoms. Records of one molecule share a canonical MDL SMILES, so
SMILES atom n of r corresponds to SMILES atom n of u (topology atoms via mdl_order). If fewer than three
non-collinear heavy atoms exist, all atoms (including H) are used.

The viewer shows the clicked record r in its own frame and every other record u superposed on it, with the
RMSD computed from exactly the shipped coordinates.

Output: data/processed/conformer_alignment.parquet
  mol_idx, ref_topology, topology, coords (float32, u aligned onto r, topology order of u), rmsd, n_fit
"""

import argparse
import itertools
import json
import pathlib

import numpy as np
import pandas as pd


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal proper rotation R and translation t with R @ mobile_i + t ≈ target_i (least squares)."""
    cm, ct = mobile.mean(axis=0), target.mean(axis=0)
    h = (mobile - cm).T @ (target - ct)
    u, _, vt = np.linalg.svd(h)
    d = (
        np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    )  # reflection → flip the smallest singular direction
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, ct - rot @ cm


def fit_indices(heavy: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Heavy atoms if they span >= 2 dimensions (>= 3 non-collinear points), else all atoms."""
    idx = np.flatnonzero(heavy)
    if len(idx) >= 3:
        s = np.linalg.svd(xyz[idx] - xyz[idx].mean(axis=0), compute_uv=False)
        if s[1] > 1e-3:
            return idx
    return np.arange(len(heavy))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed", type=pathlib.Path, default=pathlib.Path("data/processed")
    )
    args = parser.parse_args()
    p = args.processed

    from openff.toolkit import Molecule

    tops = pd.read_parquet(p / "topologies.parquet")
    confs = pd.read_parquet(
        p / "conformers.parquet", columns=["topology_idx", "coords_angstrom"]
    )
    coords = {
        int(t): np.asarray(c, dtype=np.float32).reshape(-1, 3).astype(np.float64)
        for t, c in zip(confs.topology_idx, confs.coords_angstrom)
    }
    opt = tops[tops.source == "opt"]
    rows = []
    for mol_idx, group in opt.groupby("mol_idx"):
        if len(group) < 2:
            continue
        if group.mdl_smiles.nunique() != 1:
            raise RuntimeError(
                f"molecule {mol_idx}: records do not share a canonical MDL SMILES"
            )
        order = {
            int(t): np.asarray(json.loads(o))
            for t, o in zip(group.topology_idx, group.mdl_order)
        }
        elements = {
            int(t): np.array(
                [
                    a.atomic_number
                    for a in Molecule.from_mapped_smiles(
                        s, allow_undefined_stereo=True
                    ).atoms
                ]
            )
            for t, s in zip(group.topology_idx, group.mapped_smiles)
        }
        for r, u in itertools.permutations(group.topology_idx.astype(int), 2):
            # SMILES atom n ↔ topology atom order[n]; element identity checked for every pair
            if not np.array_equal(elements[r][order[r]], elements[u][order[u]]):
                raise RuntimeError(
                    f"molecule {mol_idx}: element mismatch between records {r} and {u}"
                )
            heavy = elements[r][order[r]] != 1
            xr, xu = coords[r][order[r]], coords[u][order[u]]
            fit = fit_indices(heavy, xr)
            rot, trans = kabsch(xu[fit], xr[fit])
            aligned_all = coords[u] @ rot.T + trans  # topology order of u
            aligned_fit = (
                aligned_all[order[u]][fit].astype(np.float32).astype(np.float64)
            )
            rmsd = float(np.sqrt(((aligned_fit - xr[fit]) ** 2).sum(axis=1).mean()))
            rows.append(
                {
                    "mol_idx": int(mol_idx),
                    "ref_topology": r,
                    "topology": u,
                    "coords": aligned_all.astype(np.float32).ravel().tolist(),
                    "rmsd": rmsd,
                    "n_fit": int(len(fit)),
                    "fit_heavy_only": bool(len(fit) < len(heavy)),
                }
            )
    df = pd.DataFrame(rows)
    df.to_parquet(p / "conformer_alignment.parquet", index=False)
    print(
        f"{df.mol_idx.nunique()} molecules, {len(df)} ordered pairs; RMSD min/median/max {df.rmsd.min():.3f}/{df.rmsd.median():.3f}/{df.rmsd.max():.3f} Å"
    )


if __name__ == "__main__":
    main()
