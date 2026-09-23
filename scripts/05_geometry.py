"""Measure every assigned valence interaction in every conformer.

Values (float64), all measured with RDKit rdMolTransforms (GetBondLength / GetAngleDeg / GetDihedralDeg):
  Bonds            length (Å)
  Angles           angle (degrees)
  ProperTorsions   signed dihedral (degrees, IUPAC/OpenMM convention, label_molecules atom order)
  ImproperTorsions the three trefoil dihedrals (degrees) in the (central, a, b, c) order Interchange
                   sends to OpenMM; ``value`` holds the first term, ``improper_terms_deg`` all three

Torsion energies (kcal/mol) use the resolved Fourier terms from params.parquet:
  E = sum_n k_n / idivf_n * (1 + cos(periodicity_n * phi - phase_n)), summed over the three terms for
  impropers (so an improper group's energy is its SMIRNOFF trefoil-averaged energy).

Validity: a dihedral is undefined when either of its bond angles is within ~1.15° of 0/180
(min(sin θ1, sin θ2) < 0.02, θ from RDKit GetAngleDeg). Such rows keep RDKit's value (NaN when exactly
collinear) but have ``valid = False``; they are
excluded from circular statistics downstream and counted per parameter.

Scan flags (TorsionDrive topologies, and the constrained optimizations with frozen dihedrals):
  ``on_driven_bond`` / ``on_frozen_bond``: the torsion's central bond is a driven/frozen bond, so its
  value is imposed by the scan/constraint rather than relaxed.

Output: data/processed/values.parquet, one row per (asg_idx, conf_idx).
"""

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry import Point3D

SIN_THRESHOLD = 0.02


def rdkit_conformers(xyz: np.ndarray) -> list:
    """RDKit Conformer objects for coordinates of shape (n_conf, n_atoms, 3), in Å."""
    confs = []
    for frame in xyz:
        conf = Chem.Conformer(len(frame))
        for i, (x, y, z) in enumerate(frame):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        confs.append(conf)
    return confs


def bond_lengths(xyz: np.ndarray, atoms: np.ndarray) -> np.ndarray:
    """Bond lengths (Å) from RDKit rdMolTransforms.GetBondLength."""
    confs = rdkit_conformers(xyz)
    return np.array([[rdMolTransforms.GetBondLength(c, int(i), int(j)) for i, j in atoms] for c in confs])


def angles_deg(xyz: np.ndarray, atoms: np.ndarray) -> np.ndarray:
    """Angles (degrees) from RDKit rdMolTransforms.GetAngleDeg."""
    confs = rdkit_conformers(xyz)
    return np.array([[rdMolTransforms.GetAngleDeg(c, int(i), int(j), int(k)) for i, j, k in atoms] for c in confs])


def dihedrals_deg(xyz: np.ndarray, atoms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed dihedrals (degrees) from RDKit rdMolTransforms.GetDihedralDeg (IUPAC sign, as OpenMM) and
    a validity mask: undefined when either bond angle (RDKit GetAngleDeg) has sin < SIN_THRESHOLD.
    RDKit returns NaN for exactly collinear atoms; those rows are invalid and kept as NaN."""
    confs = rdkit_conformers(xyz)
    values = np.empty((len(confs), len(atoms)))
    valid = np.empty((len(confs), len(atoms)), dtype=bool)
    for c, conf in enumerate(confs):
        for n, (i, j, k, l) in enumerate(atoms):
            i, j, k, l = int(i), int(j), int(k), int(l)
            values[c, n] = rdMolTransforms.GetDihedralDeg(conf, i, j, k, l)
            sin1 = np.sin(np.radians(rdMolTransforms.GetAngleDeg(conf, i, j, k)))
            sin2 = np.sin(np.radians(rdMolTransforms.GetAngleDeg(conf, j, k, l)))
            valid[c, n] = min(sin1, sin2) >= SIN_THRESHOLD and np.isfinite(values[c, n])
    return values, valid


def fourier_energy(
    phi_deg: np.ndarray, terms: list[tuple[int, float, float]]
) -> np.ndarray:
    """sum_n k_eff (1 + cos(n phi - phase)); terms = [(periodicity, phase_deg, k_effective)]."""
    phi = np.radians(phi_deg)
    return sum(k * (1.0 + np.cos(n * phi - np.radians(phase))) for n, phase, k in terms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed", type=pathlib.Path, default=pathlib.Path("data/processed")
    )
    args = parser.parse_args()
    p = args.processed

    params = pd.read_parquet(p / "params.parquet")
    terms = {
        r.param_id: list(
            zip(
                json.loads(r.periodicity),
                json.loads(r.phase_deg),
                json.loads(r.k_effective),
            )
        )
        for r in params[
            params.handler.isin(["ProperTorsions", "ImproperTorsions"])
        ].itertuples()
    }
    assignments = pd.read_parquet(p / "assignments.parquet")
    improper_terms = pd.read_parquet(p / "improper_terms.parquet")
    conformers = pd.read_parquet(
        p / "conformers.parquet",
        columns=["conf_idx", "topology_idx", "coords_angstrom"],
    )
    constraints = pd.read_parquet(p / "constraints.parquet")

    central_bonds = {
        kind: {(t, *sorted(a[1:3])) for t, a in zip(g.topology_idx, g.atoms)}
        for kind, g in constraints.groupby("kind")
    }

    frames = []
    asg_by_topology = dict(tuple(assignments.groupby("topology_idx")))
    terms_by_topology = dict(tuple(improper_terms.groupby("topology_idx")))
    for topology_idx, confs in conformers.groupby("topology_idx"):
        xyz = np.stack([np.asarray(c).reshape(-1, 3) for c in confs.coords_angstrom])
        conf_idx = confs.conf_idx.to_numpy()
        asg = asg_by_topology[topology_idx]
        for handler, rows in asg.groupby("handler"):
            atoms = np.stack(rows.atoms.to_numpy()).astype(int)
            n_conf, n_int = len(conf_idx), len(rows)
            energy = np.full((n_conf, n_int), np.nan)
            improper_deg = None
            if handler == "Bonds":
                value, valid = bond_lengths(xyz, atoms), np.ones((n_conf, n_int), bool)
            elif handler == "Angles":
                value, valid = angles_deg(xyz, atoms), np.ones((n_conf, n_int), bool)
            elif handler == "ProperTorsions":
                value, valid = dihedrals_deg(xyz, atoms)
                for j, pid in enumerate(rows.param_id):
                    energy[:, j] = fourier_energy(value[:, j], terms[pid])
            else:
                t = (
                    terms_by_topology[topology_idx]
                    .set_index("asg_idx")
                    .loc[rows.asg_idx]
                )
                term_atoms = np.stack(t.atoms.to_numpy()).astype(int)
                term_deg, term_valid = dihedrals_deg(xyz, term_atoms)
                # 3 consecutive terms per group, in rows order
                order = np.argsort(
                    pd.Index(rows.asg_idx).get_indexer(t.index), kind="stable"
                )
                term_deg, term_valid = term_deg[:, order].reshape(
                    n_conf, n_int, 3
                ), term_valid[:, order].reshape(n_conf, n_int, 3)
                t_ids = t.param_id.to_numpy()[order].reshape(n_int, 3)
                value, valid = term_deg[..., 0], term_valid.all(axis=-1)
                for j in range(n_int):
                    energy[:, j] = sum(
                        fourier_energy(term_deg[:, j, m], terms[t_ids[j, m]])
                        for m in range(3)
                    )
                improper_deg = term_deg

            central = [
                (
                    (topology_idx, *sorted(a[1:3]))
                    if len(a) == 4 and handler == "ProperTorsions"
                    else None
                )
                for a in atoms
            ]
            frame = pd.DataFrame(
                {
                    "asg_idx": np.tile(rows.asg_idx.to_numpy(), n_conf),
                    "conf_idx": np.repeat(conf_idx, n_int),
                    "value": value.ravel(),
                    "valid": valid.ravel(),
                    "energy_kcal": energy.ravel(),
                    "on_driven_bond": np.tile(
                        [c in central_bonds.get("driven", ()) for c in central], n_conf
                    ),
                    "on_frozen_bond": np.tile(
                        [c in central_bonds.get("frozen", ()) for c in central], n_conf
                    ),
                }
            )
            if improper_deg is not None:
                frame["improper_terms_deg"] = list(improper_deg.reshape(-1, 3))
            frames.append(frame)

    values = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["asg_idx", "conf_idx"])
        .reset_index(drop=True)
    )
    values.to_parquet(p / "values.parquet", index=False)

    merged = values.merge(assignments[["asg_idx", "handler"]], on="asg_idx")
    summary = {
        "n_values": int(len(values)),
        "per_handler": merged.groupby("handler").size().to_dict(),
        "invalid_dihedrals": merged[~merged.valid].groupby("handler").size().to_dict(),
        "bond_length_range": [
            float(merged[merged.handler == "Bonds"].value.min()),
            float(merged[merged.handler == "Bonds"].value.max()),
        ],
        "angle_range": [
            float(merged[merged.handler == "Angles"].value.min()),
            float(merged[merged.handler == "Angles"].value.max()),
        ],
        "n_on_driven_bond": int(values.on_driven_bond.sum()),
        "n_on_frozen_bond": int(values.on_frozen_bond.sum()),
        "nan_values": int(values.value.isna().sum()),
    }
    (p / "geometry_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
