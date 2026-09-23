"""Validate the torsion geometry + energies of 05_geometry.py against OpenMM.

Two independent checks, both on the OpenMM Reference platform (double precision):

1. Real systems. For every topology, build the OpenMM system with Interchange from the pinned
   force field, as a full Sage 2.3.0 system with the force field's NAGL charges. The exported
   PeriodicTorsionForce entries are split into proper and improper forces (by matching the atom
   tuples in improper_terms.parquet), each placed in its own System, and evaluated on EVERY
   conformer. Proper and improper totals are compared separately with the sums of
   values.parquet ``energy_kcal``. Also checked: the number of exported entries equals the number
   of Fourier components implied by assignments + params.

2. Synthetic one-torsion systems. For every distinct torsion parameter, a 4-particle system with
   a PeriodicTorsionForce built from OUR params table is evaluated on a grid of dihedrals from -180
   to 180 degrees (5° steps, both atom orders), and compared term-by-term with fourier_energy().
   This exercises signs, phases, idivf and multi-component parameters independently of real
   geometries.

Tolerance: declared up front as 1e-6 kJ/mol absolute per system; the observed maximum is reported.
"""

import argparse
import importlib.util
import json
import multiprocessing
import pathlib

import numpy as np
import pandas as pd

KCAL_TO_KJ = 4.184
TOLERANCE_KJ = 1e-6
SCRIPTS = pathlib.Path(__file__).parent

_spec = importlib.util.spec_from_file_location("geometry", SCRIPTS / "05_geometry.py")
geometry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(geometry)

_FF = None


def _init_worker(offxml: str) -> None:
    # NAGL runs on PyTorch; one thread per worker avoids oversubscribing the cores across the pool
    import torch

    torch.set_num_threads(1)
    global _FF
    from openff.toolkit import ForceField

    _FF = ForceField(offxml)


def check_nagl_model(offxml: str) -> None:
    """The installed NAGL model file must match the sha256 pinned in the OFFXML."""
    import hashlib
    import pathlib

    import openff.nagl_models
    from openff.toolkit import ForceField

    handler = ForceField(offxml).get_parameter_handler("NAGLCharges")
    root = pathlib.Path(openff.nagl_models.__file__).parent
    (path,) = list(root.rglob(handler.model_file))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != handler.model_file_hash:
        raise SystemExit(f"NAGL model {handler.model_file}: sha256 {sha} != OFFXML {handler.model_file_hash}")


def rdkit_only():
    """RDKit for cheminformatics, NAGL for the force field's NAGLCharges (no OpenEye, no AmberTools)."""
    from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
    from openff.toolkit.utils.toolkit_registry import (
        ToolkitRegistry,
        toolkit_registry_manager,
    )
    from openff.toolkit.utils.toolkits import BuiltInToolkitWrapper, RDKitToolkitWrapper

    return toolkit_registry_manager(
        ToolkitRegistry([RDKitToolkitWrapper, NAGLToolkitWrapper, BuiltInToolkitWrapper])
    )


def torsion_energy_kj(entries: list, coords_nm: list[np.ndarray]) -> list[float]:
    import openmm
    from openmm import unit

    n_atoms = coords_nm[0].shape[0]
    system = openmm.System()
    for _ in range(n_atoms):
        system.addParticle(1.0)
    force = openmm.PeriodicTorsionForce()
    for e in entries:
        force.addTorsion(*e)
    system.addForce(force)
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(1.0),
        openmm.Platform.getPlatformByName("Reference"),
    )
    energies = []
    for xyz in coords_nm:
        context.setPositions(xyz)
        energies.append(
            context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
    return energies


def isolated_entry_energies(entries: list, xyz_nm: np.ndarray) -> list[float]:
    """Energy of each PeriodicTorsionForce entry alone (all others set to k=0), OpenMM Reference."""
    import openmm
    from openmm import unit

    if not entries:
        return []
    system = openmm.System()
    for _ in range(xyz_nm.shape[0]):
        system.addParticle(1.0)
    force = openmm.PeriodicTorsionForce()
    for e in entries:
        force.addTorsion(*e[:4], e[4], e[5], 0.0)
    system.addForce(force)
    context = openmm.Context(system, openmm.VerletIntegrator(1.0), openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(xyz_nm)
    out = []
    for i, e in enumerate(entries):
        force.setTorsionParameters(i, *e[:4], e[4], e[5], e[6])
        force.updateParametersInContext(context)
        out.append(context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        force.setTorsionParameters(i, *e[:4], e[4], e[5], 0.0)
    return out


def interaction_energies_openmm(isolated: list, improper_group: dict) -> dict:
    """Sum isolated entry energies per interaction key: propers by canonical atom tuple, impropers by group."""
    out: dict = {}
    for atoms, energy in isolated:
        t = tuple(atoms)
        key = ("improper", improper_group[t]) if t in improper_group else ("proper", min(t, t[::-1]))
        out[key] = out.get(key, 0.0) + energy
    return out


def compare_interactions(ours: dict, openmm_: dict, tol: float) -> list:
    """Keys missing on either side, non-finite values, or |ΔE| > tol (kJ/mol)."""
    bad = []
    for key in set(ours) | set(openmm_):
        a, b = ours.get(key), openmm_.get(key)
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)) or abs(a - b) > tol:
            bad.append((key, a, b))
    return bad


def validate_topology(task: dict) -> dict:
    import openmm
    from openff.interchange import Interchange
    from openff.interchange.exceptions import UnassignedValenceError
    from openff.toolkit import Molecule

    with rdkit_only():
        molecule = Molecule.from_mapped_smiles(
            task["mapped_smiles"], allow_undefined_stereo=True
        )
        try:
            # Full Sage 2.3.0 system: charges from the force field's own NAGLCharges handler
            # (openff-gnn-am1bcc-1.0.0.pt, hash pinned in the OFFXML and checked below)
            interchange = Interchange.from_smirnoff(_FF, molecule.to_topology())
        except UnassignedValenceError as e:
            # Recorded in unassigned.parquet by 04_label.py; OpenMM cannot be built
            return {"topology_idx": task["topology_idx"], "skipped": type(e).__name__}
        system = interchange.to_openmm_system()

    (force,) = [
        f for f in system.getForces() if isinstance(f, openmm.PeriodicTorsionForce)
    ]
    improper_set = {tuple(a) for a in task["improper_atoms"]}
    proper, improper = [], []
    for i in range(force.getNumTorsions()):
        *atoms, n, phase, k = force.getTorsionParameters(i)
        (improper if tuple(atoms) in improper_set else proper).append(
            (*atoms, n, phase, k)
        )

    coords_nm = [np.asarray(c).reshape(-1, 3) / 10.0 for c in task["coords"]]
    isolated = isolated_entry_energies(proper + improper, coords_nm[0])
    e_proper = (
        torsion_energy_kj(proper, coords_nm) if proper else [0.0] * len(coords_nm)
    )
    e_improper = (
        torsion_energy_kj(improper, coords_nm) if improper else [0.0] * len(coords_nm)
    )
    return {
        "topology_idx": task["topology_idx"],
        "n_proper_entries": len(proper),
        "n_improper_entries": len(improper),
        "expected_proper_entries": task["expected_proper_entries"],
        "expected_improper_entries": task["expected_improper_entries"],
        "conf_idx": task["conf_idx"],
        "openmm_proper_kj": e_proper,
        "openmm_improper_kj": e_improper,
        "first_conf_idx": task["conf_idx"][0],
        # (atoms, kJ/mol) for every exported entry, evaluated alone on the first conformer
        "isolated": [(list(map(int, e[:4])), en) for e, en in zip(proper + improper, isolated)],
    }


def synthetic_checks(params: pd.DataFrame) -> dict:
    """Each torsion parameter on a dihedral grid: OpenMM vs fourier_energy, both atom orders."""
    import openmm
    from openmm import unit

    grid = np.arange(-180.0, 180.1, 5.0)
    # Build 4-atom coordinates spanning the full dihedral range, nm
    base = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.15, 0.0]])
    worst = 0.0
    n_checked = 0
    torsions = params[params.handler.isin(["ProperTorsions", "ImproperTorsions"])]
    term_sets = [
        list(zip(json.loads(r.periodicity), json.loads(r.phase_deg), json.loads(r.k_effective)))
        for r in torsions.itertuples()
    ]
    # Sage phases are 0/180 only, which makes E(phi) = E(-phi) and hides sign errors;
    # add asymmetric-phase terms so a flipped dihedral sign cannot pass.
    term_sets.append([(1, 37.0, 1.3), (2, 90.0, 0.7), (3, 211.0, 0.4), (4, 300.0, 0.25)])
    for terms in term_sets:
        for order in ([0, 1, 2, 3], [3, 2, 1, 0]):
            system = openmm.System()
            for _ in range(4):
                system.addParticle(1.0)
            force = openmm.PeriodicTorsionForce()
            for n, phase, k in terms:
                force.addTorsion(*order, n, np.radians(phase), k * KCAL_TO_KJ)
            system.addForce(force)
            ctx = openmm.Context(
                system,
                openmm.VerletIntegrator(1.0),
                openmm.Platform.getPlatformByName("Reference"),
            )
            for phi in grid:
                t = np.radians(phi)
                # atom 3 rotated about the 1->2 (y) axis; OpenMM measures the dihedral itself
                p3 = base[2] + 0.1 * np.array([np.cos(t), 0.0, -np.sin(t)])
                xyz = np.vstack([base, p3])
                measured = geometry.dihedrals_deg(xyz[None], np.array([order]))[0][0, 0]
                ctx.setPositions(xyz)
                e_mm = (
                    ctx.getState(getEnergy=True)
                    .getPotentialEnergy()
                    .value_in_unit(unit.kilojoule_per_mole)
                )
                e_ours = (
                    geometry.fourier_energy(np.array([measured]), terms)[0] * KCAL_TO_KJ
                )
                diff = abs(e_mm - e_ours)
                worst = diff if not np.isfinite(diff) else max(worst, diff)
                n_checked += 1
    return {"n_evaluations": n_checked, "max_abs_diff_kj": worst}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=pathlib.Path, default=pathlib.Path("data/raw"))
    parser.add_argument(
        "--processed", type=pathlib.Path, default=pathlib.Path("data/processed")
    )
    parser.add_argument(
        "--nproc", type=int, default=max(1, multiprocessing.cpu_count() - 2)
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="also swap two stored proper energies in memory and require the per-interaction check to fail",
    )
    args = parser.parse_args()
    p = args.processed

    check_nagl_model(str(args.raw / "openff_unconstrained-2.3.0.offxml"))
    params = pd.read_parquet(p / "params.parquet")
    n_components = {
        r.param_id: len(json.loads(r.periodicity))
        for r in params.itertuples()
        if isinstance(r.periodicity, str)
    }
    synthetic = synthetic_checks(params)
    print("synthetic:", synthetic, flush=True)

    topologies = pd.read_parquet(
        p / "topologies.parquet", columns=["topology_idx", "mapped_smiles"]
    )
    conformers = pd.read_parquet(
        p / "conformers.parquet",
        columns=["conf_idx", "topology_idx", "coords_angstrom"],
    )
    assignments = pd.read_parquet(p / "assignments.parquet")
    improper_terms = pd.read_parquet(p / "improper_terms.parquet")
    values = pd.read_parquet(
        p / "values.parquet", columns=["asg_idx", "conf_idx", "energy_kcal"]
    )
    values = values.merge(assignments[["asg_idx", "handler"]], on="asg_idx")
    if not np.isfinite(values.energy_kcal.dropna()).all():
        raise SystemExit("non-finite stored torsion energies")
    ours = (
        values.dropna(subset=["energy_kcal"])
        .groupby(["conf_idx", "handler"])
        .energy_kcal.sum()
        .unstack(fill_value=0.0)
    )

    tasks = []
    confs_by_top = dict(tuple(conformers.groupby("topology_idx")))
    terms_by_top = dict(tuple(improper_terms.groupby("topology_idx")))
    asg_by_top = dict(tuple(assignments.groupby("topology_idx")))
    for t in topologies.itertuples():
        a = asg_by_top[t.topology_idx]
        c = confs_by_top[t.topology_idx]
        tasks.append(
            {
                "topology_idx": int(t.topology_idx),
                "mapped_smiles": t.mapped_smiles,
                "improper_atoms": (
                    [list(x) for x in terms_by_top[t.topology_idx].atoms]
                    if t.topology_idx in terms_by_top
                    else []
                ),
                "expected_proper_entries": int(
                    sum(
                        n_components[i]
                        for i in a[a.handler == "ProperTorsions"].param_id
                    )
                ),
                "expected_improper_entries": int(
                    3
                    * sum(
                        n_components[i]
                        for i in a[a.handler == "ImproperTorsions"].param_id
                    )
                ),
                "conf_idx": c.conf_idx.tolist(),
                "coords": [list(x) for x in c.coords_angstrom],
            }
        )

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        args.nproc,
        initializer=_init_worker,
        initargs=(str(args.raw / "openff_unconstrained-2.3.0.offxml"),),
    ) as pool:
        results = pool.map(validate_topology, tasks, chunksize=8)

    skipped = {r["topology_idx"]: r["skipped"] for r in results if "skipped" in r}
    expected_skips = set(pd.read_parquet(p / "unassigned.parquet").topology_idx)
    if set(skipped) != expected_skips:
        raise SystemExit(f"skipped topologies {sorted(skipped)} != unassigned {sorted(expected_skips)}")
    results = [r for r in results if "skipped" not in r]
    rows = []
    for r in results:
        for ci, ep, ei in zip(
            r["conf_idx"], r["openmm_proper_kj"], r["openmm_improper_kj"]
        ):
            rows.append(
                {"conf_idx": ci, "openmm_proper_kj": ep, "openmm_improper_kj": ei}
            )
    mm = pd.DataFrame(rows).set_index("conf_idx")
    mm["ours_proper_kj"] = (
        ours.get("ProperTorsions", 0.0).reindex(mm.index, fill_value=0.0) * KCAL_TO_KJ
    )
    mm["ours_improper_kj"] = (
        ours.get("ImproperTorsions", 0.0).reindex(mm.index, fill_value=0.0) * KCAL_TO_KJ
    )
    d_proper = (mm.openmm_proper_kj - mm.ours_proper_kj).abs()
    d_improper = (mm.openmm_improper_kj - mm.ours_improper_kj).abs()
    n_nonfinite = int((~np.isfinite(mm.to_numpy())).any(axis=1).sum())

    # Per-interaction: our stored energy of each interaction (first conformer) vs OpenMM isolated entries
    torsion_rows = values.dropna(subset=["energy_kcal"]).merge(assignments[["asg_idx", "topology_idx", "atoms"]], on="asg_idx")
    first_conf = {r["topology_idx"]: r["first_conf_idx"] for r in results}
    torsion_rows = torsion_rows[torsion_rows.conf_idx == torsion_rows.topology_idx.map(first_conf)]

    def ours_by_topology(rows):
        out = {}
        for r in rows.itertuples():
            a = tuple(int(x) for x in r.atoms)
            key = ("improper", int(r.asg_idx)) if r.handler == "ImproperTorsions" else ("proper", min(a, a[::-1]))
            out.setdefault(int(r.topology_idx), {})[key] = r.energy_kcal * KCAL_TO_KJ
        return out

    groups = {(int(t), tuple(int(x) for x in a)): int(g) for t, a, g in zip(improper_terms.topology_idx, improper_terms.atoms, improper_terms.asg_idx)}

    def per_interaction_failures(ours_map):
        failures = []
        for r in results:
            t = r["topology_idx"]
            improper_group = {tuple(a): groups[(t, tuple(a))] for a, _ in r["isolated"] if (t, tuple(a)) in groups}
            mm_map = interaction_energies_openmm(r["isolated"], improper_group)
            failures += [(t, *b) for b in compare_interactions(ours_map.get(t, {}), mm_map, TOLERANCE_KJ)]
        return failures

    ours_map = ours_by_topology(torsion_rows)
    interaction_failures = per_interaction_failures(ours_map)
    n_interactions = sum(len(v) for v in ours_map.values())

    self_test = None
    if args.self_test:
        # Swap the stored energies of two different proper interactions within one topology
        for t, m in ours_map.items():
            keys = [k for k in m if k[0] == "proper" and abs(m[k]) > 1e-3]
            pairs = [(a, b) for a in keys for b in keys if abs(m[a] - m[b]) > 1e-3]
            if pairs:
                a, b = pairs[0]
                corrupted = {**ours_map, t: {**m, a: m[b], b: m[a]}}
                self_test = {"topology_idx": t, "detected": len(per_interaction_failures(corrupted)) > 0}
                break
        if not self_test or not self_test["detected"]:
            raise SystemExit(f"self-test FAILED: swapped energies were not detected ({self_test})")
    count_mismatch = [
        r["topology_idx"]
        for r in results
        if r["n_proper_entries"] != r["expected_proper_entries"]
        or r["n_improper_entries"] != r["expected_improper_entries"]
    ]

    summary = {
        "platform": "Reference",
        "tolerance_kj": TOLERANCE_KJ,
        "synthetic": synthetic,
        "n_conformers": int(len(mm)),
        "proper_max_abs_diff_kj": float(d_proper.max()),
        "improper_max_abs_diff_kj": float(d_improper.max()),
        "proper_p99_abs_diff_kj": float(d_proper.quantile(0.99)),
        "improper_p99_abs_diff_kj": float(d_improper.quantile(0.99)),
        "n_over_tolerance": int(
            ((d_proper > TOLERANCE_KJ) | (d_improper > TOLERANCE_KJ)).sum()
        ),
        "n_nonfinite_conformers": n_nonfinite,
        "per_interaction": {
            "n_interactions": n_interactions,
            "n_failures": len(interaction_failures),
            "examples": [str(x) for x in interaction_failures[:5]],
            "self_test": self_test,
        },
        "entry_count_mismatch_topologies": count_mismatch[:20],
        "n_entry_count_mismatch": len(count_mismatch),
        "skipped_topologies_unassigned": {str(k): v for k, v in skipped.items()},
    }
    (p / "energy_validation.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    if (
        summary["n_over_tolerance"]
        or n_nonfinite
        or interaction_failures
        or count_mismatch
        or not np.isfinite(synthetic["max_abs_diff_kj"])
        or synthetic["max_abs_diff_kj"] > TOLERANCE_KJ
    ):
        raise SystemExit("energy validation FAILED")


if __name__ == "__main__":
    main()
