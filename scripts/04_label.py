"""Assign Sage 2.3.0 valence parameters to every topology with ForceField.label_molecules.

label_molecules is the authority for which parameter applies to which atoms. OpenFF Interchange's
SMIRNOFF collections (the code that builds OpenMM systems) are used for two things only:

  - expanding each improper (one label_molecules match) into the three trefoil terms, in exactly
    the atom order Interchange passes to OpenMM: (central, a, b, c);
  - resolving torsion parameters (k, periodicity, phase, idivf incl. default_idivf="auto").

Every topology is cross-checked: the atoms->parameter assignments from Interchange must equal those
from label_molecules, for all four handlers. The force field is loaded intact from the pinned,
hashed OFFXML; all OpenFF calls run under an RDKit-only toolkit registry.

Outputs (data/processed/):
  params.parquet          closed-world roster of all Bonds/Angles/ProperTorsions/ImproperTorsions
                          parameters, in handler (hierarchy) order, including zero-coverage ones
  assignments.parquet     one row per (topology, handler, interaction): param id, atom tuple as
                          returned by label_molecules. Impropers: one row per trefoil group
                          (atoms[1] is the central atom)
  improper_terms.parquet  3 rows per improper group: ordered 4-tuple (central first) as emitted to OpenMM
"""

import argparse
import hashlib
import json
import multiprocessing
import pathlib

import numpy as np
import pandas as pd

HANDLERS = ["Bonds", "Angles", "ProperTorsions", "ImproperTorsions"]

# Interactions known to have no Sage 2.3.0 parameter, as (source, record id, handler, sorted atoms).
# Anything else unassigned fails the build. Found 2026-09-23 and independently confirmed (GPT-6 Astra):
# O=S=C/[N+]([O-])=[NH+]\[O-], four proper torsions about the [N+]=[NH+] bond.
EXPECTED_UNASSIGNED = {
    ("opt", 146497953, "ProperTorsions", (2, 3, 5, 6)),
    ("opt", 146497953, "ProperTorsions", (2, 3, 5, 8)),
    ("opt", 146497953, "ProperTorsions", (4, 3, 5, 6)),
    ("opt", 146497953, "ProperTorsions", (4, 3, 5, 8)),
}


def check_coverage(topologies: pd.DataFrame, assignments: pd.DataFrame, unassigned: pd.DataFrame) -> list[str]:
    """Every topology labeled; unassigned interactions exactly the declared exception set."""
    problems = []
    unlabeled = set(topologies.topology_idx) - set(assignments.topology_idx)
    if unlabeled:
        problems.append(f"{len(unlabeled)} topologies without any assignment, e.g. {sorted(unlabeled)[:5]}")
    key = topologies.set_index("topology_idx")[["source", "record_id"]]
    found = {
        (key.at[t, "source"], int(key.at[t, "record_id"]), h, tuple(int(a) for a in atoms))
        for t, h, atoms in zip(unassigned.topology_idx, unassigned.handler, unassigned.atoms)
    }
    if found != EXPECTED_UNASSIGNED:
        problems.append(f"unassigned interactions differ from the declared set: new {sorted(found - EXPECTED_UNASSIGNED)[:5]}, gone {sorted(EXPECTED_UNASSIGNED - found)[:5]}")
    return problems
# Topology attribute enumerating every interaction each handler should cover. Impropers are not
# required for every trivalent centre, so they have no completeness requirement.
UNIVERSE = {"Bonds": "bonds", "Angles": "angles", "ProperTorsions": "propers"}
OFFXML = "openff_unconstrained-2.3.0.offxml"

_FF = None


def _init_worker(offxml: str) -> None:
    global _FF
    from openff.toolkit import ForceField

    _FF = ForceField(offxml)


def rdkit_only():
    from openff.toolkit.utils.toolkit_registry import (
        ToolkitRegistry,
        toolkit_registry_manager,
    )
    from openff.toolkit.utils.toolkits import BuiltInToolkitWrapper, RDKitToolkitWrapper

    return toolkit_registry_manager(
        ToolkitRegistry([RDKitToolkitWrapper, BuiltInToolkitWrapper])
    )


def canonical(handler: str, atoms: tuple[int, ...]) -> tuple[int, ...]:
    """Order-independent key of an interaction (reversal for chains; outer-set for impropers)."""
    if handler == "ImproperTorsions":
        return (atoms[1], *sorted((atoms[0], atoms[2], atoms[3])))
    return min(tuple(atoms), tuple(reversed(atoms)))


def label_topology(task: tuple[int, str]) -> dict:
    from openff.interchange.smirnoff._valence import (
        SMIRNOFFAngleCollection,
        SMIRNOFFBondCollection,
        SMIRNOFFImproperTorsionCollection,
        SMIRNOFFProperTorsionCollection,
    )
    from openff.toolkit import Molecule

    topology_idx, mapped_smiles = task
    collections = {
        "Bonds": SMIRNOFFBondCollection,
        "Angles": SMIRNOFFAngleCollection,
        "ProperTorsions": SMIRNOFFProperTorsionCollection,
        "ImproperTorsions": SMIRNOFFImproperTorsionCollection,
    }
    out = {"assignments": [], "improper_terms": [], "mismatches": [], "unassigned": []}
    with rdkit_only():
        topology = Molecule.from_mapped_smiles(
            mapped_smiles, allow_undefined_stereo=True
        ).to_topology()
        labels = _FF.label_molecules(topology)[0]

        for handler in HANDLERS:
            param_handler = _FF.get_parameter_handler(handler)
            smirks_to_id = {p.smirks: p.id for p in param_handler.parameters}
            from_label = {
                canonical(handler, atoms): p.id for atoms, p in labels[handler].items()
            }
            # label_molecules silently omits interactions no parameter matches; enumerate the
            # full universe from the topology so that gaps are recorded, not lost
            if handler in UNIVERSE:
                universe = {
                    canonical(
                        handler,
                        tuple(
                            a.molecule_atom_index
                            for a in ((item.atom1, item.atom2) if handler == "Bonds" else item)
                        ),
                    )
                    for item in getattr(topology, UNIVERSE[handler])
                }
                for atoms in sorted(universe - set(from_label)):
                    out["unassigned"].append(
                        {"topology_idx": topology_idx, "handler": handler, "atoms": list(atoms)}
                    )
            for atoms, parameter in labels[handler].items():
                out["assignments"].append(
                    {
                        "topology_idx": topology_idx,
                        "handler": handler,
                        "param_id": parameter.id,
                        "atoms": list(atoms),
                    }
                )

            collection = collections[handler]()
            collection.store_matches(param_handler, topology)
            from_interchange = {}
            for top_key, pot_key in collection.key_map.items():
                atoms = tuple(top_key.atom_indices)
                if handler == "ImproperTorsions":
                    # Interchange keys are (central, a, b, c); label_molecules keys are (a, central, b, c)
                    key = (atoms[0], *sorted(atoms[1:]))
                    if pot_key.mult == 0:
                        out["improper_terms"].append(
                            {
                                "topology_idx": topology_idx,
                                "param_id": smirks_to_id[pot_key.id],
                                "group_key": list(key),
                                "atoms": list(atoms),
                            }
                        )
                else:
                    key = canonical(handler, atoms)
                from_interchange[key] = smirks_to_id[pot_key.id]
            if from_interchange != from_label:
                diff = set(from_interchange.items()) ^ set(from_label.items())
                out["mismatches"].append(
                    {
                        "topology_idx": topology_idx,
                        "handler": handler,
                        "n": len(diff),
                        "example": str(sorted(diff)[:3]),
                    }
                )
    return out


def parameter_roster(offxml: str) -> pd.DataFrame:
    """Every valence parameter, with torsion idivf resolved exactly as Interchange does."""
    from openff.toolkit import ForceField

    ff = ForceField(offxml)
    rows = []
    for handler in HANDLERS:
        param_handler = ff.get_parameter_handler(handler)
        for order, p in enumerate(param_handler.parameters):
            row = {
                "handler": handler,
                "param_id": p.id,
                "hierarchy_index": order,
                "smirks": p.smirks,
            }
            if handler == "Bonds":
                row |= {
                    "length_angstrom": p.length.m_as("angstrom"),
                    "k": p.k.m_as("kilocalorie / mole / angstrom ** 2"),
                }
            elif handler == "Angles":
                row |= {
                    "angle_deg": p.angle.m_as("degree"),
                    "k": p.k.m_as("kilocalorie / mole / radian ** 2"),
                }
            else:
                n_terms = len(p.k)
                if p.idivf is not None:
                    idivf = [float(x) for x in p.idivf]
                elif param_handler.default_idivf == "auto":
                    # Interchange: ImproperTorsions "auto" -> 3.0; ProperTorsions in Sage always set idivf
                    idivf = [3.0 if handler == "ImproperTorsions" else 1.0] * n_terms
                    if handler == "ProperTorsions":
                        raise NotImplementedError(
                            f"{p.id}: proper without explicit idivf"
                        )
                else:
                    idivf = [float(param_handler.default_idivf)] * n_terms
                row |= {
                    "periodicity": json.dumps([int(x) for x in p.periodicity]),
                    "phase_deg": json.dumps([x.m_as("degree") for x in p.phase]),
                    "k_raw": json.dumps([x.m_as("kilocalorie / mole") for x in p.k]),
                    "idivf": json.dumps(idivf),
                    "k_effective": json.dumps(
                        [x.m_as("kilocalorie / mole") / d for x, d in zip(p.k, idivf)]
                    ),
                }
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=pathlib.Path, default=pathlib.Path("data/raw"))
    parser.add_argument(
        "--processed", type=pathlib.Path, default=pathlib.Path("data/processed")
    )
    parser.add_argument(
        "--nproc", type=int, default=max(1, multiprocessing.cpu_count() - 2)
    )
    args = parser.parse_args()

    offxml = str(args.raw / OFFXML)
    manifest = json.loads((args.raw / "download_manifest.json").read_text())
    sha = hashlib.sha256(pathlib.Path(offxml).read_bytes()).hexdigest()
    if sha != manifest["offxml"]["sha256"]:
        raise RuntimeError("OFFXML does not match the hash recorded at download")

    params = parameter_roster(offxml)
    params.to_parquet(args.processed / "params.parquet", index=False)

    topologies = pd.read_parquet(args.processed / "topologies.parquet")
    tasks = list(zip(topologies.topology_idx.astype(int), topologies.mapped_smiles))
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(args.nproc, initializer=_init_worker, initargs=(offxml,)) as pool:
        results = pool.map(label_topology, tasks, chunksize=16)

    assignments = pd.DataFrame([a for r in results for a in r["assignments"]])
    improper_terms = pd.DataFrame([t for r in results for t in r["improper_terms"]])
    mismatches = pd.DataFrame([m for r in results for m in r["mismatches"]])
    unassigned = pd.DataFrame(
        [u for r in results for u in r["unassigned"]], columns=["topology_idx", "handler", "atoms"]
    )
    unassigned.to_parquet(args.processed / "unassigned.parquet", index=False)

    assignments.insert(0, "asg_idx", np.arange(len(assignments)))
    # Link each improper term to its group (label_molecules row)
    impropers = assignments[assignments.handler == "ImproperTorsions"]
    group_keys = {
        (t, a[1], *sorted((a[0], a[2], a[3]))): i
        for i, t, a in zip(impropers.asg_idx, impropers.topology_idx, impropers.atoms)
    }
    improper_terms["asg_idx"] = [
        group_keys.get((t, *k))
        for t, k in zip(improper_terms.topology_idx, improper_terms.group_key)
    ]

    assignments.to_parquet(args.processed / "assignments.parquet", index=False)
    improper_terms.drop(columns="group_key").to_parquet(
        args.processed / "improper_terms.parquet", index=False
    )

    counts = assignments.groupby("param_id").size()
    summary = {
        "offxml_sha256": sha,
        "n_params": params.groupby("handler").size().to_dict(),
        "n_assignments": assignments.groupby("handler").size().to_dict(),
        "n_improper_terms": len(improper_terms),
        "improper_terms_per_group_ok": bool(
            (improper_terms.groupby("asg_idx").size() == 3).all()
        ),
        "improper_terms_unlinked": int(improper_terms.asg_idx.isna().sum()),
        "label_vs_interchange_mismatches": (
            0 if mismatches.empty else int(mismatches.n.sum())
        ),
        "zero_coverage_params": sorted(set(params.param_id) - set(counts.index)),
        "n_unassigned": unassigned.groupby("handler").size().to_dict(),
        "n_topologies_with_unassigned": int(unassigned.topology_idx.nunique()),
    }
    if not mismatches.empty:
        mismatches.to_parquet(args.processed / "label_mismatches.parquet", index=False)
    (args.processed / "label_summary.json").write_text(
        json.dumps(summary, indent=1) + "\n"
    )
    print(json.dumps(summary, indent=1))
    coverage_problems = check_coverage(topologies, assignments, unassigned)
    for problem in coverage_problems:
        print("COVERAGE:", problem)
    if (
        not mismatches.empty
        or summary["improper_terms_unlinked"]
        or not summary["improper_terms_per_group_ok"]
        or coverage_problems
    ):
        raise SystemExit("labeling cross-check failed")


if __name__ == "__main__":
    main()
