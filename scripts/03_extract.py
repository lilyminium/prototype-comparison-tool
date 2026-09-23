"""Extract topologies, conformers and constraints from the raw optimization view and the TorsionDrive pull.

Each QC record is kept in its OWN atom order: the topology is built from that record's
CMILES (``canonical_isomeric_explicit_hydrogen_mapped_smiles``), whose map index i refers to
QC atom i-1. That correspondence is verified for every record and every TorsionDrive grid point:

  - element symbols identical and in the same order;
  - the bond set identical to the QC ``connectivity`` field.

A record failing either check is written to exclusions (fail closed); nothing is remapped.

Diagnostics that do NOT exclude:
  - ``connectivity_changed``: bonds perceived from the final geometry (RDKit DetermineConnectivity)
    differ from the topology;
  - ``grid_deviation_deg``: |measured driven dihedral - grid key| (wrapped) for TorsionDrive points.

All OpenFF calls run under an RDKit-only toolkit registry, entered inside each worker.

Outputs (data/processed/):
  molecules.parquet    mol_idx, identity SMILES, display SMILES, charge
  topologies.parquet   topology_idx, source, entry, record id, mol_idx, mapped SMILES,
                       MDL-aromatic explicit-H SMILES + atom order for RDKit.js
  conformers.parquet   conf_idx, topology_idx, coordinates (Å), energies, diagnostics, QM spec
  constraints.parquet  topology_idx, atoms, kind ("driven" TorsionDrive dihedral or "frozen" constraint)
  exclusions.parquet   every raw item not carried forward, with the stage and reason
"""

import argparse
import hashlib
import json
import multiprocessing
import pathlib

import numpy as np
import pandas as pd

BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_KCAL = 627.509474
OPT_VIEW = "OpenFF-SMIRNOFF-Sage-2.3.0_optimization_view.sqlite"
# Declared before looking at results: geomeTRIC enforces dihedral constraints to well below this
GRID_TOLERANCE_DEG = 0.1


def rdkit_only():
    from openff.toolkit.utils.toolkit_registry import (
        ToolkitRegistry,
        toolkit_registry_manager,
    )
    from openff.toolkit.utils.toolkits import BuiltInToolkitWrapper, RDKitToolkitWrapper

    return toolkit_registry_manager(
        ToolkitRegistry([RDKitToolkitWrapper, BuiltInToolkitWrapper])
    )


def dihedral_deg(xyz: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """Signed dihedral in degrees from RDKit rdMolTransforms.GetDihedralDeg (IUPAC/OpenMM convention)."""
    from rdkit import Chem
    from rdkit.Chem import rdMolTransforms
    from rdkit.Geometry import Point3D

    conf = Chem.Conformer(len(xyz))
    for n, (x, y, z) in enumerate(xyz):
        conf.SetAtomPosition(n, Point3D(float(x), float(y), float(z)))
    return float(rdMolTransforms.GetDihedralDeg(conf, int(i), int(j), int(k), int(l)))


def wrap(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def perceived_bonds(
    symbols: list[str], xyz: np.ndarray, charge: int
) -> set[tuple[int, int]] | None:
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    block = f"{len(symbols)}\n\n" + "\n".join(
        f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(symbols, xyz)
    )
    mol = Chem.MolFromXYZBlock(block)
    try:
        rdDetermineBonds.DetermineConnectivity(mol, charge=charge)
    except Exception:
        return None
    return {
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()
    }


def site_smiles(molecule) -> dict:
    """SMILES for the browser: OpenFF's RDKit (MDL-aromatic) form with explicit H, plus atom order."""
    from rdkit import Chem

    rdmol = molecule.to_rdkit()  # OpenFF applies MDL aromaticity here
    mdl = Chem.MolToSmiles(rdmol, canonical=True)
    order = list(rdmol.GetProp("_smilesAtomOutputOrder").strip("[],").split(","))
    return {
        "mdl_smiles": mdl,
        # mdl_smiles atom n corresponds to topology atom mdl_order[n]
        "mdl_order": json.dumps([int(x) for x in order if x != ""]),
    }


def process_topology(task: dict) -> dict:
    """Build one topology and validate all of its conformers. Runs in a worker process."""
    from openff.toolkit import Molecule
    from rdkit import Chem

    out = {"topology": None, "conformers": [], "constraints": [], "exclusions": []}
    base_excl = {k: task[k] for k in ("source", "entry_name", "record_id")}
    with rdkit_only():
        try:
            molecule = Molecule.from_mapped_smiles(
                task["mapped_smiles"], allow_undefined_stereo=True
            )
        except Exception as e:
            out["exclusions"].append(
                {
                    **base_excl,
                    "conf_record_id": None,
                    "stage": "from_mapped_smiles",
                    "reason": repr(e),
                }
            )
            return out

        symbols = [a.symbol for a in molecule.atoms]
        bonds = {tuple(sorted((b.atom1_index, b.atom2_index))) for b in molecule.bonds}
        charge = int(round(molecule.total_charge.m))
        identity = molecule.to_smiles(
            isomeric=True, explicit_hydrogens=True, mapped=False
        )
        display = Chem.MolToSmiles(Chem.RemoveHs(molecule.to_rdkit()))
        smiles = site_smiles(molecule)

        for conf in task["conformers"]:
            excl = {**base_excl, "conf_record_id": conf["conf_record_id"], "grid_key": conf.get("grid_key")}
            if conf["symbols"] != symbols:
                out["exclusions"].append(
                    {
                        **excl,
                        "stage": "atom_order",
                        "reason": "QC symbols differ from mapped SMILES order",
                    }
                )
                continue
            qc_bonds = {
                tuple(sorted((int(a), int(b)))) for a, b, *_ in conf["connectivity"]
            }
            if qc_bonds != bonds:
                out["exclusions"].append(
                    {
                        **excl,
                        "stage": "atom_order",
                        "reason": f"QC connectivity differs: {len(qc_bonds ^ bonds)} bonds",
                    }
                )
                continue
            xyz = (
                np.asarray(conf["geometry_bohr"], dtype=float).reshape(-1, 3)
                * BOHR_TO_ANGSTROM
            )
            perceived = perceived_bonds(symbols, xyz, charge)
            row = {
                "topology_key": task["topology_key"],
                "conf_record_id": conf["conf_record_id"],
                "grid_key": conf.get("grid_key"),
                "energy_hartree": conf.get("energy_hartree"),
                "spec": conf["spec"],
                "coords_angstrom": xyz.ravel().tolist(),
                "connectivity_changed": (
                    None if perceived is None else perceived != bonds
                ),
                "grid_deviation_deg": None,
            }
            if task["driven"] and conf.get("grid_key") is not None:
                (target,) = json.loads(conf["grid_key"])
                row["grid_deviation_deg"] = abs(
                    wrap(dihedral_deg(xyz, *task["driven"]) - target)
                )
            out["conformers"].append(row)

        if not out["conformers"]:
            return out

        out["topology"] = {
            "topology_key": task["topology_key"],
            **base_excl,
            "identity_smiles": identity,
            "display_smiles": display,
            "mapped_smiles": task["mapped_smiles"],
            "n_atoms": molecule.n_atoms,
            "charge": charge,
            **smiles,
        }
        for kind, atoms in [("driven", task["driven"])] + [
            ("frozen", f) for f in task["frozen"]
        ]:
            if atoms is None:
                continue
            chain = all(tuple(sorted(p)) in bonds for p in zip(atoms, atoms[1:]))
            out["constraints"].append(
                {
                    "topology_key": task["topology_key"],
                    "kind": kind,
                    "atoms": list(atoms),
                    "is_bonded_chain": chain,
                }
            )
    return out


def spec_summary(spec: dict) -> str:
    """Level of theory + optimizer settings, excluding per-molecule dihedral/constraint indices."""
    qc = spec["qc_specification"]
    keywords = {
        k: v for k, v in (spec.get("keywords") or {}).items() if k != "constraints"
    }
    return json.dumps(
        {
            "qc_program": qc["program"],
            "method": qc["method"].replace("(bj)", "bj"),
            "basis": qc["basis"],
            "optimizer": spec["program"],
            "optimizer_keywords": keywords,
        },
        sort_keys=True,
    )


def optimization_tasks(raw: pathlib.Path) -> list[dict]:
    import qcportal

    ds = qcportal.load_dataset_view(raw / OPT_VIEW)
    entries = {e.name: e for e in ds.iterate_entries()}
    tasks = []
    for entry_name, _spec_name, record in ds.iterate_records():
        spec = record.specification.model_dump()
        freeze = ((spec.get("keywords") or {}).get("constraints") or {}).get(
            "freeze"
        ) or []
        mol = record.final_molecule
        tasks.append(
            {
                "topology_key": f"opt:{record.id}",
                "source": "opt",
                "entry_name": entry_name,
                "record_id": record.id,
                "mapped_smiles": entries[entry_name].attributes[
                    "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                ],
                "driven": None,
                "frozen": [
                    tuple(f["indices"]) for f in freeze if f["type"] == "dihedral"
                ],
                "conformers": [
                    {
                        "conf_record_id": record.id,
                        "symbols": list(mol.symbols),
                        "geometry_bohr": mol.geometry.ravel().tolist(),
                        "connectivity": [list(b) for b in mol.connectivity],
                        "energy_hartree": (
                            record.energies[-1] if record.energies else None
                        ),
                        "spec": spec_summary(spec),
                    }
                ],
            }
        )
        if any(f["type"] != "dihedral" for f in freeze):
            raise NotImplementedError(f"non-dihedral constraint in record {record.id}")
    return tasks


def torsiondrive_tasks(raw: pathlib.Path) -> list[dict]:
    df = pd.read_parquet(raw / "td_qcarchive.parquet")
    manifest = json.loads((raw / "td_qcarchive_manifest.json").read_text())
    specs = {
        name: spec_summary(s["optimization_specification"])
        for name, s in manifest["specifications"].items()
    }
    tasks = []
    for td_id, group in df.groupby("td_record_id", sort=True):
        first = group.iloc[0]
        (driven,) = json.loads(first.dihedrals)
        tasks.append(
            {
                "topology_key": f"td:{td_id}",
                "source": "td",
                "entry_name": first.entry_name,
                "record_id": int(td_id),
                "mapped_smiles": first.mapped_smiles,
                "driven": tuple(driven),
                "frozen": [],
                "conformers": [
                    {
                        "conf_record_id": int(row.opt_record_id),
                        "grid_key": row.grid_key,
                        "symbols": json.loads(row.symbols),
                        "geometry_bohr": list(row.geometry_bohr),
                        "connectivity": json.loads(row.connectivity),
                        "energy_hartree": row.final_energy_hartree,
                        "spec": specs[row.specification_name],
                    }
                    for row in group.itertuples()
                ],
            }
        )
    return tasks


def source_keys(raw: pathlib.Path, processed: pathlib.Path) -> set[tuple]:
    """Exact keys of every raw item, from the persisted ledger and the hash-verified TD pull.

    Key = (source, parent record id, grid key or None, child optimization record id).
    """
    ledger = pd.read_parquet(processed / "ledger_opt.parquet")
    td = pd.read_parquet(raw / "td_qcarchive.parquet")
    manifest = json.loads((raw / "td_qcarchive_manifest.json").read_text())
    sha = hashlib.sha256((raw / "td_qcarchive.parquet").read_bytes()).hexdigest()
    if sha != manifest["sha256"]:
        raise RuntimeError("td_qcarchive.parquet does not match the sha256 recorded when it was fetched")
    opt_keys = [("opt", int(r), None, int(r)) for r in ledger.record_id]
    td_keys = [("td", int(a), g, int(c)) for a, g, c in zip(td.td_record_id, td.grid_key, td.opt_record_id)]
    for name, keys in (("optimization ledger", opt_keys), ("TorsionDrive pull", td_keys)):
        if len(set(keys)) != len(keys):
            raise RuntimeError(f"duplicate source keys in {name}")
    grid = [(k[1], k[2]) for k in td_keys]
    if len(set(grid)) != len(grid):
        raise RuntimeError("more than one selected optimization for a (TorsionDrive, grid key)")
    return set(opt_keys) | set(td_keys)


def output_keys(conformers: pd.DataFrame, exclusions: pd.DataFrame) -> list[tuple]:
    def key(source, parent, grid, child):
        return (source, int(parent), None if grid is None or pd.isna(grid) else grid, int(child))

    kept = [key(tk.split(":")[0], tk.split(":")[1], g, c) for tk, g, c in zip(conformers.topology_key, conformers.grid_key, conformers.conf_record_id)]
    dropped = [key(s_, r, g, c) for s_, r, g, c in zip(exclusions.source, exclusions.record_id, exclusions.grid_key, exclusions.conf_record_id)]
    return kept + dropped


def reconcile(expected: set[tuple], produced: list[tuple]) -> None:
    """Every raw item is kept or excluded exactly once: no missing, duplicated or substituted keys."""
    if len(set(produced)) != len(produced):
        raise RuntimeError("an item was output more than once")
    missing, extra = expected - set(produced), set(produced) - expected
    if missing or extra:
        raise RuntimeError(f"reconciliation failed: {len(missing)} missing (e.g. {sorted(missing, key=str)[:3]}), {len(extra)} unexpected (e.g. {sorted(extra, key=str)[:3]})")


def check_gates(conformers: pd.DataFrame) -> None:
    """Finite coordinates everywhere; every TorsionDrive driven dihedral within tolerance of its grid key."""
    if not all(np.isfinite(np.asarray(c, dtype=float)).all() for c in conformers.coords_angstrom):
        raise RuntimeError("non-finite coordinates")
    td_rows = conformers.grid_key.notna()
    dev = conformers.loc[td_rows, "grid_deviation_deg"].astype(float)
    ok = np.isfinite(dev) & (dev <= GRID_TOLERANCE_DEG)
    if not ok.all():
        bad = conformers.loc[td_rows][~ok.to_numpy()]
        raise RuntimeError(
            f"{len(bad)} TorsionDrive grid points deviate > {GRID_TOLERANCE_DEG}° from their grid key "
            f"(or are non-finite), e.g. {bad[['topology_key', 'grid_key', 'grid_deviation_deg']].head(3).to_dict('records')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=pathlib.Path, default=pathlib.Path("data/raw"))
    parser.add_argument(
        "--out", type=pathlib.Path, default=pathlib.Path("data/processed")
    )
    parser.add_argument(
        "--nproc", type=int, default=max(1, multiprocessing.cpu_count() - 2)
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tasks = optimization_tasks(args.raw) + torsiondrive_tasks(args.raw)
    n_raw_conformers = sum(len(t["conformers"]) for t in tasks)
    print(
        f"{len(tasks)} topologies, {n_raw_conformers} conformers to process", flush=True
    )

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(args.nproc) as pool:
        results = pool.map(process_topology, tasks, chunksize=16)

    topologies = pd.DataFrame(
        [r["topology"] for r in results if r["topology"] is not None]
    )
    conformers = pd.DataFrame([c for r in results for c in r["conformers"]])
    constraints = pd.DataFrame([c for r in results for c in r["constraints"]])
    exclusions = pd.DataFrame(
        [e for r in results for e in r["exclusions"]],
        columns=[
            "source",
            "entry_name",
            "record_id",
            "conf_record_id",
            "grid_key",
            "stage",
            "reason",
        ],
    )

    # Closed world against the persisted ledger / hash-verified TD pull (not the tasks built above)
    reconcile(source_keys(args.raw, args.out), output_keys(conformers, exclusions))

    check_gates(conformers)

    # Integer indices. mol_idx groups topologies by chemical identity only; it never transfers atom indices.
    molecules = (
        topologies.groupby("identity_smiles", sort=True)
        .agg(display_smiles=("display_smiles", "first"), charge=("charge", "first"))
        .reset_index()
    )
    molecules.insert(0, "mol_idx", np.arange(len(molecules)))
    topologies = topologies.sort_values(["source", "record_id"]).reset_index(drop=True)
    topologies.insert(0, "topology_idx", np.arange(len(topologies)))
    topologies = topologies.merge(
        molecules[["identity_smiles", "mol_idx"]], on="identity_smiles"
    )
    key_to_idx = dict(zip(topologies.topology_key, topologies.topology_idx))

    conformers["topology_idx"] = conformers.topology_key.map(key_to_idx)
    conformers = conformers.sort_values(["topology_idx", "conf_record_id"]).reset_index(
        drop=True
    )
    conformers.insert(0, "conf_idx", np.arange(len(conformers)))
    # Relative energy within each TorsionDrive, referenced to that drive's minimum
    is_td = conformers.grid_key.notna()
    rel = (
        conformers[is_td]
        .groupby("topology_idx")
        .energy_hartree.transform(lambda e: (e - e.min()) * HARTREE_TO_KCAL)
    )
    conformers["rel_energy_kcal"] = rel
    constraints["topology_idx"] = constraints.topology_key.map(key_to_idx)

    for name, df in [
        ("molecules", molecules),
        ("topologies", topologies.drop(columns="topology_key")),
        ("conformers", conformers.drop(columns="topology_key")),
        ("constraints", constraints.drop(columns="topology_key")),
        ("exclusions", exclusions),
    ]:
        df.to_parquet(args.out / f"{name}.parquet", index=False)

    td_dev = conformers.grid_deviation_deg.dropna()
    summary = {
        "n_molecules": len(molecules),
        "n_topologies": {
            k: int(v) for k, v in topologies.source.value_counts().items()
        },
        "n_conformers": int(len(conformers)),
        "n_exclusions": (
            exclusions.groupby(["source", "stage"]).size().to_dict()
            if len(exclusions)
            else {}
        ),
        "connectivity_changed": int(
            conformers.connectivity_changed.fillna(False).sum()
        ),
        "connectivity_undetermined": int(conformers.connectivity_changed.isna().sum()),
        "constraints": constraints.groupby("kind").size().to_dict(),
        "constraints_not_bonded_chain": int((~constraints.is_bonded_chain).sum()),
        "grid_deviation_deg": {
            "max": float(td_dev.max()),
            "p99": float(td_dev.quantile(0.99)),
            "n_over_1deg": int((td_dev > 1).sum()),
            "tolerance_deg": GRID_TOLERANCE_DEG,
        },
        "n_distinct_specs": int(conformers.spec.nunique()),
    }
    summary["n_exclusions"] = {
        f"{k[0]}:{k[1]}": v for k, v in summary["n_exclusions"].items()
    }
    (args.out / "extract_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
