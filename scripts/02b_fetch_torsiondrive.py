"""Fetch the Sage 2.3.0 TorsionDrive training set directly from QCArchive.

The 14.7 GB Zenodo TorsionDrive view stores every child optimization in full; we only need the
minimum-energy optimization at each grid point. QCPortal can batch-fetch exactly those children
up front with ``include=["minimum_optimizations", ...]``, which takes minutes instead of hours.

The dataset is addressed by name (the same dataset the Zenodo view was made from). Provenance
(server, dataset id, qcportal version, fetch time) is written alongside the data.

Output: data/raw/td_qcarchive.parquet, one row per (TorsionDrive record, grid key), plus
data/raw/td_qcarchive_manifest.json.
"""

import argparse
import datetime
import hashlib
import json
import pathlib

import pandas as pd
import qcportal

SERVER = "https://api.qcarchive.molssi.org:443"
DATASET_TYPE = "torsiondrive"
DATASET_NAME = "OpenFF SMIRNOFF Sage 2.3.0"
MAPPED_KEY = "canonical_isomeric_explicit_hydrogen_mapped_smiles"
INCLUDE = ["minimum_optimizations", "final_molecule", "initial_molecules"]


def spec_hash(spec) -> str:
    canonical = json.dumps(spec.model_dump(), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/raw"))
    args = parser.parse_args()

    client = qcportal.PortalClient(SERVER)
    ds = client.get_dataset(DATASET_TYPE, DATASET_NAME)
    entries = {e.name: e for e in ds.iterate_entries()}
    specs = {
        name: ds.specifications[name].specification for name in ds.specification_names
    }

    rows, td_seen = [], set()
    for i, (entry_name, spec_name, td) in enumerate(
        ds.iterate_records(include=INCLUDE)
    ):
        if i % 100 == 0:
            print(f"{i} TorsionDrive records fetched", flush=True)
        entry = entries[entry_name]
        keywords = td.specification.keywords
        base = {
            "entry_name": entry_name,
            "specification_name": spec_name,
            "td_record_id": td.id,
            "td_status": str(td.status.value),
            "spec_hash": spec_hash(td.specification),
            "mapped_smiles": (entry.attributes or {}).get(MAPPED_KEY),
            "dihedrals": json.dumps([list(d) for d in keywords.dihedrals]),
            "grid_spacing": json.dumps(list(keywords.grid_spacing)),
            "dihedral_ranges": json.dumps(keywords.dihedral_ranges),
            "energy_upper_limit": keywords.energy_upper_limit,
        }
        td_seen.add(td.id)
        minimum = td.minimum_optimizations_ or {}
        if not minimum:
            rows.append({**base, "grid_key": None})
            continue
        final_energies = {json.dumps(list(k)): v for k, v in td.final_energies.items()}
        for key, opt in td.minimum_optimizations.items():
            key_json = json.dumps(list(key))
            mol = opt.final_molecule
            rows.append(
                {
                    **base,
                    "grid_key": key_json,
                    "opt_record_id": opt.id,
                    "opt_status": str(opt.status.value),
                    "final_energy_hartree": final_energies.get(key_json),
                    "symbols": json.dumps(list(mol.symbols)),
                    "geometry_bohr": mol.geometry.ravel().tolist(),
                    "connectivity": json.dumps(
                        [list(b) for b in (mol.connectivity or [])]
                    ),
                    "molecular_charge": mol.molecular_charge,
                }
            )

    missing_entries = set(entries) - {r["entry_name"] for r in rows}
    if missing_entries:
        raise RuntimeError(
            f"{len(missing_entries)} entries without records: {sorted(missing_entries)[:5]}"
        )

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "td_qcarchive.parquet"
    df.to_parquet(out, index=False)

    manifest = {
        "server": SERVER,
        "dataset_type": DATASET_TYPE,
        "dataset_name": DATASET_NAME,
        "dataset_id": ds.id,
        "qcportal_version": qcportal.__version__,
        "fetched_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "include": INCLUDE,
        "n_entries": len(entries),
        "n_td_records": len(td_seen),
        "n_rows": len(df),
        "specifications": {k: v.model_dump() for k, v in specs.items()},
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    (args.out / "td_qcarchive_manifest.json").write_text(
        json.dumps(manifest, indent=1, default=str) + "\n"
    )
    print(json.dumps({k: manifest[k] for k in ("n_entries", "n_td_records", "n_rows")}))


if __name__ == "__main__":
    main()
