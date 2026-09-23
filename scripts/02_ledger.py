"""Closed-world ledger of every raw item in the dataset views, written BEFORE extraction.

Every row written here must later resolve to either an extracted output row or an explicit
exclusion row (see 03_extract.py). Nothing is filtered at this stage.

Outputs (data/processed/):
  ledger_opt.parquet   one row per (entry, specification, record)
  ledger_specs.json    every distinct raw specification (canonical JSON) keyed by sha256
  ledger_summary.json  counts per view, status counts, specification counts
"""

import argparse
import collections
import hashlib
import json
import pathlib

import pandas as pd
import qcportal

OPT_VIEW = "OpenFF-SMIRNOFF-Sage-2.3.0_optimization_view.sqlite"
MAPPED_KEY = "canonical_isomeric_explicit_hydrogen_mapped_smiles"


def spec_hash(spec) -> tuple[str, str]:
    """Hash of the canonical JSON of a raw QCPortal specification."""
    canonical = json.dumps(spec.dict(), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16], canonical


def ledger_optimization(path: pathlib.Path, specs: dict) -> pd.DataFrame:
    ds = qcportal.load_dataset_view(path)
    entries = {e.name: e for e in ds.iterate_entries()}
    rows = []
    for entry_name, spec_name, record in ds.iterate_records():
        entry = entries[entry_name]
        h, canonical = spec_hash(record.specification)
        specs.setdefault(h, json.loads(canonical))
        final = record.final_molecule
        rows.append(
            {
                "source": "opt",
                "entry_name": entry_name,
                "specification_name": spec_name,
                "record_id": record.id,
                "status": str(record.status.value),
                "spec_hash": h,
                "mapped_smiles": entry.attributes.get(MAPPED_KEY),
                "n_atoms_initial": len(entry.initial_molecule.symbols),
                "n_atoms_final": None if final is None else len(final.symbols),
                "has_connectivity": final is not None
                and final.connectivity is not None,
                "n_energies": len(record.energies or []),
            }
        )
    df = pd.DataFrame(rows)
    # Closed world: every entry appears exactly once with a record
    missing = set(entries) - set(df.entry_name)
    if missing:
        raise RuntimeError(
            f"{len(missing)} entries without records, e.g. {sorted(missing)[:5]}"
        )
    if df.entry_name.duplicated().any():
        raise RuntimeError("entries with more than one record")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=pathlib.Path, default=pathlib.Path("data/raw"))
    parser.add_argument(
        "--out", type=pathlib.Path, default=pathlib.Path("data/processed")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    specs: dict = {}
    opt = ledger_optimization(args.raw / OPT_VIEW, specs)
    opt.to_parquet(args.out / "ledger_opt.parquet", index=False)

    summary = {
        "opt": {
            "n_entries": int(opt.entry_name.nunique()),
            "n_records": int(opt.record_id.nunique()),
            "status": dict(collections.Counter(opt.status)),
            "n_missing_mapped_smiles": int(opt.mapped_smiles.isna().sum()),
            "n_missing_final_molecule": int(opt.n_atoms_final.isna().sum()),
            "n_missing_connectivity": int((~opt.has_connectivity).sum()),
            "n_unique_mapped_smiles": int(opt.mapped_smiles.nunique()),
            "spec_counts": dict(collections.Counter(opt.spec_hash)),
        },
    }
    (args.out / "ledger_specs.json").write_text(
        json.dumps(specs, indent=1, sort_keys=True) + "\n"
    )
    (args.out / "ledger_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
