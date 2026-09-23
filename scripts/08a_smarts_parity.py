"""SMARTS/SMIRKS matching parity between OpenFF (Python) and the shipped RDKit.js, on the training set.

The website matches user SMARTS (and Sage SMIRKS) against training molecules in the browser. Those
molecules are shipped as OpenFF's MDL-aromatic explicit-H SMILES and loaded in RDKit.js with
{setAromaticity: false, kekulize: false, removeHs: false}. This test checks that the browser then
finds exactly the tagged-atom matches OpenFF finds.

Reference: RDKitToolkitWrapper._find_smarts_matches (what label_molecules uses) on
Molecule.to_rdkit() (MDL aromaticity), unique=False. Every SMIRKS of every handler in the force
field (not only valence) is tested against every topology; each (topology, SMIRKS) pair with
matches is summarised by the count and an md5 of the sorted tagged tuples (topology atom indices).

  python scripts/08a_smarts_parity.py           -> data/processed/parity_input.json, parity_python.tsv
  node   scripts/08b_smarts_parity.mjs          -> data/processed/parity_js.tsv
  python scripts/08a_smarts_parity.py --compare -> data/processed/parity_summary.json
"""

import argparse
import hashlib
import json
import multiprocessing
import pathlib

import pandas as pd

_SMIRKS: list[str] = []


def _init(smirks: list[str]) -> None:
    global _SMIRKS
    _SMIRKS = smirks


def digest(matches) -> str:
    canonical = json.dumps(sorted(list(m) for m in set(matches)), separators=(",", ":"))
    return hashlib.md5(canonical.encode()).hexdigest()


def python_matches(task: tuple[int, str]) -> list[str]:
    from openff.toolkit import Molecule
    from openff.toolkit.utils.toolkits import RDKitToolkitWrapper

    topology_idx, mapped_smiles = task
    rdmol = Molecule.from_mapped_smiles(
        mapped_smiles, allow_undefined_stereo=True
    ).to_rdkit()
    lines = []
    for s_idx, smirks in enumerate(_SMIRKS):
        matches = RDKitToolkitWrapper._find_smarts_matches(rdmol, smirks, unique=False)
        if matches:
            lines.append(
                f"{topology_idx}\t{s_idx}\t{len(set(matches))}\t{digest(matches)}"
            )
    return lines


def tag_indices(smirks: str) -> list[int]:
    """Query-atom index of tags :1..:n, computed exactly as OpenFF's _find_smarts_matches does."""
    from rdkit import Chem

    qmol = Chem.MolFromSmarts(smirks)
    idx_map = {a.GetAtomMapNum() - 1: a.GetIdx() for a in qmol.GetAtoms() if a.GetAtomMapNum()}
    return [idx_map[x] for x in sorted(idx_map)]


def force_field_smirks(offxml: str) -> list[dict]:
    from openff.toolkit import ForceField

    ff = ForceField(offxml)
    out = []
    for name in ff.registered_parameter_handlers:
        handler = ff.get_parameter_handler(name)
        for p in getattr(handler, "parameters", []):
            if getattr(p, "smirks", None):
                out.append(
                    {
                        "handler": name,
                        "param_id": getattr(p, "id", None),
                        "smirks": p.smirks,
                        "tag_index": tag_indices(p.smirks),
                    }
                )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=pathlib.Path, default=pathlib.Path("data/raw"))
    parser.add_argument(
        "--processed", type=pathlib.Path, default=pathlib.Path("data/processed")
    )
    parser.add_argument(
        "--nproc", type=int, default=max(1, multiprocessing.cpu_count() - 2)
    )
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    p = args.processed

    if args.compare:
        cols = ["topology_idx", "smirks_idx", "n", "md5"]
        py = pd.read_csv(p / "parity_python.tsv", sep="\t", names=cols)
        js = pd.read_csv(p / "parity_js.tsv", sep="\t", names=cols)
        merged = py.merge(
            js,
            on=["topology_idx", "smirks_idx"],
            how="outer",
            suffixes=("_py", "_js"),
            indicator=True,
        )
        bad = merged[(merged._merge != "both") | (merged.md5_py != merged.md5_js)]
        smirks = json.loads((p / "parity_input.json").read_text())["smirks"]
        summary = {
            "n_topologies": int(
                json.loads((p / "parity_input.json").read_text())["n_topologies"]
            ),
            "n_smirks": len(smirks),
            "n_pairs_with_matches_python": int(len(py)),
            "n_pairs_with_matches_js": int(len(js)),
            "n_mismatched_pairs": int(len(bad)),
            "mismatched_smirks": sorted(
                {smirks[i]["param_id"] or smirks[i]["smirks"] for i in bad.smirks_idx}
            )[:50],
        }
        bad.to_csv(p / "parity_mismatches.tsv", sep="\t", index=False)
        (p / "parity_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps(summary, indent=1))
        if len(bad):
            raise SystemExit("SMARTS parity FAILED")
        return

    smirks = force_field_smirks(str(args.raw / "openff_unconstrained-2.3.0.offxml"))
    topologies = pd.read_parquet(
        p / "topologies.parquet",
        columns=["topology_idx", "mapped_smiles", "mdl_smiles", "mdl_order"],
    )
    (p / "parity_input.json").write_text(
        json.dumps(
            {
                "n_topologies": len(topologies),
                "smirks": smirks,
                "molecules": [
                    {
                        "topology_idx": int(t),
                        "mdl_smiles": s,
                        "mdl_order": json.loads(o),
                    }
                    for t, s, o in zip(
                        topologies.topology_idx,
                        topologies.mdl_smiles,
                        topologies.mdl_order,
                    )
                ],
            }
        )
    )
    tasks = list(zip(topologies.topology_idx.astype(int), topologies.mapped_smiles))
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        args.nproc, initializer=_init, initargs=([s["smirks"] for s in smirks],)
    ) as pool:
        results = pool.map(python_matches, tasks, chunksize=16)
    with open(p / "parity_python.tsv", "w") as f:
        for lines in results:
            for line in lines:
                f.write(line + "\n")
    print(
        f"{len(smirks)} SMIRKS x {len(tasks)} topologies; {sum(map(len, results))} pairs with matches"
    )


if __name__ == "__main__":
    main()
