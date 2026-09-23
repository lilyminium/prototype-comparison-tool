"""Export reference rows from the Python pipeline for the node tests of the site code (tests/js/).

Writes tests/js/fixtures.json:
  rows    random sample of values.parquet rows with everything needed to locate them in the shipped
          shards (param, position of the assignment within its shard, conformer index within the
          topology) and the float64 reference value / improper terms / torsion energy
  smirks  every force-field SMIRKS with the tag -> query-atom indices OpenFF uses
"""

import argparse
import json
import pathlib

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed", type=pathlib.Path, default=pathlib.Path("data/processed")
    )
    parser.add_argument(
        "--out", type=pathlib.Path, default=pathlib.Path("tests/js/fixtures.json")
    )
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    p = args.processed

    assignments = pd.read_parquet(p / "assignments.parquet")
    conformers = pd.read_parquet(
        p / "conformers.parquet", columns=["conf_idx", "topology_idx"]
    )
    values = pd.read_parquet(p / "values.parquet")

    # Position of each assignment within its parameter's shard (shards are in asg_idx order)
    assignments = assignments.sort_values("asg_idx")
    assignments["shard_pos"] = assignments.groupby("param_id").cumcount()
    conf_start = conformers.groupby("topology_idx").conf_idx.min()

    rng = np.random.default_rng(args.seed)
    # Stratify so every handler is represented, plus all invalid dihedrals up to 200
    picks = []
    merged = values.merge(assignments[["asg_idx", "handler"]], on="asg_idx")
    for handler, g in merged.groupby("handler"):
        picks.append(
            g.sample(
                n=min(len(g), args.n // 4), random_state=int(rng.integers(1 << 31))
            )
        )
    invalid = merged[~merged.valid]
    picks.append(invalid.sample(n=min(len(invalid), 200), random_state=1))
    sample = pd.concat(picks).drop_duplicates(["asg_idx", "conf_idx"])
    sample = sample.drop(columns="handler").merge(assignments, on="asg_idx")
    sample["conf_j"] = sample.conf_idx - sample.topology_idx.map(conf_start)

    rows = []
    for r in sample.itertuples():
        rows.append(
            {
                "param_id": r.param_id,
                "handler": r.handler,
                "shard_pos": int(r.shard_pos),
                "topology_idx": int(r.topology_idx),
                "conf_j": int(r.conf_j),
                "atoms": [int(a) for a in r.atoms],
                "value": float(r.value),
                "valid": bool(r.valid),
                "energy_kcal": None if pd.isna(r.energy_kcal) else float(r.energy_kcal),
                "improper_terms_deg": (
                    None
                    if r.improper_terms_deg is None
                    else [float(x) for x in r.improper_terms_deg]
                ),
            }
        )
    smirks = json.loads((p / "parity_input.json").read_text())["smirks"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows, "smirks": smirks}))
    print(f"wrote {len(rows)} rows, {len(smirks)} SMIRKS to {args.out}")


if __name__ == "__main__":
    main()
