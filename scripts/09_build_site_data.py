"""Build the static data files served by the GitHub Pages site (docs/data/). See docs/data/SCHEMA.md.

Files (all little-endian; JSON files use column arrays to stay small):
  meta.json            provenance, counts, validation results, QM specifications, file list with
                       sha256 + bytes, size-gate results  (initial load)
  params.json          closed-world valence parameter roster + precomputed summary statistics (initial load)
  molecules.json       per unique molecule: display SMILES, charge, topology list (initial load)
  topologies.json      per QC record: source, ids, MDL SMILES + atom order, conformer/coordinate
                       offsets, driven/frozen dihedrals, unassigned interactions (initial load)
  conformers.json      per conformer: grid angle, relative energy, spec index, flags (lazy)
  coords_opt.bin       Float32 xyz (Å) of optimization conformers (lazy, SMARTS geometry / viewer)
  coords_td.bin        Float32 xyz (Å) of TorsionDrive grid points (lazy, opt-in)
  fp_morgan.bin        Uint8, 256 bytes per molecule (Morgan r=2, 2048 bits; vendored RDKit.js)
  shards/<param>.bin   per-parameter observations (lazy, one file per parameter)
  assignments.bin      per-topology parameter assignments (lazy): which parameter covers which atoms

Size gates (build fails if exceeded): total ≤ 60 MB compressed (gzip -6, as a proxy for HTTP
compression); initial load ≤ 5 MB compressed excluding RDKit.js; any single file ≤ 10 MB compressed.
"""

import argparse
import gzip
import hashlib
import json
import pathlib
import shutil
import struct

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
HANDLER_CODE = {"Bonds": 0, "Angles": 1, "ProperTorsions": 2, "ImproperTorsions": 3}
INITIAL_FILES = ["meta.json", "params.json", "molecules.json", "topologies.json"]
GATES_MB = {
    "total_compressed": 60.0,
    "initial_compressed": 5.0,
    "file_compressed": 10.0,
}

# Asg flag bits (per assignment, constant across conformers)
FLAG_ON_DRIVEN_BOND = 1
FLAG_ON_FROZEN_BOND = 2
FLAG_IS_DRIVEN_TORSION = 4


# Statistic definitions are shared with docs/js/stats.js (weightedLinearStats, weightedCircularStats);
# tests/js/site.test.mjs checks that both give the same numbers on the shipped data.
def circular_stats(deg: np.ndarray, w: np.ndarray) -> dict:
    """Weighted circular mean, circular SD sqrt(-2 ln R) and mean resultant length R (degrees)."""
    if len(deg) == 0:
        return {"circ_mean": None, "circ_std": None, "resultant_length": None}
    rad = np.radians(deg)
    c, s = (w * np.cos(rad)).sum() / w.sum(), (w * np.sin(rad)).sum() / w.sum()
    r = min(1.0, float(np.hypot(c, s)))
    return {
        "circ_mean": float(np.degrees(np.arctan2(s, c))),
        "circ_std": float(np.degrees(np.sqrt(-2.0 * np.log(r)))) if r > 0 else None,
        "resultant_length": r,
    }


def linear_stats(x: np.ndarray, w: np.ndarray) -> dict:
    """Weighted mean, population SD (divide by total weight), lower weighted median, min, max."""
    if len(x) == 0:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None}
    sw = w.sum()
    mean = float((w * x).sum() / sw)
    order = np.argsort(x, kind="stable")
    cum = np.cumsum(w[order])
    median = float(x[order][np.searchsorted(cum, sw / 2)])
    return {
        "mean": mean,
        "std": float(np.sqrt((w * (x - mean) ** 2).sum() / sw)),
        "median": median,
        "min": float(x.min()),
        "max": float(x.max()),
    }


def write_json(path: pathlib.Path, obj) -> None:
    path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False))


def pad4(buf: bytearray) -> None:
    buf.extend(b"\0" * (-len(buf) % 4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=pathlib.Path, default=pathlib.Path("data/raw"))
    parser.add_argument(
        "--processed", type=pathlib.Path, default=pathlib.Path("data/processed")
    )
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/data"))
    args = parser.parse_args()
    p, out = args.processed, args.out
    # Build into a staging directory; the existing output is replaced only after every gate passes.
    # SCHEMA.md is hand-written and carried over.
    KEEP = {"SCHEMA.md"}
    final = out
    if final.name != "data" or not (final.parent / "index.html").exists():
        raise SystemExit(f"refusing to write to {final}: expected the site's data directory (docs/data)")
    if final.exists() and not ((final / "meta.json").exists() or {c.name for c in final.iterdir()} <= KEEP):
        raise SystemExit(f"refusing to replace {final}: it does not look like generated site data")
    out = final.parent / (final.name + ".staging")
    if out.exists():
        shutil.rmtree(out)  # leftover from an interrupted build of this script
    (out / "shards").mkdir(parents=True)
    for name in KEEP:
        if (final / name).exists():
            shutil.copy2(final / name, out / name)

    molecules = pd.read_parquet(p / "molecules.parquet")
    topologies = (
        pd.read_parquet(p / "topologies.parquet")
        .sort_values("topology_idx")
        .reset_index(drop=True)
    )
    conformers = (
        pd.read_parquet(p / "conformers.parquet")
        .sort_values("conf_idx")
        .reset_index(drop=True)
    )
    constraints = pd.read_parquet(p / "constraints.parquet")
    params = pd.read_parquet(p / "params.parquet")
    assignments = pd.read_parquet(p / "assignments.parquet")
    values = pd.read_parquet(p / "values.parquet")
    unassigned = pd.read_parquet(p / "unassigned.parquet")
    exclusions = pd.read_parquet(p / "exclusions.parquet")
    parity = json.loads((p / "parity_input.json").read_text())
    tag_index = {s["smirks"]: s["tag_index"] for s in parity["smirks"]}

    assert (topologies.topology_idx.to_numpy() == np.arange(len(topologies))).all()
    assert (conformers.conf_idx.to_numpy() == np.arange(len(conformers))).all()
    # Conformers of a topology must be contiguous: shards rely on it
    starts = conformers.groupby("topology_idx").conf_idx.agg(["min", "max", "count"])
    assert ((starts["max"] - starts["min"] + 1) == starts["count"]).all()

    # ---------------- coordinates (opt and TD files; each topology contiguous) ----------------
    topologies["conf_start"] = starts["min"].reindex(topologies.topology_idx).to_numpy()
    topologies["n_conf"] = starts["count"].reindex(topologies.topology_idx).to_numpy()
    coord_offset = np.zeros(len(topologies), dtype=np.int64)
    for source, fname in [("opt", "coords_opt.bin"), ("td", "coords_td.bin")]:
        chunks, offset = [], 0
        for t in topologies[topologies.source == source].itertuples():
            coord_offset[t.topology_idx] = offset
            confs = conformers.iloc[t.conf_start : t.conf_start + t.n_conf]
            for c in confs.coords_angstrom:
                chunks.append(np.asarray(c, dtype=np.float32))
            offset += t.n_conf * t.n_atoms
        arr = np.concatenate(chunks).astype("<f4")
        assert len(arr) == offset * 3
        arr.tofile(out / fname)
    topologies["coord_offset"] = coord_offset

    # ---------------- specs / conformers ----------------
    specs = sorted(conformers.spec.unique())
    spec_idx = {s: i for i, s in enumerate(specs)}
    grid = conformers.grid_key.map(lambda k: None if pd.isna(k) else json.loads(k)[0])

    # QM energies of optimization records. For molecules with several optimization records (conformers),
    # the energy relative to the lowest of them, only when all share the same QC level (program, method,
    # basis); otherwise null and flagged. HARTREE_TO_KCAL as in 03_extract.py.
    HARTREE_TO_KCAL = 627.509474
    source_by_top = topologies.set_index("topology_idx").source
    mol_by_top = topologies.set_index("topology_idx").mol_idx
    qc_level = conformers.spec.map(lambda sp: tuple(json.loads(sp)[k] for k in ("qc_program", "method", "basis")))
    is_opt = conformers.topology_idx.map(source_by_top).eq("opt")
    opt_rel = pd.Series(np.nan, index=conformers.index)
    opt_same_level = pd.Series(pd.NA, index=conformers.index, dtype="boolean")
    opt_rows = conformers[is_opt].assign(mol_idx=lambda d: d.topology_idx.map(mol_by_top), level=qc_level[is_opt])
    for _, g in opt_rows.groupby("mol_idx"):
        if len(g) < 2:
            continue
        same = g.level.nunique() == 1
        opt_same_level[g.index] = same
        if same:
            opt_rel[g.index] = (g.energy_hartree - g.energy_hartree.min()) * HARTREE_TO_KCAL
    if not np.isfinite(conformers.loc[is_opt, "energy_hartree"]).all():
        raise SystemExit("non-finite QM energy for an optimization record")

    write_json(
        out / "conformers.json",
        {
            "energy_hartree": [round(float(e), 10) for e in conformers.energy_hartree],
            "opt_group_rel_kcal": [None if pd.isna(e) else round(float(e), 6) for e in opt_rel],
            "opt_group_same_level": [None if pd.isna(x) else bool(x) for x in opt_same_level],
            "record_id": conformers.conf_record_id.astype(int).tolist(),
            "grid_deg": [None if pd.isna(g) else int(g) for g in grid],
            "rel_energy_kcal": [
                None if pd.isna(e) else round(float(e), 6)
                for e in conformers.rel_energy_kcal
            ],
            "spec": conformers.spec.map(spec_idx).tolist(),
            "connectivity_changed": conformers.connectivity_changed.fillna(False)
            .astype(int)
            .tolist(),
        },
    )

    # ---------------- topologies ----------------
    cons: dict[tuple[int, str], list[list[int]]] = {}
    for r in constraints.itertuples():
        cons.setdefault((int(r.topology_idx), r.kind), []).append(
            [int(a) for a in r.atoms]
        )
    unas = (
        unassigned.groupby("topology_idx")
        .atoms.apply(lambda s: [list(map(int, a)) for a in s])
        .to_dict()
    )
    write_json(
        out / "topologies.json",
        {
            "source": topologies.source.tolist(),
            "entry_name": topologies.entry_name.tolist(),
            "record_id": topologies.record_id.astype(int).tolist(),
            "mol_idx": topologies.mol_idx.astype(int).tolist(),
            "n_atoms": topologies.n_atoms.astype(int).tolist(),
            "mdl_smiles": topologies.mdl_smiles.tolist(),
            "mdl_order": [json.loads(o) for o in topologies.mdl_order],
            "conf_start": topologies.conf_start.astype(int).tolist(),
            "n_conf": topologies.n_conf.astype(int).tolist(),
            "coord_offset": topologies.coord_offset.astype(int).tolist(),
            "driven": [
                (cons.get((i, "driven")) or [None])[0] for i in topologies.topology_idx
            ],
            "frozen": [cons.get((i, "frozen"), []) for i in topologies.topology_idx],
            "unassigned_propers": [unas.get(i, []) for i in topologies.topology_idx],
        },
    )

    # ---------------- molecules + fingerprints ----------------
    tops_by_mol = (
        topologies.groupby("mol_idx")
        .topology_idx.apply(lambda s: sorted(map(int, s)))
        .to_dict()
    )
    write_json(
        out / "molecules.json",
        {
            "display_smiles": molecules.display_smiles.tolist(),
            "charge": molecules.charge.astype(int).tolist(),
            "topologies": [tops_by_mol[i] for i in molecules.mol_idx],
        },
    )
    shutil.copy(p / "fp_morgan.bin", out / "fp_morgan.bin")

    # ---------------- per-parameter shards + stats ----------------
    driven_bonds = {
        (t, *sorted(a[1:3]))
        for (t, k), lst in cons.items()
        if k == "driven"
        for a in lst
    }
    frozen_bonds = {
        (t, *sorted(a[1:3]))
        for (t, k), lst in cons.items()
        if k == "frozen"
        for a in lst
    }
    driven_exact = {
        (t, *a) for (t, k), lst in cons.items() if k == "driven" for a in lst
    }
    driven_exact |= {
        (t, *reversed(a)) for (t, k), lst in cons.items() if k == "driven" for a in lst
    }

    source_of = topologies.source.to_numpy()
    n_conf_of = topologies.n_conf.to_numpy()
    values = values.sort_values(["asg_idx", "conf_idx"]).reset_index(drop=True)
    values_by_param = dict(
        tuple(
            values.merge(assignments[["asg_idx", "param_id"]], on="asg_idx").groupby(
                "param_id"
            )
        )
    )
    asg_by_param = dict(tuple(assignments.groupby("param_id")))

    param_rows = []
    for r in params.itertuples():
        row = {
            k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in r._asdict().items()
            if k != "Index"
        }
        for key in ("periodicity", "phase_deg", "k_raw", "idivf", "k_effective"):
            if isinstance(row.get(key), str):
                row[key] = json.loads(row[key])
        row["tag_index"] = tag_index[r.smirks]
        asg = asg_by_param.get(r.param_id)
        if asg is None:
            row |= {
                "n_molecules": 0,
                "n_topologies": 0,
                "n_assignments": 0,
                "n_obs_opt": 0,
                "n_obs_td": 0,
                "shard": None,
            }
            param_rows.append(row)
            continue
        asg = asg.sort_values("asg_idx")
        vals = values_by_param[r.param_id]
        t_idx = asg.topology_idx.to_numpy()
        is_torsion = r.handler in ("ProperTorsions", "ImproperTorsions")
        # flags per assignment
        flags = np.zeros(len(asg), dtype=np.uint8)
        if r.handler == "ProperTorsions":
            for i, (t, a) in enumerate(zip(t_idx, asg.atoms)):
                key = (int(t), *sorted((int(a[1]), int(a[2]))))
                flags[i] |= FLAG_ON_DRIVEN_BOND * (key in driven_bonds)
                flags[i] |= FLAG_ON_FROZEN_BOND * (key in frozen_bonds)
                flags[i] |= FLAG_IS_DRIVEN_TORSION * (
                    (int(t), *map(int, a)) in driven_exact
                )
        # rows: for each asg (in asg_idx order) the topology's conformers in conf order
        n_rows = int(n_conf_of[t_idx].sum())
        assert len(vals) == n_rows, (r.param_id, len(vals), n_rows)
        atoms = np.full((len(asg), 4), -1, dtype="<i2")
        for i, a in enumerate(asg.atoms):
            atoms[i, : len(a)] = a
        if r.handler == "ImproperTorsions":
            width = 3
            v = np.stack(vals.improper_terms_deg.to_numpy()).astype("<f4")
        else:
            width = 1
            v = vals.value.to_numpy().astype("<f4")
        valid = vals.valid.to_numpy().astype(np.uint8)

        buf = bytearray(struct.pack("<4I", SCHEMA_VERSION, len(asg), n_rows, width))
        buf += t_idx.astype("<i4").tobytes()
        buf += atoms.tobytes()
        buf += flags.tobytes()
        pad4(buf)
        buf += v.tobytes()
        if is_torsion:
            buf += valid.tobytes()
        shard = f"shards/{r.param_id}.bin"
        (out / shard).write_bytes(bytes(buf))

        # summary statistics: optimization geometries only (unconstrained minima; frozen-bond rows excluded)
        row_src = np.repeat(source_of[t_idx], n_conf_of[t_idx])
        row_flags = np.repeat(flags, n_conf_of[t_idx])
        opt_mask = (row_src == "opt") & ((row_flags & FLAG_ON_FROZEN_BOND) == 0)
        # Improper summaries use all three trefoil terms, each weighted 1/3 (as the detail view does)
        vv = v.reshape(-1) if width == 3 else v
        ww = np.full(len(vv), 1.0 / width)
        mask = np.repeat(opt_mask if not is_torsion else opt_mask & (valid == 1), width)
        row |= {
            "n_molecules": int(topologies.mol_idx.iloc[np.unique(t_idx)].nunique()),
            "n_topologies": int(len(np.unique(t_idx))),
            "n_assignments": int(len(asg)),
            "n_obs_opt": int((row_src == "opt").sum()),
            "n_obs_td": int((row_src == "td").sum()),
            "n_invalid": int((valid == 0).sum()) if is_torsion else 0,
            "shard": shard,
            "shard_bytes": len(buf),
        }
        if is_torsion:
            row["opt_stats"] = circular_stats(vv[mask].astype(float), ww[mask])
        else:
            stats = linear_stats(vv[mask].astype(float), ww[mask])
            centre = (
                row["length_angstrom"] if r.handler == "Bonds" else row["angle_deg"]
            )
            stats["mean_minus_centre"] = (
                None if stats["mean"] is None else stats["mean"] - centre
            )
            row["opt_stats"] = stats
        param_rows.append(row)

    roster = {(r.handler, r.param_id) for r in params.itertuples()}
    assert roster == {
        (r["handler"], r["param_id"]) for r in param_rows
    }, "parameter roster not closed-world"
    write_json(out / "params.json", param_rows)

    # ---------------- per-topology assignment index ----------------
    # header <3I (schema, n_topologies, n_rows); Int32 row_start[n_topologies + 1];
    # Uint16 param_index[n_rows] (index into params.json), padded to 4; Int16 atoms[n_rows * 4] (-1 padded)
    param_index = {pid: i for i, pid in enumerate(r["param_id"] for r in param_rows)}
    a = assignments.sort_values(["topology_idx", "asg_idx"])
    counts = a.groupby("topology_idx").size().reindex(topologies.topology_idx, fill_value=0).to_numpy()
    row_start = np.concatenate([[0], np.cumsum(counts)]).astype("<i4")
    atoms = np.full((len(a), 4), -1, dtype="<i2")
    for i, at in enumerate(a.atoms):
        atoms[i, : len(at)] = at
    buf = bytearray(struct.pack("<3I", SCHEMA_VERSION, len(topologies), len(a)))
    buf += row_start.tobytes()
    buf += a.param_id.map(param_index).to_numpy().astype("<u2").tobytes()
    pad4(buf)
    buf += atoms.tobytes()
    (out / "assignments.bin").write_bytes(bytes(buf))

    # ---------------- meta ----------------
    download = json.loads((args.raw / "download_manifest.json").read_text())
    td_manifest = json.loads((args.raw / "td_qcarchive_manifest.json").read_text())
    fp_meta = json.loads((p / "fp_morgan.json").read_text())
    meta = {
        "schema_version": SCHEMA_VERSION,
        "sources": {
            "optimization": {
                "zenodo_record": download["zenodo"]["record"],
                "doi": download["zenodo"]["doi"],
                "title": download["zenodo"]["title"],
                "license": download["zenodo"]["license"],
                "file_sha256": {
                    k: v["sha256"] for k, v in download["zenodo"]["files"].items()
                },
            },
            "torsiondrive": {
                k: td_manifest[k]
                for k in (
                    "server",
                    "dataset_type",
                    "dataset_name",
                    "dataset_id",
                    "qcportal_version",
                    "fetched_utc",
                )
            },
            "force_field": download["offxml"],
            "rdkit_js": {
                "version": fp_meta["rdkit_js_version"],
                "sha256": fp_meta["vendored_sha256"],
            },
        },
        "counts": {
            "molecules": len(molecules),
            "topologies": topologies.source.value_counts().to_dict(),
            "conformers": {
                "opt": int((source_of[conformers.topology_idx] == "opt").sum()),
                "td": int((source_of[conformers.topology_idx] == "td").sum()),
            },
            "assignments": assignments.handler.value_counts().to_dict(),
            "exclusions": len(exclusions),
        },
        "specs": [json.loads(s) for s in specs],
        "exclusions": exclusions.to_dict(orient="records"),
        "validation": {
            name: json.loads((p / f"{name}.json").read_text())
            for name in (
                "extract_summary",
                "label_summary",
                "geometry_summary",
                "energy_validation",
                "parity_summary",
            )
        },
        "flags": {
            "on_driven_bond": FLAG_ON_DRIVEN_BOND,
            "on_frozen_bond": FLAG_ON_FROZEN_BOND,
            "is_driven_torsion": FLAG_IS_DRIVEN_TORSION,
        },
    }

    # ---------------- sizes, hashes, gates ----------------
    def file_info(path: pathlib.Path) -> dict:
        data = path.read_bytes()
        return {
            "bytes": len(data),
            "gzip_bytes": len(gzip.compress(data, 6)),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    files = {
        str(f.relative_to(out)): file_info(f)
        for f in sorted(out.rglob("*"))
        if f.is_file() and f.name not in KEEP
    }
    write_json(out / "meta.json", meta)  # provisional, to measure its own size
    files["meta.json"] = file_info(out / "meta.json")
    total = sum(f["gzip_bytes"] for f in files.values()) / 1e6
    initial = sum(files[f]["gzip_bytes"] for f in INITIAL_FILES) / 1e6
    largest = max(files.items(), key=lambda kv: kv[1]["gzip_bytes"])
    gates = {
        "total_compressed_mb": round(total, 2),
        "initial_compressed_mb": round(initial, 2),
        "largest_file": largest[0],
        "largest_file_compressed_mb": round(largest[1]["gzip_bytes"] / 1e6, 2),
        "limits_mb": GATES_MB,
    }
    gates["pass"] = (
        total <= GATES_MB["total_compressed"]
        and initial <= GATES_MB["initial_compressed"]
        and largest[1]["gzip_bytes"] / 1e6 <= GATES_MB["file_compressed"]
    )
    meta["files"] = {k: v for k, v in files.items() if k != "meta.json"}
    meta["size_gates"] = gates
    write_json(out / "meta.json", meta)
    print(json.dumps(gates, indent=1))
    print(
        f"{len(files)} files; uncompressed {sum(f['bytes'] for f in files.values()) / 1e6:.1f} MB"
    )
    if not gates["pass"]:
        raise SystemExit(f"size gate FAILED; previous output in {final} left unchanged (new build in {out})")
    # Swap: previous output -> .old, staging -> final, then remove .old
    old_dir = final.parent / (final.name + ".old")
    if old_dir.exists():
        shutil.rmtree(old_dir)
    if final.exists():
        final.rename(old_dir)
    out.rename(final)
    if old_dir.exists():
        shutil.rmtree(old_dir)


if __name__ == "__main__":
    main()
