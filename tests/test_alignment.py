"""Acceptance tests for the precomputed conformer superposition (PLAN_precompute.md S3)."""

import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

ROOT = pathlib.Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "align", ROOT / "scripts" / "08d_align_conformers.py"
)
align = importlib.util.module_from_spec(spec)
spec.loader.exec_module(align)
RNG = np.random.default_rng(7)


def _cloud(n=12):
    return RNG.normal(size=(n, 3)) * 2.0


def test_proper_rotation_and_recovery():
    x = _cloud()
    rot_true = Rotation.random(random_state=3).as_matrix()
    y = x @ rot_true.T + np.array([4.0, -2.0, 7.0])
    rot, trans = align.kabsch(y, x)
    assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-9)
    assert np.sqrt(((y @ rot.T + trans - x) ** 2).sum(axis=1).mean()) < 1e-6


def test_mirror_image_is_not_superposed_by_reflection():
    x = _cloud()
    mirror = x * np.array([1.0, 1.0, -1.0])
    rot, trans = align.kabsch(mirror, x)
    assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-9)
    rmsd = np.sqrt(((mirror @ rot.T + trans - x) ** 2).sum(axis=1).mean())
    assert rmsd > 0.1  # a chiral cloud cannot be matched by a proper rotation


def test_identity_is_not_accepted_as_alignment():
    x = _cloud()
    y = x @ Rotation.from_euler("xyz", [40, 10, 70], degrees=True).as_matrix().T
    rot, _ = align.kabsch(y, x)
    assert not np.allclose(rot, np.eye(3))


@pytest.mark.skipif(
    not (ROOT / "data/processed/conformer_alignment.parquet").exists(),
    reason="run 08d first",
)
def test_every_shipped_pair_matches_independent_oracle_and_is_rigid():
    df = pd.read_parquet(ROOT / "data/processed/conformer_alignment.parquet")
    tops = pd.read_parquet(ROOT / "data/processed/topologies.parquet").set_index(
        "topology_idx"
    )
    confs = pd.read_parquet(
        ROOT / "data/processed/conformers.parquet",
        columns=["topology_idx", "coords_angstrom"],
    ).set_index("topology_idx")
    assert (
        len(df) == 2 * 45 + 6 * 3
    )  # 45 molecules with 2 records, 3 with 3 records (ordered pairs)
    for r in df.itertuples():
        orig_u = (
            np.asarray(confs.at[r.topology, "coords_angstrom"], dtype=np.float32)
            .reshape(-1, 3)
            .astype(float)
        )
        ref = (
            np.asarray(confs.at[r.ref_topology, "coords_angstrom"], dtype=np.float32)
            .reshape(-1, 3)
            .astype(float)
        )
        shipped = np.asarray(r.coords, dtype=float).reshape(-1, 3)
        # rigid: all interatomic distances preserved
        d0 = np.linalg.norm(orig_u[:, None] - orig_u[None], axis=-1)
        d1 = np.linalg.norm(shipped[:, None] - shipped[None], axis=-1)
        assert np.abs(d0 - d1).max() < 1e-4
        # independent oracle on the same fitted atoms: scipy's optimal proper rotation gives the same RMSD
        ou, orf = np.asarray(json.loads(tops.at[r.topology, "mdl_order"])), np.asarray(
            json.loads(tops.at[r.ref_topology, "mdl_order"])
        )
        heavy = np.array([sym != "H" for sym in _symbols(tops.at[r.ref_topology, "mapped_smiles"])])[orf]
        fit = align.fit_indices(heavy, ref[orf])
        a, b = orig_u[ou][fit], ref[orf][fit]
        rot, _ = Rotation.align_vectors(b - b.mean(0), a - a.mean(0))
        oracle = np.sqrt(
            (((a - a.mean(0)) @ rot.as_matrix().T - (b - b.mean(0))) ** 2)
            .sum(axis=1)
            .mean()
        )
        assert r.rmsd == pytest.approx(oracle, abs=1e-5)
        # reported RMSD equals the RMSD of exactly the shipped coordinates
        shipped_fit = shipped[ou][fit]
        assert r.rmsd == pytest.approx(
            np.sqrt(((shipped_fit - b) ** 2).sum(axis=1).mean()), abs=1e-5
        )


def _symbols(mapped_smiles):
    from openff.toolkit import Molecule

    return [
        a.symbol
        for a in Molecule.from_mapped_smiles(
            mapped_smiles, allow_undefined_stereo=True
        ).atoms
    ]
