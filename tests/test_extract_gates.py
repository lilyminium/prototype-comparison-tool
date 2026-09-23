"""Fault injection for the extraction gates: each corruption must be rejected on its own."""

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("extract", pathlib.Path(__file__).parents[1] / "scripts" / "03_extract.py")
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)

EXPECTED = {("opt", 1, None, 1), ("opt", 2, None, 2), ("td", 10, "[0]", 100), ("td", 10, "[15]", 101)}
GOOD = sorted(EXPECTED, key=str)


def test_reconcile_accepts_exact_keys():
    extract.reconcile(EXPECTED, GOOD)


@pytest.mark.parametrize(
    "produced",
    [
        GOOD[:-1],  # missing
        GOOD + [GOOD[0]],  # duplicated
        GOOD[:-1] + [("td", 10, "[15]", 999)],  # substituted child optimization, same count
        GOOD[:-1] + [GOOD[0]],  # duplicate replacing a missing one, same count
    ],
)
def test_reconcile_rejects(produced):
    with pytest.raises(RuntimeError):
        extract.reconcile(EXPECTED, produced)


def _conformers(dev, coords=(0.0, 0.0, 0.0)):
    return pd.DataFrame(
        {
            "topology_key": ["opt:1", "td:10"],
            "grid_key": [None, "[0]"],
            "grid_deviation_deg": [None, dev],
            "coords_angstrom": [list(coords), [0.0, 0.0, 0.0]],
        }
    )


def test_gates_pass():
    extract.check_gates(_conformers(0.01))


@pytest.mark.parametrize("dev", [0.5, np.nan, np.inf])
def test_grid_gate_rejects(dev):
    with pytest.raises(RuntimeError):
        extract.check_gates(_conformers(dev))


def test_coordinate_gate_rejects_nan():
    with pytest.raises(RuntimeError):
        extract.check_gates(_conformers(0.01, coords=(np.nan, 0.0, 0.0)))
