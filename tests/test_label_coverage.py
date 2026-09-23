"""The labeling coverage gate: every topology labeled, and only the declared unassigned interactions."""

import importlib.util
import pathlib

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("label", pathlib.Path(__file__).parents[1] / "scripts" / "04_label.py")
label = importlib.util.module_from_spec(spec)
spec.loader.exec_module(label)

TOPS = pd.DataFrame({"topology_idx": [0, 1], "source": ["opt", "opt"], "record_id": [146497953, 7]})
ASG = pd.DataFrame({"topology_idx": [0, 1]})
EXPECTED = pd.DataFrame(
    {
        "topology_idx": [0, 0, 0, 0],
        "handler": ["ProperTorsions"] * 4,
        "atoms": [[2, 3, 5, 6], [2, 3, 5, 8], [4, 3, 5, 6], [4, 3, 5, 8]],
    }
)


def test_declared_gaps_pass():
    assert label.check_coverage(TOPS, ASG, EXPECTED) == []


def test_unlabeled_topology_fails():
    assert label.check_coverage(TOPS, ASG[ASG.topology_idx == 0], EXPECTED)


@pytest.mark.parametrize(
    "unassigned",
    [
        pd.concat([EXPECTED, pd.DataFrame({"topology_idx": [1], "handler": ["Angles"], "atoms": [[0, 1, 2]]})]),  # new gap
        EXPECTED.iloc[:3],  # a declared gap disappeared
    ],
)
def test_unexpected_gaps_fail(unassigned):
    assert label.check_coverage(TOPS, ASG, unassigned)
