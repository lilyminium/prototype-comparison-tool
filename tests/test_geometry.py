import importlib.util
import pathlib

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

SCRIPTS = pathlib.Path(__file__).parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("extract", SCRIPTS / "03_extract.py")
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)


@pytest.fixture(scope="module")
def butane():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    AllChem.EmbedMultipleConfs(mol, numConfs=5, randomSeed=42)
    return mol


@pytest.mark.parametrize("target", [-179.0, -120.0, -60.0, 0.0, 60.0, 120.0, 179.0])
def test_dihedral_sign_matches_rdkit(butane, target):
    """Signed dihedral follows the IUPAC convention, as RDKit's GetDihedralDeg does."""
    conf = butane.GetConformer(0)
    rdMolTransforms.SetDihedralDeg(conf, 0, 1, 2, 3, target)
    xyz = np.array(conf.GetPositions())
    assert extract.dihedral_deg(xyz, 0, 1, 2, 3) == pytest.approx(target, abs=1e-6)
    assert extract.dihedral_deg(xyz, 0, 1, 2, 3) == pytest.approx(
        rdMolTransforms.GetDihedralDeg(conf, 0, 1, 2, 3), abs=1e-6
    )


def test_dihedral_reversal_invariant(butane):
    for conf in butane.GetConformers():
        xyz = np.array(conf.GetPositions())
        assert extract.dihedral_deg(xyz, 0, 1, 2, 3) == pytest.approx(
            extract.dihedral_deg(xyz, 3, 2, 1, 0), abs=1e-9
        )


@pytest.mark.parametrize(
    "value,expected", [(190.0, -170.0), (-190.0, 170.0), (180.0, -180.0), (0.0, 0.0)]
)
def test_wrap(value, expected):
    assert extract.wrap(value) == pytest.approx(expected)
