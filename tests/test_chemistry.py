"""Tests for RDKit fingerprint generation."""

import sys

sys.path.insert(0, "src")

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def smiles_to_fingerprint(smiles: str) -> np.ndarray:
    """Convert a SMILES string to a 1024-bit Morgan fingerprint.

    Args:
        smiles: SMILES string of a molecule.

    Returns:
        Binary fingerprint array of shape (1024,).
    """
    mol = Chem.MolFromSmiles(smiles)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    return np.array(fp)


def test_water_fingerprint() -> None:
    """Verify water's Morgan fingerprint has correct shape and binary values."""
    fp = smiles_to_fingerprint("O")
    assert fp.shape == (1024,)
    assert fp.dtype == np.int8 or fp.dtype == np.int32 or fp.dtype == np.int64
    assert set(fp).issubset({0, 1})


def test_aspirin_fingerprint() -> None:
    """Verify aspirin's Morgan fingerprint has correct shape and binary values."""
    fp = smiles_to_fingerprint("CC(=O)Oc1ccccc1C(=O)O")
    assert fp.shape == (1024,)
    assert set(fp).issubset({0, 1})
