"""Tests for feature engineering functions."""

import numpy as np
import pandas as pd
from rdkit import Chem

from src.features import N_BITS, build_features, mol_to_props


def _n_props() -> int:
    return len(mol_to_props(Chem.MolFromSmiles("O")))


def test_build_features_output_shape():
    dummy = pd.DataFrame(
        {
            "smiles_a": ["O", "CCO"],
            "smiles_b": ["CCO", "O"],
        }
    )
    X = build_features(dummy)
    expected = 4 * N_BITS + 1 + 2 * _n_props()
    assert X.shape == (2, expected)


def test_build_features_returns_float_array():
    dummy = pd.DataFrame(
        {
            "smiles_a": ["O"],
            "smiles_b": ["CCO"],
        }
    )
    X = build_features(dummy)
    assert X.dtype == np.float64


def test_build_features_no_nan():
    dummy = pd.DataFrame(
        {
            "smiles_a": ["O", "CCO", "c1ccccc1"],
            "smiles_b": ["CCO", "c1ccccc1", "O"],
        }
    )
    X = build_features(dummy)
    assert not np.any(np.isnan(X))


def test_build_features_single_row():
    dummy = pd.DataFrame(
        {
            "smiles_a": ["O"],
            "smiles_b": ["CCO"],
        }
    )
    X = build_features(dummy)
    expected = 4 * N_BITS + 1 + 2 * _n_props()
    assert X.shape == (1, expected)


def test_build_features_tanimoto_range():
    dummy = pd.DataFrame(
        {
            "smiles_a": ["O", "O"],
            "smiles_b": ["CCO", "O"],
        }
    )
    X = build_features(dummy)
    sim_col = 4 * N_BITS
    assert 0.0 <= X[0, sim_col] <= 1.0
    assert X[1, sim_col] == 1.0
