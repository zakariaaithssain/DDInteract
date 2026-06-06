"""Tests for feature engineering output dimensions."""

import numpy as np
import pandas as pd
from rdkit import Chem

from src.features import N_BITS, mol_to_fingerprint, mol_to_props, tanimoto_similarity


def test_interaction_feature_dimensions() -> None:
    """Verify that the interaction feature matrix has the expected column count."""
    dummy = pd.DataFrame(
        {
            "smiles_a": ["O", "CCO"],
            "smiles_b": ["CCO", "O"],
        }
    )
    mols_a = [Chem.MolFromSmiles(s) for s in dummy["smiles_a"]]
    mols_b = [Chem.MolFromSmiles(s) for s in dummy["smiles_b"]]

    fp_a = np.array([mol_to_fingerprint(m) for m in mols_a])
    fp_b = np.array([mol_to_fingerprint(m) for m in mols_b])
    diff = np.abs(fp_a - fp_b)
    product = fp_a * fp_b
    sim = np.array([tanimoto_similarity(ma, mb) for ma, mb in zip(mols_a, mols_b)])

    props_a = np.array([mol_to_props(m) for m in mols_a])
    props_b = np.array([mol_to_props(m) for m in mols_b])
    prop_diff = np.abs(props_a - props_b)
    prop_sum = props_a + props_b

    X = np.column_stack(
        [
            np.hstack([fp_a, fp_b, diff, product]),
            sim,
            prop_diff,
            prop_sum,
        ]
    )
    n_props = len(mol_to_props(mols_a[0]))
    expected = 4 * N_BITS + 1 + 2 * n_props
    assert X.shape[1] == expected, f"Expected {expected}, got {X.shape[1]}"
