import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Descriptors

rdBase.DisableLog("rdApp.warning")

N_BITS = 256

_MORGAN_GEN = AllChem.GetMorganGenerator(radius=2, fpSize=N_BITS)


def mol_to_fingerprint(mol: Chem.Mol) -> np.ndarray:
    """Convert an RDKit molecule to a Morgan fingerprint bit vector.

    Args:
        mol: RDKit Mol object.

    Returns:
        NumPy array of shape (N_BITS,) with binary fingerprint bits.
    """
    return _MORGAN_GEN.GetFingerprintAsNumPy(mol).astype(np.int8)


def mol_to_props(mol: Chem.Mol) -> np.ndarray:
    """Compute 10 molecular descriptors for a molecule.

    Args:
        mol: RDKit Mol object.

    Returns:
        NumPy array of 10 descriptor values: MolWt, LogP, H-donors,
        H-acceptors, TPSA, rotatable bonds, aromatic rings,
        aliphatic rings, FractionCSP3, and heteroatom count.
    """
    return np.array(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.NumHeteroatoms(mol),
        ]
    )


def build_features(df: pd.DataFrame) -> np.ndarray:
    """Build a 533-dimensional feature matrix from SMILES pairs.

    Features include Morgan fingerprint differences/products,
    Tanimoto similarity, and 10 molecular descriptor sums/differences.

    Args:
        df: DataFrame with 'smiles_a' and 'smiles_b' columns.

    Returns:
        Feature matrix of shape (n_samples, 2 * N_BITS + 1 + 2 * 10).
    """
    mols_a = [Chem.MolFromSmiles(s) for s in df["smiles_a"]]
    mols_b = [Chem.MolFromSmiles(s) for s in df["smiles_b"]]

    fp_a = np.array([mol_to_fingerprint(m) for m in mols_a])
    fp_b = np.array([mol_to_fingerprint(m) for m in mols_b])
    diff = np.abs(fp_a - fp_b)
    product = fp_a * fp_b
    intersection = product.sum(axis=1)
    union = fp_a.sum(axis=1) + fp_b.sum(axis=1) - intersection
    sim = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=np.float64), where=union != 0)

    props_a = np.array([mol_to_props(m) for m in mols_a])
    props_b = np.array([mol_to_props(m) for m in mols_b])
    prop_diff = np.abs(props_a - props_b)
    prop_sum = props_a + props_b

    X = np.column_stack(
        [
            np.hstack([diff, product]),
            sim,
            prop_diff,
            prop_sum,
        ]
    )
    return X
