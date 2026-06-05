import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors, AllChem, Descriptors

N_BITS = 256


def mol_to_fingerprint(mol) -> np.ndarray:
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=N_BITS)
    return np.array(fp)


def mol_to_props(mol) -> np.ndarray:
    return np.array([
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
    ])


def tanimoto_similarity(mol_a, mol_b) -> float:
    fpa = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius=2, nBits=N_BITS)
    fpb = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius=2, nBits=N_BITS)
    return float(DataStructs.TanimotoSimilarity(fpa, fpb))


def build_features(df: pd.DataFrame) -> np.ndarray:
    mols_a = [Chem.MolFromSmiles(s) for s in df["smiles_a"]]
    mols_b = [Chem.MolFromSmiles(s) for s in df["smiles_b"]]

    fp_a = np.array([mol_to_fingerprint(m) for m in mols_a])
    fp_b = np.array([mol_to_fingerprint(m) for m in mols_b])
    diff = np.abs(fp_a - fp_b)
    product = fp_a * fp_b
    sim = np.array([tanimoto_similarity(ma, mb) for ma, mb in zip(mols_a, mols_b)])

    props_a = np.array([mol_to_props(m) for m in mols_a])
    props_b = np.array([mol_to_props(m) for m in mols_b])
    prop_diff = np.abs(props_a - props_b)
    prop_sum = props_a + props_b

    X = np.column_stack([
        np.hstack([fp_a, fp_b, diff, product]),
        sim,
        prop_diff,
        prop_sum,
    ])
    return X
