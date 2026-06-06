"""Integration tests: drift detection with real feature distributions."""

import json

import numpy as np
import pandas as pd

from src.drift import (
    compute_reference_stats,
    detect_drift,
    fingerprint_density,
    save_report,
)
from src.features import build_features


def _make_drug_pairs(smiles_list_a, smiles_list_b, label=0):
    n = len(smiles_list_a)
    return pd.DataFrame(
        {
            "smiles_a": smiles_list_a,
            "smiles_b": smiles_list_b[:n],
            "severity_label": [label] * n,
        }
    )


# Verified valid SMILES strings — small molecules
_SMALL = [
    "O",
    "CCO",
    "CC",
    "CO",
    "C1CC1",
    "C=C",
    "CCN",
    "CCOCC",
    "CCCO",
    "CCC",
    "CCCC",
    "CC(C)C",
    "C1CO1",
    "CC=O",
    "CC#N",
    "CCS",
    "C1CCCCC1",
    "CCOC",
    "CCCl",
    "C1CCOC1",
    "CC(C)=O",
    "CCN(C)C",
    "CCCOC",
    "CCCCO",
    "CCCCN",
    "C1CCOCC1",
    "CCOCCN",
    "CCCCCN",
    "C1CCCC1",
    "C1COCCO1",
    "CCOCCO",
    "CC1CCCC1",
    "CCl",
    "CBr",
    "CC(C)(C)C",
    "CC(=O)O",
    "CN",
    "O=C=O",
    "C#N",
    "CC(C)O",
    "CC(C)CCO",
    "CC(=O)N",
    "CC(=O)C",
    "CCBr",
    "CCCCCl",
    "CCOP",
    "CCSCC",
    "C1CC1C",
    "CC1CO1",
    "CC(C)=C",
    "C1=CCCC1",
    "C1=CCCCC1",
    "CCOCCCO",
    "CCCCC",
    "CC#CC",
    "CC(=O)CBr",
    "COCCO",
    "C1CCCCC1C",
    "CCC(C)=O",
    "CCCCCO",
    "CCCCCN",
    "CCCCCCN",
    "CCCCCCCO",
    "C1CCCCCC1",
]

# Verified valid SMILES strings — large molecules
_LARGE = [
    "c1ccc2ccccc2c1",
    "c1ccc3cc2ccccc2cc3c1",
    "c1ccc4cc3cc2ccccc2cc3cc4c1",
    "C1CCC2CCCCC2C1",
    "C1CC3CC2CCCC2C3C1",
    "CCCCCCCCCCCCCCCC",
    "CCCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCCCCC",
    "CCCCCCCCCCCCCCCCCCCCCC",
    "CCCCCCCCCCCCCCCCCCCCCCCC",
    "CC1=C(C)C2CCC(C2)C1(C)C",
    "CC1(C)CCC2C3CCCC3CCC12",
    "C1=CC2C3CCCC3C4C2C1C4",
    "CC1(C)C2CCC(C2)C1C3CCC4CCCCC34",
    "CC12CCC3C4CCCC4CCC1C2C3",
    "CC1(C2)CC3CC2CC(C3)C1",
    "C1CC2C3CCCC3C4C2C1C4",
    "CC1(C2)C3CCC2C(C1)C3",
    "CC1(C2)CCC3C4C2C1CC3C4",
    "C1CC2C3CC4C2C1C3C4",
    "CC1(C2)C3CC4C2C1C3C4",
    "CC12CC3CC(C1)CC(C3)C2",
    "c1ccc(cc1)c2ccccc2",
    "c1ccc2c(c1)ccc3c2cccc3",
    "CCCCCCCCCCCCCCCCCC",
    "C1CC2CC(C1)C3C2C4C3C5C4C5",
    "CC1(C2)C3C4C2C1C5C3C4C5",
    "CC1(C2)C3C4C2C1C3C4",
    "C1C2C3C4C1C5C2C3C4C5",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    "C1CC2C3C4C1C2C3C4",
    "CC1(C2)CCC3C4C2C1CC3C4",
    "CC1(C2)C3CC4C2C1C5C3C4C5",
    "CC12CC3CC4CC(C3C1)C2C4",
    "CCCCCCCCCCCCCCCCCCCCCCCC",
    "C1CCCC2CCCCC2C1",
    "C1CC3C2CCCC2C3C1",
    "C1CCC2C3CCCC3C4C2C1C4",
    "C1CC2CC3CCCC3C2C1",
    "CC1(C2)CC3CC2CC4C3C4C1",
    "CC1(C2)CC3C4C2C1C3C4",
    "C1CC2CC3C4C2C1C3C4",
    "CC12CC3CC(C1)CC4CC3C4C2",
]


class TestDriftIntegration:
    """Integration: real features → drift detection."""

    def test_different_molecular_sizes_produce_different_densities(self):
        small_df = _make_drug_pairs(_SMALL[:3], _SMALL[3:])
        large_df = _make_drug_pairs(_LARGE[:3], _LARGE[3:])

        X_small = build_features(small_df)
        X_large = build_features(large_df)

        dens_small = fingerprint_density(X_small)
        dens_large = fingerprint_density(X_large)

        assert dens_small.shape == (3,)
        assert dens_large.shape == (3,)
        assert not np.allclose(dens_small, dens_large, atol=0.1)

    def test_detect_drift_returns_expected_keys(self):
        df = _make_drug_pairs(_SMALL[:3], _SMALL[3:])
        X = build_features(df)

        ref = compute_reference_stats(X)
        assert "fp_density_mean" in ref
        assert "fp_density_std" in ref
        assert "fp_density_p5" in ref
        assert "fp_density_p95" in ref
        assert "n_samples" in ref

    def test_no_drift_on_same_distribution(self):
        df = _make_drug_pairs(_SMALL[:3], _SMALL[3:])
        X_ref = build_features(df)
        X_new = build_features(df)

        ref = compute_reference_stats(X_ref)
        result = detect_drift(X_new, ref, p_threshold=0.05)

        assert not result["drift_detected"]

    def test_drift_detected_on_different_distribution(self):
        small_df = _make_drug_pairs(_SMALL[:21], _SMALL[21:42])
        large_df = _make_drug_pairs(_LARGE[:21], _LARGE[21:42])

        X_ref = build_features(small_df)
        X_new = build_features(large_df)

        ref = compute_reference_stats(X_ref)
        result = detect_drift(X_new, ref, p_threshold=0.05)

        assert result["drift_detected"]

    def test_save_report_writes_valid_json(self, tmp_path, monkeypatch):
        report = {
            "drift_detected": True,
            "ks_statistic": 0.75,
            "p_value": 0.001,
            "mean_density_ref": 0.3,
            "mean_density_new": 0.6,
        }
        report_path = tmp_path / "drift_report.json"
        monkeypatch.setattr("src.drift.DRIFT_REPORT", str(report_path))
        save_report(report)

        with open(report_path) as f:
            loaded = json.load(f)

        assert loaded["drift_detected"] is True
        assert loaded["ks_statistic"] == 0.75

    def test_fingerprint_density_ranges(self):
        df = _make_drug_pairs(_SMALL[:3], _SMALL[3:])
        X = build_features(df)
        densities = fingerprint_density(X)

        assert np.all(densities >= 0.0)
        assert np.all(densities <= 1.0)
