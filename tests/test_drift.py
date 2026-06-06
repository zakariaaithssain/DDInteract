"""Tests for data drift detection."""

import json
from pathlib import Path

import numpy as np

from src.drift import (
    DRIFT_REPORT,
    _extract_drift_features,
    compute_reference_stats,
    detect_drift,
    fingerprint_density,
    save_report,
)

N_BITS = 256
N_FEATURES = 4 * N_BITS + 1 + 20  # 1045


def _make_feature_matrix(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Build a synthetic full feature matrix (1045 columns) for testing."""
    if rng is None:
        rng = np.random.default_rng(42)
    X = np.zeros((n, N_FEATURES))
    # fingerprint block (first 4*N_BITS cols) — binary
    X[:, : 4 * N_BITS] = rng.integers(0, 2, size=(n, 4 * N_BITS)).astype(np.float64)
    # tanimoto (col 4*N_BITS) — [0, 1]
    X[:, 4 * N_BITS] = rng.uniform(0, 1, size=n)
    # descriptor block (last 20 cols) — continuous
    X[:, 4 * N_BITS + 1 :] = rng.normal(0, 1, size=(n, 20))
    return X


class TestFingerprintDensity:
    def test_output_shape(self):
        fp = np.zeros((10, 256))
        result = fingerprint_density(fp)
        assert result.shape == (10,)

    def test_all_ones_density(self):
        fp = np.ones((5, 256))
        result = fingerprint_density(fp)
        assert np.allclose(result, 1.0)

    def test_all_zeros_density(self):
        fp = np.zeros((5, 256))
        result = fingerprint_density(fp)
        assert np.allclose(result, 0.0)

    def test_half_ones_density(self):
        fp = np.zeros((3, 256))
        fp[:, :128] = 1.0
        result = fingerprint_density(fp)
        assert np.allclose(result, 0.5)


class TestExtractDriftFeatures:
    def test_output_columns(self):
        X = _make_feature_matrix(10)
        df = _extract_drift_features(X)
        expected = {"fp_a_density", "fp_b_density", "fp_diff_mean", "fp_product_mean", "tanimoto"}
        assert set(df.columns) == expected

    def test_output_shape(self):
        X = _make_feature_matrix(7)
        df = _extract_drift_features(X)
        assert df.shape == (7, 5)

    def test_fp_a_density_values(self):
        X = np.zeros((3, N_FEATURES))
        X[0, :256] = 1.0  # all bits on for first sample
        X[1, :128] = 1.0  # half bits on
        X[2, :64] = 1.0  # quarter bits on
        df = _extract_drift_features(X)
        assert np.allclose(df["fp_a_density"].values, [1.0, 0.5, 0.25])

    def test_tanimoto_column(self):
        X = _make_feature_matrix(5)
        df = _extract_drift_features(X)
        assert np.allclose(df["tanimoto"].values, X[:, 4 * N_BITS])


class TestComputeReferenceStats:
    def test_output_keys(self):
        X = _make_feature_matrix(100)
        stats = compute_reference_stats(X)
        assert set(stats.keys()) == {
            "fp_density_mean",
            "fp_density_std",
            "fp_density_p5",
            "fp_density_p95",
            "n_samples",
            "reference_features",
            "reference_densities",
        }

    def test_n_samples(self):
        X = _make_feature_matrix(50)
        stats = compute_reference_stats(X)
        assert stats["n_samples"] == 50

    def test_reference_features_is_dict_of_lists(self):
        X = _make_feature_matrix(10)
        stats = compute_reference_stats(X)
        assert isinstance(stats["reference_features"], dict)
        assert set(stats["reference_features"].keys()) == {
            "fp_a_density",
            "fp_b_density",
            "fp_diff_mean",
            "fp_product_mean",
            "tanimoto",
        }

    def test_reference_densities_length_matches_n_samples(self):
        X = _make_feature_matrix(73)
        stats = compute_reference_stats(X)
        assert len(stats["reference_densities"]) == 73


class TestDetectDrift:
    def test_insufficient_samples(self):
        X_new = _make_feature_matrix(4)
        stats = {"n_samples": 100, "reference_features": {}}
        result = detect_drift(X_new, stats)
        assert result["drift_detected"] is False
        assert result["reason"] == "insufficient_samples"

    def test_no_reference_features(self):
        X_new = _make_feature_matrix(10)
        stats = {"n_samples": 100}
        result = detect_drift(X_new, stats)
        assert result["drift_detected"] is False
        assert result["reason"] == "no_reference_features"

    def test_no_drift_same_distribution(self):
        rng = np.random.default_rng(42)
        X_ref = _make_feature_matrix(500, rng)
        stats = compute_reference_stats(X_ref)
        # Different seed, same distribution
        X_new = _make_feature_matrix(100, np.random.default_rng(99))
        result = detect_drift(X_new, stats)
        assert result["drift_detected"] is False

    def test_drift_different_distribution(self):
        rng = np.random.default_rng(42)
        X_ref = _make_feature_matrix(500, rng)
        stats = compute_reference_stats(X_ref)
        # All-zero fingerprints = very different from random binary
        X_new = np.zeros((100, N_FEATURES))
        result = detect_drift(X_new, stats, p_threshold=0.01)
        assert result["drift_detected"] is True

    def test_drift_output_keys(self):
        rng = np.random.default_rng(42)
        X_ref = _make_feature_matrix(500, rng)
        stats = compute_reference_stats(X_ref)
        X_new = _make_feature_matrix(100, np.random.default_rng(99))
        result = detect_drift(X_new, stats)
        expected = {"drift_detected", "drift_share", "drifted_columns", "column_p_values", "threshold", "n_new_samples"}
        assert expected.issubset(result.keys())

    def test_drift_share_is_float(self):
        X_ref = _make_feature_matrix(100)
        stats = compute_reference_stats(X_ref)
        X_new = _make_feature_matrix(20)
        result = detect_drift(X_new, stats)
        assert isinstance(result["drift_share"], float)
        assert 0.0 <= result["drift_share"] <= 1.0


class TestSaveReport:
    def test_saves_valid_json(self):
        report = {"drift_detected": False, "n_new_samples": 50}
        save_report(report)
        assert Path(DRIFT_REPORT).exists()
        with open(DRIFT_REPORT) as f:
            loaded = json.load(f)
        assert loaded == report

    def test_overwrites_previous_report(self):
        save_report({"a": 1})
        save_report({"b": 2})
        with open(DRIFT_REPORT) as f:
            loaded = json.load(f)
        assert loaded == {"b": 2}

    def tearDown(self):
        Path(DRIFT_REPORT).unlink(missing_ok=True)
