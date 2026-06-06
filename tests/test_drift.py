"""Tests for data drift detection."""

import sys

sys.path.insert(0, "src")

import json
from pathlib import Path

import numpy as np

from drift import (
    DRIFT_REPORT,
    compute_reference_stats,
    detect_drift,
    fingerprint_density,
    save_report,
)


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


class TestComputeReferenceStats:
    def test_output_keys(self):
        rng = np.random.default_rng(42)
        X = rng.integers(0, 2, size=(100, 256), dtype=np.int32).astype(np.float64)
        stats = compute_reference_stats(X)
        assert set(stats.keys()) == {
            "fp_density_mean",
            "fp_density_std",
            "fp_density_p5",
            "fp_density_p95",
            "n_samples",
        }

    def test_n_samples(self):
        X = np.zeros((50, 256))
        stats = compute_reference_stats(X)
        assert stats["n_samples"] == 50

    def test_density_mean_value(self):
        X = np.ones((10, 256))
        stats = compute_reference_stats(X)
        assert stats["fp_density_mean"] == 1.0

    def test_density_std_zero_for_uniform(self):
        X = np.ones((10, 256))
        stats = compute_reference_stats(X)
        assert stats["fp_density_std"] == 0.0


class TestDetectDrift:
    def test_insufficient_samples(self):
        X_new = np.zeros((5, 256))
        stats = {"n_samples": 100, "fp_density_mean": 0.5, "fp_density_std": 0.1}
        result = detect_drift(X_new, stats)
        assert result["drift_detected"] is False
        assert result["reason"] == "insufficient_samples"

    def test_no_drift_same_distribution(self):
        rng = np.random.default_rng(42)
        X_ref = rng.integers(0, 2, size=(500, 256), dtype=np.int32).astype(np.float64)
        stats = compute_reference_stats(X_ref)
        X_new = rng.integers(0, 2, size=(100, 256), dtype=np.int32).astype(np.float64)
        result = detect_drift(X_new, stats)
        assert result["drift_detected"] is False

    def test_drift_different_distribution(self):
        rng = np.random.default_rng(42)
        X_ref = rng.integers(0, 2, size=(500, 256), dtype=np.int32).astype(np.float64)
        stats = compute_reference_stats(X_ref)
        X_new = np.zeros((100, 256))
        result = detect_drift(X_new, stats, p_threshold=0.01)
        assert result["drift_detected"] is True

    def test_drift_output_keys(self):
        rng = np.random.default_rng(42)
        X_ref = rng.integers(0, 2, size=(500, 256), dtype=np.int32).astype(np.float64)
        stats = compute_reference_stats(X_ref)
        X_new = rng.integers(0, 2, size=(100, 256), dtype=np.int32).astype(np.float64)
        result = detect_drift(X_new, stats)
        expected_keys = {
            "drift_detected",
            "ks_statistic",
            "p_value",
            "threshold",
            "new_fp_density_mean",
            "reference_fp_density_mean",
            "n_new_samples",
        }
        assert expected_keys.issubset(result.keys())


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
