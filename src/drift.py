import json
from typing import Any

import numpy as np
from scipy.stats import ks_2samp

from logger import logger

DRIFT_REPORT: str = "drift_report.json"
CLASS_NAMES: list[str] = ["Minor", "Moderate", "Major"]


def fingerprint_density(fp_matrix: np.ndarray) -> np.ndarray:
    """Compute per-sample fingerprint density (mean across bit dimensions).

    Args:
        fp_matrix: Binary fingerprint matrix of shape (n_samples, n_bits).

    Returns:
        Array of mean bit densities per sample of shape (n_samples,).
    """
    return fp_matrix.mean(axis=1)


def compute_reference_stats(X: np.ndarray) -> dict[str, Any]:
    """Compute reference fingerprint density statistics from training data.

    Args:
        X: Training fingerprint matrix of shape (n_samples, n_bits).

    Returns:
        Dictionary with fp_density_mean, fp_density_std, fp_density_p5,
        fp_density_p95, and n_samples.
    """
    densities = fingerprint_density(X)
    return {
        "fp_density_mean": float(densities.mean()),
        "fp_density_std": float(densities.std()),
        "fp_density_p5": float(np.percentile(densities, 5)),
        "fp_density_p95": float(np.percentile(densities, 95)),
        "n_samples": len(X),
    }


def detect_drift(X_new: np.ndarray, reference_stats: dict[str, Any], p_threshold: float = 0.05) -> dict[str, Any]:
    """Detect data drift using a two-sample Kolmogorov-Smirnov test.

    Args:
        X_new: New fingerprint matrix of shape (n_samples, n_bits).
        reference_stats: Reference statistics from compute_reference_stats.
        p_threshold: P-value threshold for drift detection.

    Returns:
        Dictionary with drift_detected flag, KS statistic, p-value,
        and summary statistics.
    """
    new_densities = fingerprint_density(X_new)
    n_new = len(X_new)

    if n_new < 20:
        return {"drift_detected": False, "reason": "insufficient_samples"}

    n_ref_samples = reference_stats["n_samples"]
    ref_densities = np.random.default_rng(42).normal(
        reference_stats["fp_density_mean"],
        reference_stats["fp_density_std"],
        min(n_ref_samples, 10000),
    )

    stat, p_value = ks_2samp(ref_densities, new_densities)
    drift = bool(p_value < p_threshold)

    result: dict[str, Any] = {
        "drift_detected": drift,
        "ks_statistic": float(stat),
        "p_value": float(p_value),
        "threshold": p_threshold,
        "new_fp_density_mean": float(new_densities.mean()),
        "reference_fp_density_mean": reference_stats["fp_density_mean"],
        "n_new_samples": n_new,
    }

    if drift:
        logger.warning("Data drift detected! KS p-value=%.4f (threshold=%s)", p_value, p_threshold)
    else:
        logger.info("No drift detected (p=%.4f)", p_value)

    return result


def save_report(report: dict[str, Any]) -> None:
    """Save drift detection report to a JSON file.

    Args:
        report: Dictionary with drift detection results.
    """
    with open(DRIFT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Drift report saved to %s", DRIFT_REPORT)
