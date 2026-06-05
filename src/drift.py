import numpy as np
import json
from pathlib import Path
from scipy.stats import ks_2samp
from logger import logger

DRIFT_REPORT = "drift_report.json"
CLASS_NAMES = ["Minor", "Moderate", "Major"]


def fingerprint_density(fp_matrix: np.ndarray) -> np.ndarray:
    return fp_matrix.mean(axis=1)


def compute_reference_stats(X: np.ndarray) -> dict:
    densities = fingerprint_density(X)
    return {
        "fp_density_mean": float(densities.mean()),
        "fp_density_std": float(densities.std()),
        "fp_density_p5": float(np.percentile(densities, 5)),
        "fp_density_p95": float(np.percentile(densities, 95)),
        "n_samples": len(X),
    }


def detect_drift(X_new: np.ndarray, reference_stats: dict, p_threshold: float = 0.05) -> dict:
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

    result = {
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


def save_report(report: dict):
    with open(DRIFT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Drift report saved to %s", DRIFT_REPORT)
