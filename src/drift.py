import json
from typing import Any

import numpy as np
import pandas as pd

from src.logger import logger

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
except ImportError:
    Report = None  # type: ignore[misc]
    DataDriftPreset = None  # type: ignore[misc]

from src.config import DRIFT_REPORT_PATH

DRIFT_REPORT: str = DRIFT_REPORT_PATH
N_BITS: int = 256


def fingerprint_density(fp_matrix: np.ndarray) -> np.ndarray:
    """Compute per-sample fingerprint density (mean across bit dimensions).

    Args:
        fp_matrix: Binary fingerprint matrix of shape (n_samples, n_bits).

    Returns:
        Array of mean bit densities per sample of shape (n_samples,).
    """
    return fp_matrix.mean(axis=1)


def _extract_drift_features(X: np.ndarray) -> pd.DataFrame:
    """Extract per-sample aggregate features from the full feature matrix.

    The full ``build_features`` output has 4 × N_BITS fingerprint columns,
    Tanimoto similarity, and 20 molecular-descriptor columns.  This function
    collapses the fingerprint block into four interpretable per-sample
    aggregates so that drift can be compared across a handful of meaningful
    dimensions instead of hundreds of individual bits.

    Args:
        X: Full feature matrix from ``build_features``, shape (n, 4*N_BITS + 1 + 20).

    Returns:
        DataFrame with columns ``fp_a_density``, ``fp_b_density``,
        ``fp_diff_mean``, ``fp_product_mean``, ``tanimoto``,
        and the 20 descriptor-based features.
    """
    fp_a = X[:, :N_BITS]
    fp_b = X[:, N_BITS : 2 * N_BITS]
    fp_diff = X[:, 2 * N_BITS : 3 * N_BITS]
    fp_prod = X[:, 3 * N_BITS : 4 * N_BITS]
    tanimoto = X[:, 4 * N_BITS]

    return pd.DataFrame(
        {
            "fp_a_density": fp_a.mean(axis=1),
            "fp_b_density": fp_b.mean(axis=1),
            "fp_diff_mean": fp_diff.mean(axis=1),
            "fp_product_mean": fp_prod.mean(axis=1),
            "tanimoto": tanimoto,
        }
    )


def compute_reference_stats(X: np.ndarray) -> dict[str, Any]:
    """Compute and store reference drift features from training data.

    Stores both the aggregate feature DataFrame and per-fingerprint densities
    so that subsequent ``detect_drift`` calls can compare actual distributions.

    Args:
        X: Full training feature matrix from ``build_features``, shape
           (n_samples, 4*N_BITS + 1 + 20).

    Returns:
        Dictionary with summary statistics and the full reference feature
        DataFrame.
    """
    ref_df = _extract_drift_features(X)
    densities_a = fingerprint_density(X[:, :N_BITS])
    return {
        "fp_density_mean": float(densities_a.mean()),
        "fp_density_std": float(densities_a.std()),
        "fp_density_p5": float(np.percentile(densities_a, 5)),
        "fp_density_p95": float(np.percentile(densities_a, 95)),
        "n_samples": int(len(X)),
        "reference_features": {k: [float(v) for v in vals] for k, vals in ref_df.to_dict("list").items()},
        "reference_densities": [float(d) for d in densities_a],
    }


def detect_drift(X_new: np.ndarray, reference_stats: dict[str, Any], p_threshold: float = 0.05) -> dict[str, Any]:
    """Detect data drift using Evidently's per-column KS tests.

    Compares five feature-space aggregates (fp_a density, fp_b density,
    fingerprint diff mean, fingerprint product mean, Tanimoto similarity)
    between the reference and new data. Drift is reported when more than
    half of the features exceed the p-value threshold.

    Args:
        X_new: New full-feature matrix from ``build_features``, shape
               (n_samples, 4*N_BITS + 1 + 20).
        reference_stats: Reference statistics from ``compute_reference_stats``.
        p_threshold: P-value threshold for per-column drift detection.

    Returns:
        Dictionary with drift_detected flag, share of drifted columns,
        per-column p-values, and summary statistics.
    """
    n_new = len(X_new)
    if n_new < 5:
        return {"drift_detected": False, "reason": "insufficient_samples"}

    ref_features = reference_stats.get("reference_features")
    if ref_features is None:
        return {"drift_detected": False, "reason": "no_reference_features"}

    ref_df = pd.DataFrame(ref_features)
    cur_df = _extract_drift_features(X_new)

    report = Report(metrics=[DataDriftPreset(num_threshold=p_threshold)])
    snapshot = report.run(reference_data=ref_df, current_data=cur_df)
    result = snapshot.dict()

    # Parse drift share from DriftedColumnsCount and p-values from ValueDrift metrics
    drift_share = 0.0
    threshold = p_threshold
    col_pvalues: dict[str, float] = {}
    for metric in result.get("metrics", []):
        value = metric.get("value")
        name = metric.get("metric_name", "")
        if name.startswith("DriftedColumnsCount"):
            if isinstance(value, dict):
                drift_share = float(value.get("share", 0.0))
                threshold = float(value.get("num_threshold", p_threshold))
        elif name.startswith("ValueDrift"):
            # value is a float p-value directly for ValueDrift metrics
            col = name.split("column=", 1)[1].split(",")[0] if "column=" in name else name
            if isinstance(value, float):
                col_pvalues[col] = value

    drifted_columns = [col for col, pv in col_pvalues.items() if pv < threshold]

    result_dict: dict[str, Any] = {
        "drift_detected": drift_share > 0.5,
        "drift_share": drift_share,
        "drifted_columns": drifted_columns,
        "column_p_values": col_pvalues,
        "threshold": p_threshold,
        "n_new_samples": n_new,
    }

    if drift_share > 0.5:
        logger.info(
            "Data drift detected — %.0f%% of features drifted (threshold p=%.2f)",
            drift_share * 100,
            p_threshold,
        )
    else:
        logger.info("No significant drift (%.0f%% features drifted)", drift_share * 100)

    return result_dict


def save_report(report: dict[str, Any]) -> None:
    """Save drift detection report to a JSON file.

    Args:
        report: Dictionary with drift detection results.
    """
    with open(DRIFT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Drift report saved to %s", DRIFT_REPORT)
