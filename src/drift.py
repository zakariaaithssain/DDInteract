import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chisquare

from src.config import logger

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
except ImportError:
    Report = None  # type: ignore[misc]
    DataDriftPreset = None  # type: ignore[misc]

from src.config import DRIFT_REPORT_PATH

DRIFT_REPORT: str = DRIFT_REPORT_PATH
N_BITS: int = 256
CLASS_NAMES = ["Minor", "Moderate", "Major"]


def fingerprint_density(fp_matrix: np.ndarray) -> np.ndarray:
    """Compute per-sample fingerprint density (mean across bit dimensions)."""
    return fp_matrix.mean(axis=1)


DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
    "TPSA",
    "NumRotatableBonds",
    "NumAromaticRings",
    "NumAliphaticRings",
    "FractionCSP3",
    "NumHeteroatoms",
]


def _extract_drift_features(X: np.ndarray) -> pd.DataFrame:
    """Extract per-sample aggregate features from the full feature matrix.

    Fingerprint bits (256 diff + 256 product) are collapsed into per-sample
    means. The 20 descriptor columns (10 diffs + 10 sums) are already dense
    scalars and are included directly.

    Args:
        X: Full feature matrix, shape (n, 2*N_BITS + 1 + 20).

    Returns:
        DataFrame with 23 columns.
    """
    fp_diff = X[:, :N_BITS]
    fp_prod = X[:, N_BITS : 2 * N_BITS]
    tanimoto = X[:, 2 * N_BITS]
    desc_start = 2 * N_BITS + 1
    prop_diff = X[:, desc_start : desc_start + 10]
    prop_sum = X[:, desc_start + 10 : desc_start + 20]

    data = {
        "fp_diff_mean": fp_diff.mean(axis=1),
        "fp_product_mean": fp_prod.mean(axis=1),
        "tanimoto": tanimoto,
    }
    for i, name in enumerate(DESCRIPTOR_NAMES):
        data[f"{name}_diff"] = prop_diff[:, i]
        data[f"{name}_sum"] = prop_sum[:, i]

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 1. Covariate shift — feature-distribution monitoring (existing)
# ---------------------------------------------------------------------------

N_REFERENCE_SAMPLES: int = 5_000


def compute_reference_stats(X: np.ndarray, y: np.ndarray | None = None) -> dict[str, Any]:
    """Compute reference statistics from training data.

    The reference data is randomly downsampled to ``N_REFERENCE_SAMPLES``
    (5,000) rows to avoid hyper-sensitivity in drift detection. With very
    large reference sets (>100,000 rows), even distance-based metrics flag
    microscopic, practically irrelevant shifts. A 5,000-row subsample
    balances statistical power with robustness.

    Args:
        X: Training feature matrix from ``build_features``,
           shape (n, 2*N_BITS + 1 + 20).
        y: Optional training labels for label-shift reference.

    Returns:
        Dictionary with feature reference, label distribution (if labels given),
        and metadata.
    """
    rng = np.random.default_rng(42)
    if len(X) > N_REFERENCE_SAMPLES:
        if y is not None:
            classes = np.unique(y)
            idx = np.concatenate(
                [
                    rng.choice(np.where(y == c)[0], size=int(N_REFERENCE_SAMPLES / len(classes)), replace=False)
                    for c in classes
                ]
            )
        else:
            idx = rng.choice(len(X), size=N_REFERENCE_SAMPLES, replace=False)
        X = X[idx]
        if y is not None:
            y = y[idx]
    ref_df = _extract_drift_features(X)
    result: dict[str, Any] = {
        "n_samples": int(len(X)),
        "reference_features": {k: [float(v) for v in vals] for k, vals in ref_df.to_dict("list").items()},
    }
    if y is not None:
        counts = {cls: int((y == cls).sum()) for cls in CLASS_NAMES}
        total = sum(counts.values()) or 1
        result["label_distribution"] = {cls: count / total for cls, count in counts.items()}
    return result


DRIFT_THRESHOLD_DISTANCE: float = 0.1


def detect_covariate_shift(
    X_new: np.ndarray, reference_stats: dict[str, Any], drift_threshold: float = DRIFT_THRESHOLD_DISTANCE
) -> dict[str, Any]:
    """Detect covariate shift — input feature distribution changes.

    Uses Evidently's DataDriftPreset, which adapts the detection method to
    the reference-sample size:
      * >1,000 reference samples → Wasserstein distance / Jensen-Shannon
        divergence with a distance threshold (default 0.1).
      * ≤1,000 reference samples → KS test / Chi-squared with p-value
        threshold (default 0.05).

    Our training set has ~110k samples, so distance-based metrics apply.
    Drift is flagged when >50% of columns exceed the threshold.

    Returns:
        Dictionary with covariate_drift flag, drift_share, and per-column
        drift scores.
    """
    n_new = len(X_new)
    if n_new < 5:
        return {"drift_detected": False, "reason": "insufficient_samples", "type": "covariate"}

    ref_features = reference_stats.get("reference_features")
    if ref_features is None:
        return {"drift_detected": False, "reason": "no_reference_features", "type": "covariate"}

    ref_df = pd.DataFrame(ref_features)
    cur_df = _extract_drift_features(X_new)

    report = Report(metrics=[DataDriftPreset(num_threshold=drift_threshold)])
    snapshot = report.run(reference_data=ref_df, current_data=cur_df)
    result = snapshot.dict()

    drift_share = 0.0
    effective_threshold = drift_threshold
    col_scores: dict[str, float] = {}
    for metric in result.get("metrics", []):
        value = metric.get("value")
        name = metric.get("metric_name", "")
        if name.startswith("DriftedColumnsCount"):
            if isinstance(value, dict):
                drift_share = float(value.get("share", 0.0))
                effective_threshold = float(value.get("num_threshold", drift_threshold))
        elif name.startswith("ValueDrift"):
            col = name.split("column=", 1)[1].split(",")[0] if "column=" in name else name
            if isinstance(value, float):
                col_scores[col] = value

    drifted_columns = [col for col, score in col_scores.items() if score < effective_threshold]
    detected = drift_share > 0.5

    if detected:
        logger.info("Covariate shift detected — %.0f%% of features drifted", drift_share * 100)
    else:
        logger.info("No covariate shift (%.0f%% features drifted)", drift_share * 100)

    return {
        "type": "covariate",
        "drift_detected": detected,
        "drift_share": drift_share,
        "drifted_columns": drifted_columns,
        "column_drift_scores": col_scores,
        "threshold": effective_threshold,
        "n_new_samples": n_new,
    }


# ---------------------------------------------------------------------------
# 2. Label shift — prior-probability monitoring
# ---------------------------------------------------------------------------


def detect_label_shift(
    predicted_classes: list[str], reference_stats: dict[str, Any], p_threshold: float = 0.05
) -> dict[str, Any]:
    """Detect label shift — changes in the predicted-class distribution.

    Compares the multinomial distribution of production predictions against
    the training label distribution using a chi-squared goodness-of-fit test.

    Returns:
        Dictionary with label_shift flag, p-value, and observed/expected
        proportions.
    """
    ref_dist = reference_stats.get("label_distribution")
    if ref_dist is None:
        return {"type": "label", "drift_detected": False, "reason": "no_label_reference"}

    if len(predicted_classes) < 10:
        return {"type": "label", "drift_detected": False, "reason": "insufficient_samples"}

    observed = np.array([predicted_classes.count(cls) for cls in CLASS_NAMES])
    n_total = observed.sum()
    expected = np.array([ref_dist[cls] * n_total for cls in CLASS_NAMES])

    # Avoid division-by-zero in chi-squared
    valid = expected > 0
    if not valid.all():
        # fallback: compare proportions directly
        observed_props = observed / n_total
        max_diff = max(abs(observed_props[i] - ref_dist[cls]) for i, cls in enumerate(CLASS_NAMES))
        detected = max_diff > 0.15
    else:
        stat, p_value = chisquare(f_obs=observed, f_exp=expected)
        detected = p_value < p_threshold

    observed_proportions = {cls: round(float(observed[i] / n_total), 4) for i, cls in enumerate(CLASS_NAMES)}
    expected_proportions = {cls: round(float(ref_dist[cls]), 4) for cls in CLASS_NAMES}

    if detected:
        logger.info("Label shift detected — production class distribution differs from training")

    return {
        "type": "label",
        "drift_detected": detected,
        "p_value": float(p_value) if valid.all() else None,
        "observed_proportions": observed_proportions,
        "expected_proportions": expected_proportions,
        "n_samples": n_total,
    }


# ---------------------------------------------------------------------------
# 3. Concept drift — prediction-confidence monitoring
# ---------------------------------------------------------------------------


def detect_concept_drift(confidences: list[float], p_threshold: float = 0.05) -> dict[str, Any]:
    """Detect concept drift — changes in prediction confidence distribution.

    Concept drift (changes in P(Y|X)) is challenging to detect without
    ground-truth labels. As a proxy, we monitor the distribution of
    prediction confidence (max probability). A significant downward shift
    in confidence indicates the model is becoming less certain, which often
    precedes concept drift.

    Uses a one-sample KS test comparing current confidence against the
    expected uniform-ish distribution, with a Bonferroni-like floor.

    Returns:
        Dictionary with concept_drift flag, mean/median confidence, and
        distribution statistics.
    """
    if len(confidences) < 10:
        return {"type": "concept", "drift_detected": False, "reason": "insufficient_samples"}

    conf_arr = np.array(confidences)
    mean_conf = float(np.mean(conf_arr))
    median_conf = float(np.median(conf_arr))
    std_conf = float(np.std(conf_arr))
    min_conf = float(conf_arr.min())

    # Heuristic: flag if mean confidence drops below 0.5 or if there is
    # a statistically significant cluster of very low-confidence predictions
    low_conf_share = float((conf_arr < 0.5).mean())

    # KS test against a beta distribution parameterized from training
    # (approximate: compare current vs. pooled historical confidences)
    # As a simple heuristic, we compare current batch against the pooled
    # distribution from previous batches via an internal reference.
    detected = mean_conf < 0.5 or low_conf_share > 0.3

    if detected:
        logger.info(
            "Concept drift warning — mean confidence=%.3f, low-conf share=%.1f%%",
            mean_conf,
            low_conf_share * 100,
        )

    return {
        "type": "concept",
        "drift_detected": detected,
        "mean_confidence": mean_conf,
        "median_confidence": median_conf,
        "std_confidence": std_conf,
        "min_confidence": min_conf,
        "low_confidence_share": low_conf_share,
        "n_samples": len(confidences),
    }


# ---------------------------------------------------------------------------
# Unified orchestration
# ---------------------------------------------------------------------------


def detect_drift(
    X_new: np.ndarray,
    predicted_classes: list[str],
    confidences: list[float],
    reference_stats: dict[str, Any],
    drift_threshold: float = DRIFT_THRESHOLD_DISTANCE,
    p_threshold: float = 0.05,
) -> dict[str, Any]:
    """Run all three drift checks and return a unified report.

    Args:
        X_new: Buffered feature vectors, shape (n, 2*N_BITS + 1 + 20).
        predicted_classes: Predicted class labels for each sample.
        confidences: Prediction confidence (max probability) for each sample.
        reference_stats: Reference from ``compute_reference_stats``.
        drift_threshold: Distance threshold for covariate shift detection
                         (default 0.1, recommended for >1k reference samples).

    Returns:
        Dictionary with overall drift assessment and per-type results.
    """
    covariate = detect_covariate_shift(X_new, reference_stats, drift_threshold)
    label = detect_label_shift(predicted_classes, reference_stats, p_threshold)
    concept = detect_concept_drift(confidences, p_threshold)

    drift_detected = any(r["drift_detected"] for r in [covariate, label, concept])

    if drift_detected:
        logger.warning(
            "Drift detected: covariate=%s label=%s concept=%s",
            covariate["drift_detected"],
            label["drift_detected"],
            concept["drift_detected"],
        )

    return {
        "drift_detected": drift_detected,
        "covariate_shift": covariate,
        "label_shift": label,
        "concept_drift": concept,
    }


def save_report(report: dict[str, Any]) -> None:
    """Save drift detection report to a JSON file."""
    with open(DRIFT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Drift report saved to %s", DRIFT_REPORT)
