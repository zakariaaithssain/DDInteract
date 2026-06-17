"""Step 2: VarianceThreshold feature selection and StandardScaler.

Loads cached features, splits into train/test, fits and saves
VarianceThreshold selector + StandardScaler, and saves transformed
arrays for the next step.
"""

import os

import joblib
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURE_CACHE,
    LABEL_CACHE,
    LOGS_DIR,
    MODELS_DIR,
    SCALER_PATH,
    TEST_SIZE,
    VARIANCE_THRESHOLD,
    VT_PATH,
    X_TEST_PATH,
    X_TRAIN_PATH,
    Y_TEST_PATH,
    Y_TRAIN_PATH,
)
from src.features import N_BITS
from src.logger import logger
from src.models import RANDOM_STATE

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


def _feature_label(idx: int) -> str:
    if idx < N_BITS:
        return f"fp_a bit {idx}"
    if idx < 2 * N_BITS:
        return f"fp_b bit {idx - N_BITS}"
    if idx < 3 * N_BITS:
        return f"diff bit {idx - 2 * N_BITS}"
    if idx < 4 * N_BITS:
        return f"product bit {idx - 3 * N_BITS}"
    if idx == 4 * N_BITS:
        return "tanimoto"
    desc_idx = idx - (4 * N_BITS + 1)
    if desc_idx < 10:
        return f"prop_diff {DESCRIPTOR_NAMES[desc_idx]}"
    return f"prop_sum {DESCRIPTOR_NAMES[desc_idx - 10]}"


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    X = np.load(FEATURE_CACHE)
    y = np.load(LABEL_CACHE)
    logger.info("Loaded features: %s, labels: %s", X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train/test split: %d train, %d test", len(X_train), len(X_test))

    selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    X_train_sel = selector.fit_transform(X_train)
    X_test_sel = selector.transform(X_test)

    support = selector.get_support()
    n_kept = int(support.sum())
    n_dropped = len(X_train[0]) - n_kept
    logger.info(
        "VarianceThreshold: %d / %d features kept (%.1f%%)", n_kept, len(X_train[0]), 100.0 * n_kept / len(X_train[0])
    )
    if n_dropped > 0:
        dropped_idx = np.where(~support)[0]
        dropped_vars = selector.variances_[dropped_idx]
        max_log = min(n_dropped, 30)
        logger.info("Dropped features (showing first %d of %d):", max_log, n_dropped)
        for i in range(max_log):
            fi = dropped_idx[i]
            logger.info("  [%4d] %-30s variance=%.6f", fi, _feature_label(int(fi)), dropped_vars[i])
        if n_dropped > max_log:
            logger.info("  ... and %d more", n_dropped - max_log)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    joblib.dump(selector, VT_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Saved selector and scaler to %s/", MODELS_DIR)

    np.save(X_TRAIN_PATH, X_train_scaled)
    np.save(X_TEST_PATH, X_test_scaled)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(Y_TEST_PATH, y_test)
    logger.info("Saved transformed splits to %s/", MODELS_DIR)


if __name__ == "__main__":
    main()
