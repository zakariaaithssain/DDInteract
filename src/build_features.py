"""Step 1: Load raw data, build feature matrix, cache features, save drift reference."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PATH, DRIFT_REFERENCE_PATH, FEATURE_CACHE, LABEL_CACHE, LOGS_DIR, MODELS_DIR, logger
from src.drift import compute_reference_stats
from src.features import build_features as _build_features


def load_or_build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if Path(FEATURE_CACHE).exists() and Path(LABEL_CACHE).exists():
        logger.info("Loading cached features from %s", FEATURE_CACHE)
        X = np.load(FEATURE_CACHE)
        y = np.load(LABEL_CACHE)
        if len(X) == len(df):
            return X, y
        logger.warning("Cache size mismatch, rebuilding features")
    logger.info("Building features from SMILES (this may take a few minutes)")
    y = df["severity_label"].values
    X = _build_features(df)
    np.save(FEATURE_CACHE, X)
    np.save(LABEL_CACHE, y)
    logger.info("Features cached to %s and %s", FEATURE_CACHE, LABEL_CACHE)
    return X, y


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded %d rows from %s", len(df), DATA_PATH)

    X, y = load_or_build_features(df)

    drift_ref = compute_reference_stats(X)
    with open(DRIFT_REFERENCE_PATH, "w") as f:
        json.dump(drift_ref, f, indent=2)
    logger.info("Drift reference stats saved to %s", DRIFT_REFERENCE_PATH)


if __name__ == "__main__":
    main()
