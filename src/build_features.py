"""Step 1: Load raw data, build feature matrix, cache features, save drift reference."""

import json
import os

import pandas as pd

from src.config import DATA_PATH, DRIFT_REFERENCE_PATH, LOGS_DIR, MODELS_DIR
from src.data import load_or_build_features
from src.drift import compute_reference_stats
from src.logger import logger


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
