from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_CACHE, LABEL_CACHE
from src.features import build_features
from src.logger import logger


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
    X = build_features(df)
    np.save(FEATURE_CACHE, X)
    np.save(LABEL_CACHE, y)
    logger.info("Features cached to %s and %s", FEATURE_CACHE, LABEL_CACHE)
    return X, y
