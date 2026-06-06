from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    BEST_MODEL_PATH,
    DATA_PATH,
    FEATURE_CACHE,
    LABEL_CACHE,
    MODEL_PATH,
    MODELS_DIR,
    PCA_PATH,
    SCALER_PATH,
    TEST_SIZE,
)
from src.features import build_features
from src.logger import logger
from src.models import RANDOM_STATE

EXPERIMENT_NAME: str = "DDI_Structural_Severity"
N_PCA: int = 50


def _best_model_path() -> Path:
    """Return the path to the locally saved best model."""
    return Path(BEST_MODEL_PATH)


def _load_model_from_mlflow(run_id: str) -> object:
    """Load a model from an MLflow run, trying sklearn then xgboost.

    Args:
        run_id: MLflow run ID.

    Returns:
        Loaded model object.
    """
    try:
        return mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    except Exception:
        return mlflow.xgboost.load_model(f"runs:/{run_id}/model")


def _rebuild_scaler_pca(X: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, PCA]:
    """Rebuild the scaler and PCA from training data.

    Uses the same train/test split as the training pipeline.

    Args:
        X: Full feature matrix.
        y: Full label array.

    Returns:
        Tuple of (fitted StandardScaler, fitted PCA).
    """
    X_train, _, _, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler()
    scaler.fit(X_train)
    pca = PCA(n_components=N_PCA, random_state=RANDOM_STATE)
    pca.fit(scaler.transform(X_train))
    return scaler, pca


def main() -> None:
    """Export a model for the API.

    Prefers loading the best model found by a previous ``train`` run from
    ``BEST_MODEL_PATH``.  If that doesn't exist, queries MLflow for the run
    with the highest macro F1 and downloads it.  Then rebuilds the scaler
    and PCA from training data and saves everything under ``MODELS_DIR``.
    """
    Path(MODELS_DIR).mkdir(exist_ok=True)

    local_path = _best_model_path()
    if local_path.exists():
        logger.info("Loading best model from %s", local_path)
        model = joblib.load(local_path)
    else:
        logger.info("Querying MLflow for best run")
        mlflow.set_experiment(EXPERIMENT_NAME)
        runs = mlflow.search_runs()
        runs = runs[~runs["tags.mlflow.runName"].str.startswith("best_", na=False)]
        runs = runs[pd.notna(runs["metrics.macro_f1"])]
        if runs.empty:
            logger.error("No completed training runs found in MLflow")
            return
        best = runs.loc[runs["metrics.macro_f1"].idxmax()]
        run_id: str = best["run_id"]  # type: ignore[call-overload]
        model_name: str = best["tags.mlflow.runName"]  # type: ignore[call-overload]
        logger.info("Loading best model: %s (run_id=%s)", model_name, run_id)
        model = _load_model_from_mlflow(run_id)

    joblib.dump(model, MODEL_PATH)
    logger.info("Saved %s", MODEL_PATH)

    if Path(FEATURE_CACHE).exists():
        logger.info("Loading cached features")
        X = np.load(FEATURE_CACHE)
        y = np.load(LABEL_CACHE)
    else:
        logger.info("Building features from SMILES")
        df = pd.read_csv(DATA_PATH)
        y = df["severity_label"].values
        X = build_features(df)

    scaler, pca = _rebuild_scaler_pca(X, y)

    joblib.dump(scaler, SCALER_PATH)
    logger.info("Saved %s", SCALER_PATH)

    joblib.dump(pca, PCA_PATH)
    logger.info("Saved %s", PCA_PATH)

    logger.info("Export complete — model, scaler, PCA saved to %s/", MODELS_DIR)


if __name__ == "__main__":
    main()
