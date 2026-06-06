from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.features import build_features
from src.logger import logger
from src.models import RANDOM_STATE

EXPERIMENT_NAME: str = "DDI_Structural_Severity"
DATA_PATH: str = "data/chemical_ddi.csv"
TEST_SIZE: float = 0.2
N_PCA: int = 50
FEATURE_CACHE: str = "data/features.npy"
LABEL_CACHE: str = "data/labels.npy"


def main() -> None:
    """Export the best model from MLflow along with its scaler and PCA.

    Queries MLflow for the run with the highest macro F1, downloads the
    model, rebuilds the scaler and PCA from training data, and saves all
    three as joblib files to the models/ directory.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)
    runs = mlflow.search_runs()

    runs = runs[~runs["tags.mlflow.runName"].str.startswith("best_", na=False)]
    runs = runs[pd.notna(runs["metrics.macro_f1"])]
    best = runs.loc[runs["metrics.macro_f1"].idxmax()]

    run_id: str = best["run_id"]  # type: ignore[call-overload]
    model_name: str = best["tags.mlflow.runName"]  # type: ignore[call-overload]
    logger.info("Loading best model: %s (run_id=%s)", model_name, run_id)

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/model.joblib")
    logger.info("Saved models/model.joblib")

    if Path(FEATURE_CACHE).exists():
        logger.info("Loading cached features")
        X = np.load(FEATURE_CACHE)
        y = np.load(LABEL_CACHE)
    else:
        logger.info("Building features from SMILES")
        df = pd.read_csv(DATA_PATH)
        y = df["severity_label"].values
        X = build_features(df)

    X_train, _, _, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, "models/scaler.joblib")
    logger.info("Saved models/scaler.joblib")

    pca = PCA(n_components=N_PCA, random_state=RANDOM_STATE)
    pca.fit(scaler.transform(X_train))
    joblib.dump(pca, "models/pca.joblib")
    logger.info("Saved models/pca.joblib")

    logger.info("Export complete — model, scaler, PCA saved to models/")


if __name__ == "__main__":
    main()
