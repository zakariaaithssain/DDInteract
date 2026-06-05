import joblib
import mlflow
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from features import build_features
from models import RANDOM_STATE
from logger import logger

EXPERIMENT_NAME = "DDI_Structural_Severity"
DATA_PATH = "data/chemical_ddi.csv"
TEST_SIZE = 0.2
N_PCA = 50
FEATURE_CACHE = "data/features.npy"
LABEL_CACHE = "data/labels.npy"


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    runs = mlflow.search_runs()

    runs = runs[~runs["tags.mlflow.runName"].str.startswith("best_", na=False)]
    runs = runs[pd.notna(runs["metrics.macro_f1"])]
    best = runs.loc[runs["metrics.macro_f1"].idxmax()]

    run_id = best["run_id"]
    model_name = best["tags.mlflow.runName"]
    logger.info("Loading best model: %s (run_id=%s)", model_name, run_id)

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/model.joblib")
    logger.info("Saved models/model.joblib")

    if Path(FEATURE_CACHE).exists():
        logger.info("Loading cached features")
        import numpy as np
        X = np.load(FEATURE_CACHE)
        y = np.load(LABEL_CACHE)
    else:
        logger.info("Building features from SMILES")
        df = pd.read_csv(DATA_PATH)
        y = df["severity_label"].values
        X = build_features(df)

    X_train, _, _, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

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
