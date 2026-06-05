import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from features import build_features
from models import RANDOM_STATE

EXPERIMENT_NAME = "DDI_Structural_Severity"
DATA_PATH = "data/chemical_ddi.csv"
TEST_SIZE = 0.2
N_PCA = 50


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    runs = mlflow.search_runs()

    runs = runs[~runs["tags.mlflow.runName"].str.startswith("best_", na=False)]
    runs = runs[pd.notna(runs["metrics.macro_f1"])]
    best = runs.loc[runs["metrics.macro_f1"].idxmax()]

    run_id = best["run_id"]
    model_name = best["tags.mlflow.runName"]
    print(f"Loading best model: {model_name} (run_id={run_id})")

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    joblib.dump(model, "models/model.joblib")
    print("  saved models/model.joblib")

    df = pd.read_csv(DATA_PATH)
    y = df["severity_label"].values
    X = build_features(df)

    X_train, _, _, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    joblib.dump(scaler, "models/scaler.joblib")
    print("  saved models/scaler.joblib")

    pca = PCA(n_components=N_PCA, random_state=RANDOM_STATE)
    pca.fit(X_train_s)
    joblib.dump(pca, "models/pca.joblib")
    print("  saved models/pca.joblib")

    print("Export complete.")


if __name__ == "__main__":
    main()
