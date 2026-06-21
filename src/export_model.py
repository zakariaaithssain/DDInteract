"""Model export and MLflow Model Registry utilities."""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from src.config import BEST_MODEL_PATH, EXPERIMENT_NAME, MODEL_PATH, MODELS_DIR, REGISTRY_NAME, logger


def register_best_model(run_id: str, family: str, composite: float, best_trial_number: int | None = None) -> None:
    model_uri = f"runs:/{run_id}/model"
    try:
        result = mlflow.register_model(model_uri, REGISTRY_NAME)
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(REGISTRY_NAME, "production", result.version)
        client.set_model_version_tag(REGISTRY_NAME, result.version, "family", family)
        client.set_model_version_tag(REGISTRY_NAME, result.version, "composite", str(composite))
        if best_trial_number is not None:
            client.set_model_version_tag(REGISTRY_NAME, result.version, "best_trial_number", str(best_trial_number))
        logger.info(
            "Registered %s as version %s of '%s' (composite=%.4f)", family, result.version, REGISTRY_NAME, composite
        )
    except Exception as e:
        logger.warning("Model registration failed (MLflow registry may be local-only): %s", e)


def _best_model_path() -> Path:
    return Path(BEST_MODEL_PATH)


def _load_model_from_mlflow(run_id: str) -> object:
    try:
        return mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    except Exception:
        return mlflow.xgboost.load_model(f"runs:/{run_id}/model")


def main() -> None:
    """Export a model for the API.

    Prefers loading the best model found by a previous ``train`` run from
    ``BEST_MODEL_PATH``.  If that doesn't exist, queries MLflow for the run
    with the highest macro F1 and downloads it.  Saves the model under ``MODELS_DIR``.
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
        runs = runs[pd.notna(runs["metrics.composite"])]
        if runs.empty:
            logger.error("No completed training runs found in MLflow")
            return
        best = runs.loc[runs["metrics.composite"].idxmax()]
        run_id: str = best["run_id"]
        model_name: str = best["tags.mlflow.runName"]
        logger.info("Loading best model: %s (run_id=%s)", model_name, run_id)
        model = _load_model_from_mlflow(run_id)

    joblib.dump(model, MODEL_PATH)
    logger.info("Saved %s", MODEL_PATH)

    logger.info("Export complete — model saved to %s/", MODELS_DIR)


if __name__ == "__main__":
    main()
