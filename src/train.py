"""Step 3: Full-data training with best hyperparameters.

Reads ``best_params.json`` from the search step, trains the
chosen model family on the full training set, logs all artifacts
to MLflow, and registers the model in the Model Registry.
"""

import json
import os

import joblib
import mlflow
import mlflow.models
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.config import (
    BEST_MODEL_PATH,
    BEST_PARAMS_PATH,
    EXPERIMENT_NAME,
    FEATURE_CACHE,
    LABEL_CACHE,
    MODELS_DIR,
    TEST_SIZE,
    logger,
)
from src.evaluate import evaluate_and_log
from src.export_model import register_best_model

RANDOM_STATE: int = 42


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    mlflow.set_experiment(EXPERIMENT_NAME)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is not None:
        client.set_experiment_tag(experiment.experiment_id, "project", "DDI_Severity_Predictor")
        client.set_experiment_tag(experiment.experiment_id, "task", "multi_class_classification")
        client.set_experiment_tag(experiment.experiment_id, "metrics_primary", "macro_f1")
        client.set_experiment_tag(experiment.experiment_id, "model_families", "RandomForest,XGBoost")

    with open(BEST_PARAMS_PATH) as f:
        best_params = json.load(f)
    best_family: str = best_params.pop("family")
    logger.info("Loaded best params for family=%s: %s", best_family, best_params)

    X = np.load(FEATURE_CACHE)
    y = np.load(LABEL_CACHE)
    logger.info("Loaded features: %s, labels: %s", X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Full train/test split: %d train, %d test", len(X_train), len(X_test))

    if best_family == "RandomForest":
        model = RandomForestClassifier(**best_params)
    else:
        model = XGBClassifier(**best_params, tree_method="hist", n_jobs=-1, verbosity=0)
    model.fit(X_train, y_train)
    logger.info("Trained %s on full training set", best_family)

    joblib.dump(model, BEST_MODEL_PATH)
    logger.info("Model saved to %s", BEST_MODEL_PATH)

    with mlflow.start_run(run_name="best_overall") as best_run:
        mlflow.set_tag("best_overall", "true")
        mlflow.set_tag("model_family", best_family)
        mlflow.log_params(best_params)

        metrics = evaluate_and_log(model, X_test, y_test, "best_overall", best_params, output_dir=MODELS_DIR)

        signature = mlflow.models.infer_signature(X_test[:5], model.predict(X_test[:5]))
        mlflow.sklearn.log_model(model, name="model", signature=signature)

        with open("/tmp/model_signature.json", "w") as _f:
            json.dump(signature.to_dict(), _f, indent=2)
        mlflow.log_artifact("/tmp/model_signature.json")

        mlflow.log_artifact(BEST_PARAMS_PATH)

    logger.info("MLflow run: %s", best_run.info.run_id)

    macro_f1 = metrics["macro_f1"]
    register_best_model(
        best_run.info.run_id,
        best_family,
        macro_f1,
        best_trial_number=None,
    )

    logger.info("Training complete — model saved to %s/", MODELS_DIR)


if __name__ == "__main__":
    main()
