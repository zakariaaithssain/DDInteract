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
    THRESHOLDS_PATH,
    logger,
)
from src.evaluate import evaluate_and_log
from src.export_model import register_best_model
from src.threshold_optimizer import (
    optimize_thresholds,
    save_thresholds,
)

RANDOM_STATE: int = 42


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    mlflow.set_experiment(EXPERIMENT_NAME)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is not None:
        client.set_experiment_tag(experiment.experiment_id, "project", "DDI_Severity_Predictor")
        client.set_experiment_tag(experiment.experiment_id, "task", "multi_class_classification")
        client.set_experiment_tag(experiment.experiment_id, "metrics_primary", "composite")
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

    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    logger.info("Sub-split: %d train, %d val for threshold optimization", len(X_train_sub), len(X_val))

    if best_family == "RandomForest":
        model = RandomForestClassifier(**best_params, class_weight="balanced")
        model.fit(X_train, y_train)
    else:
        model = XGBClassifier(**best_params, tree_method="hist", n_jobs=-1, verbosity=0)
        classes, counts = np.unique(y_train, return_counts=True)
        class_weights = len(y_train) / (len(classes) * counts)
        sample_weight = class_weights[y_train]
        model.fit(X_train, y_train, sample_weight=sample_weight)
    logger.info("Trained %s on full training set", best_family)

    val_probs = model.predict_proba(X_val)
    thresholds = optimize_thresholds(y_val, val_probs, n_trials=200)
    save_thresholds(thresholds)

    joblib.dump(model, BEST_MODEL_PATH)
    logger.info("Model saved to %s", BEST_MODEL_PATH)

    with mlflow.start_run(run_name="best_overall") as best_run:
        mlflow.set_tag("best_overall", "true")
        mlflow.set_tag("model_family", best_family)
        mlflow.log_params(best_params)
        mlflow.log_params({"t_major": thresholds["t_major"], "t_minor": thresholds["t_minor"]})

        metrics = evaluate_and_log(model, X_test, y_test, "best_overall", best_params, output_dir=MODELS_DIR)

        composite = 0.4 * metrics["Minor_precision"] + 0.4 * metrics["Major_recall"] + 0.2 * metrics["Moderate_f1"]
        mlflow.log_metric("composite", composite)

        signature = mlflow.models.infer_signature(X_test[:5], model.predict(X_test[:5]))
        mlflow.sklearn.log_model(model, name="model", signature=signature)

        with open("/tmp/model_signature.json", "w") as _f:
            json.dump(signature.to_dict(), _f, indent=2)
        mlflow.log_artifact("/tmp/model_signature.json")

        mlflow.log_artifact(BEST_PARAMS_PATH)
        mlflow.log_artifact(THRESHOLDS_PATH)

    logger.info("MLflow run: %s", best_run.info.run_id)

    register_best_model(
        best_run.info.run_id,
        best_family,
        composite,
        best_trial_number=None,
    )

    logger.info("Training complete — model saved to %s/", MODELS_DIR)


if __name__ == "__main__":
    main()
