"""Step 2: Optuna TPE hyperparameter search.

Loads cached features, runs trials of RandomForest / XGBoost
on a 20 % stratified subsample with dual pruning, and saves
the best hyperparameters to disk.
"""

import json
import os
from typing import Any

import mlflow
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from src.config import (
    BEST_PARAMS_PATH,
    EXPERIMENT_NAME,
    FEATURE_CACHE,
    LABEL_CACHE,
    LOGS_DIR,
    MODELS_DIR,
    TEST_SIZE,
    TRIALS_LEADERBOARD_PATH,
    logger,
)
from src.evaluate import evaluate_and_log
from src.features import N_BITS

RANDOM_STATE: int = 42
N_TRIALS = 50


def suggest_rf_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("rf_n_estimators", 100, 300, log=True),
        "max_depth": trial.suggest_int("rf_max_depth", 4, 32),
        "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None]),
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    }


def suggest_xgb_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 300, log=True),
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 12),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True),
        "reg_lambda": trial.suggest_float("xgb_reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("xgb_reg_alpha", 0.0, 5.0),
        "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 10),
        "objective": "multi:softprob",
        "num_class": 3,
        "random_state": RANDOM_STATE,
    }


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    skip = {"objective", "num_class", "random_state", "class_weight", "verbosity"}
    return {k: v for k, v in params.items() if k not in skip}


def _reconstruct_params(trial_params: dict[str, Any], family: str) -> dict[str, Any]:
    prefix = "rf_" if family == "RandomForest" else "xgb_"
    return {k.removeprefix(prefix): v for k, v in trial_params.items() if k.startswith(prefix)}


def objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_features_raw: int,
    train_size: int,
    test_size: int,
) -> float:
    mlflow.set_experiment(EXPERIMENT_NAME)

    family = trial.suggest_categorical("model_family", ["RandomForest", "XGBoost"])

    if family == "RandomForest":
        params = suggest_rf_params(trial)

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        fold_scores: list[float] = []
        for step, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]

            fold_model = RandomForestClassifier(**params)
            fold_model.fit(X_fold_train, y_fold_train)
            preds = fold_model.predict(X_fold_val)
            fold_f1 = f1_score(y_fold_val, preds, average="macro")

            fold_scores.append(fold_f1)
            trial.report(fold_f1, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        cv_mean = float(np.mean(fold_scores))
        cv_std = float(np.std(fold_scores))

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        estimator_type = "sklearn.ensemble.RandomForestClassifier"
    else:
        params = suggest_xgb_params(trial)
        model = XGBClassifier(**params, tree_method="hist", n_jobs=-1, verbosity=0)
        cv_scores = cross_val_score(model, X_train, y_train, cv=3)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        try:
            from optuna_integration.xgboost import XGBoostPruningCallback

            pruning_callback = XGBoostPruningCallback(trial, "validation_0-mlogloss")
            model.set_params(callbacks=[pruning_callback])
        except ImportError:
            pass
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        estimator_type = "xgboost.XGBClassifier"

    run_name = f"{family}_trial_{trial.number}"
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        mlflow.set_tag("model_family", family)
        mlflow.set_tag("trial_number", trial.number)
        mlflow.set_tag("estimator_type", estimator_type)

        mlflow.log_params(
            {
                "model_family": family,
                "n_bits": N_BITS,
                "n_features": n_features_raw,
                "dataset_source": FEATURE_CACHE,
                "train_size": train_size,
                "test_size": test_size,
            }
        )
        mlflow.log_params(_clean_params(params))

        mlflow.log_metric("cv_mean", cv_mean)
        mlflow.log_metric("cv_std", cv_std)

        metrics = evaluate_and_log(model, X_test, y_test, run_name, _clean_params(params))

        trial.set_user_attr("run_id", run.info.run_id)

    trial.set_user_attr("family", family)
    trial.set_user_attr("metrics", metrics)

    return metrics["macro_f1"]


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    mlflow.set_experiment(EXPERIMENT_NAME)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is not None:
        client.set_experiment_tag(experiment.experiment_id, "project", "DDI_Severity_Predictor")
        client.set_experiment_tag(experiment.experiment_id, "task", "multi_class_classification")
        client.set_experiment_tag(experiment.experiment_id, "metrics_primary", "macro_f1")
        client.set_experiment_tag(experiment.experiment_id, "model_families", "RandomForest,XGBoost")

    X = np.load(FEATURE_CACHE)
    y = np.load(LABEL_CACHE)
    logger.info("Loaded features: %s, labels: %s", X.shape, y.shape)

    X_search, _, y_search, _ = train_test_split(X, y, test_size=0.8, random_state=RANDOM_STATE, stratify=y)
    logger.info("Search subsample (20%%): %s", X_search.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X_search, y_search, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_search
    )
    logger.info("Search train/test split: %d train, %d test", len(X_train), len(X_test))

    n_features_raw = X_train.shape[1]
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner()

    storage = f"sqlite:///{MODELS_DIR}/optuna_study.db"
    study = optuna.create_study(
        study_name="DDI_Severity_TPE",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_test, y_test, n_features_raw, train_size, test_size),
        n_trials=N_TRIALS,
        n_jobs=-1,
    )

    all_results: list[dict[str, Any]] = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None:
            attrs = t.user_attrs
            all_results.append(
                {
                    "trial_number": t.number,
                    "family": attrs.get("family"),
                    **attrs.get("metrics", {}),
                }
            )

    best_trial = study.best_trial
    best_family: str = best_trial.user_attrs["family"]
    assert best_trial.value is not None
    best_macro_f1: float = best_trial.value
    best_params = _reconstruct_params(dict(best_trial.params), best_family)

    logger.info("Best trial: %s (macro_f1=%.4f) — %s", best_trial.number, best_macro_f1, best_family)

    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump({"family": best_family, **best_params}, f, indent=2)
    logger.info("Best params saved to %s", BEST_PARAMS_PATH)

    results_summary = sorted(all_results, key=lambda x: -x.get("macro_f1", 0))

    logger.info("--- Results (sorted by macro F1) ---")
    for r in results_summary:
        logger.info(
            "Trial %-3d %-15s  acc=%.4f  macro_f1=%.4f  qwk=%.4f  mae=%.4f",
            r.get("trial_number", -1),
            r.get("family", "?"),
            r.get("accuracy", 0),
            r.get("macro_f1", 0),
            r.get("qwk", 0),
            r.get("mae", 0),
        )

    with open(TRIALS_LEADERBOARD_PATH, "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info("Trials leaderboard saved to %s", TRIALS_LEADERBOARD_PATH)


if __name__ == "__main__":
    main()
