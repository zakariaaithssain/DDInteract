from typing import Any

import optuna

RANDOM_STATE: int = 42


def suggest_rf_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("rf_n_estimators", 100, 500, log=True),
        "max_depth": trial.suggest_int("rf_max_depth", 4, 32),
        "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None]),
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    }


def suggest_xgb_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 500, log=True),
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
