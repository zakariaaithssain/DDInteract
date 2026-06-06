from collections.abc import Sequence
from itertools import product
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

RANDOM_STATE: int = 42


def expand_grid(fixed: dict[str, Any], grid: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Expand a fixed parameter dict with a grid of hyperparameters.

    Produces the Cartesian product of all grid values, each merged with
    the fixed parameters.

    Args:
        fixed: Parameters that stay constant across all combinations.
        grid: Dict mapping parameter names to lists of values to search.

    Returns:
        List of parameter dicts representing all combinations.
    """
    keys = grid.keys()
    return [{**fixed, **dict(zip(keys, vals))} for vals in product(*grid.values())]


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "LogisticRegression": {
        "model": LogisticRegression,
        "fixed": {"solver": "lbfgs", "max_iter": 1000, "class_weight": "balanced"},
        "grid": {"C": [0.01, 0.1, 1.0]},
    },
    "LinearSVC": {
        "model": LinearSVC,
        "fixed": {"max_iter": 10000, "class_weight": "balanced"},
        "grid": {"C": [0.01, 0.1, 1.0]},
    },
    "RandomForest": {
        "model": RandomForestClassifier,
        "fixed": {"class_weight": "balanced"},
        "grid": {"n_estimators": [100, 200, 300], "max_depth": [8, 16, 24]},
    },
    "KNN": {
        "model": KNeighborsClassifier,
        "fixed": {},
        "grid": {"n_neighbors": [5, 9, 15]},
    },
    "XGBoost": {
        "model": XGBClassifier,
        "params": [
            {
                "objective": "multi:softprob",
                "num_class": 3,
                "max_depth": 4,
                "reg_lambda": 1.0,
                "reg_alpha": 0.5,
                "learning_rate": 0.1,
                "n_estimators": 100,
            },
            {
                "objective": "multi:softprob",
                "num_class": 3,
                "max_depth": 6,
                "reg_lambda": 2.0,
                "reg_alpha": 1.0,
                "learning_rate": 0.1,
                "n_estimators": 200,
            },
            {
                "objective": "multi:softprob",
                "num_class": 3,
                "max_depth": 8,
                "reg_lambda": 4.0,
                "reg_alpha": 2.0,
                "learning_rate": 0.05,
                "n_estimators": 300,
            },
        ],
    },
}


def get_param_grid(family: str) -> list[dict[str, Any]]:
    """Get the list of hyperparameter dicts for a given model family.

    Args:
        family: Model family name key in MODEL_CONFIGS.

    Returns:
        List of parameter dicts to iterate over during training.

    Raises:
        KeyError: If the family is not in MODEL_CONFIGS.
    """
    cfg = MODEL_CONFIGS[family]
    if "params" in cfg:
        return cfg["params"]
    return expand_grid(cfg["fixed"], cfg["grid"])
