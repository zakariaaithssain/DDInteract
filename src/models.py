from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42

MODEL_GRIDS = {
    "LogisticRegression": {
        "model": LogisticRegression,
        "params": [
            {"C": 0.01, "solver": "lbfgs", "max_iter": 1000, "class_weight": "balanced"},
            {"C": 0.1, "solver": "lbfgs", "max_iter": 1000, "class_weight": "balanced"},
            {"C": 1.0, "solver": "lbfgs", "max_iter": 1000, "class_weight": "balanced"},
        ],
    },
    "LinearSVC": {
        "model": LinearSVC,
        "params": [
            {"C": 0.01, "max_iter": 10000, "class_weight": "balanced"},
            {"C": 0.1, "max_iter": 10000, "class_weight": "balanced"},
            {"C": 1.0, "max_iter": 10000, "class_weight": "balanced"},
        ],
    },
    "RandomForest": {
        "model": RandomForestClassifier,
        "params": [
            {"n_estimators": 100, "max_depth": 8, "class_weight": "balanced"},
            {"n_estimators": 200, "max_depth": 16, "class_weight": "balanced"},
            {"n_estimators": 300, "max_depth": 24, "class_weight": "balanced"},
        ],
    },
    "KNN": {
        "model": KNeighborsClassifier,
        "params": [
            {"n_neighbors": 5},
            {"n_neighbors": 9},
            {"n_neighbors": 15},
        ],
    },
    "XGBoost": {
        "model": XGBClassifier,
        "params": [
            {
                "objective": "multi:softprob", "num_class": 3,
                "max_depth": 4, "reg_lambda": 1.0, "reg_alpha": 0.5,
                "learning_rate": 0.1, "n_estimators": 100,
            },
            {
                "objective": "multi:softprob", "num_class": 3,
                "max_depth": 6, "reg_lambda": 2.0, "reg_alpha": 1.0,
                "learning_rate": 0.1, "n_estimators": 200,
            },
            {
                "objective": "multi:softprob", "num_class": 3,
                "max_depth": 8, "reg_lambda": 4.0, "reg_alpha": 2.0,
                "learning_rate": 0.05, "n_estimators": 300,
            },
        ],
    },
}
