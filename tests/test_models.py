"""Tests for model definitions and hyperparameter grids."""

import sys

sys.path.insert(0, "src")

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from models import MODEL_CONFIGS, expand_grid, get_param_grid


class TestExpandGrid:
    def test_expand_single_param(self):
        fixed = {"solver": "lbfgs"}
        grid = {"C": [0.1, 1.0]}
        result = expand_grid(fixed, grid)
        assert len(result) == 2
        assert result[0] == {"solver": "lbfgs", "C": 0.1}
        assert result[1] == {"solver": "lbfgs", "C": 1.0}

    def test_expand_multiple_params(self):
        fixed = {"class_weight": "balanced"}
        grid = {"n_estimators": [10, 50], "max_depth": [3, 5]}
        result = expand_grid(fixed, grid)
        assert len(result) == 4
        for r in result:
            assert r["class_weight"] == "balanced"

    def test_expand_no_fixed(self):
        grid = {"k": [1, 2]}
        result = expand_grid({}, grid)
        assert len(result) == 2

    def test_expand_empty_grid(self):
        result = expand_grid({"a": 1}, {})
        assert result == [{"a": 1}]


class TestGetParamGrid:
    def test_logistic_regression(self):
        params = get_param_grid("LogisticRegression")
        assert len(params) == 3
        for p in params:
            assert p["solver"] == "lbfgs"
            assert p["C"] in [0.01, 0.1, 1.0]

    def test_linear_svc(self):
        params = get_param_grid("LinearSVC")
        assert len(params) == 3
        for p in params:
            assert p["C"] in [0.01, 0.1, 1.0]

    def test_random_forest(self):
        params = get_param_grid("RandomForest")
        assert len(params) == 9
        for p in params:
            assert p["n_estimators"] in [100, 200, 300]
            assert p["max_depth"] in [8, 16, 24]

    def test_knn(self):
        params = get_param_grid("KNN")
        assert len(params) == 3
        for p in params:
            assert p["n_neighbors"] in [5, 9, 15]

    def test_xgboost_explicit_params(self):
        params = get_param_grid("XGBoost")
        assert len(params) == 3
        for p in params:
            assert p["objective"] == "multi:softprob"
            assert p["num_class"] == 3

    def test_invalid_family(self):
        with pytest.raises(KeyError):
            get_param_grid("NonExistent")


class TestModelConfigs:
    def test_all_families_have_model_class(self):
        for name, cfg in MODEL_CONFIGS.items():
            assert issubclass(
                cfg["model"],
                (
                    LogisticRegression,
                    LinearSVC,
                    RandomForestClassifier,
                    KNeighborsClassifier,
                    XGBClassifier,
                ),
            )

    def test_all_families_have_grid_or_explicit_params(self):
        for name, cfg in MODEL_CONFIGS.items():
            assert "fixed" in cfg or "params" in cfg
