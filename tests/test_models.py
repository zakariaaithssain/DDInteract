"""Tests for model definitions and hyperparameter search spaces."""

import optuna

from src.search_hyperparams import RANDOM_STATE, suggest_rf_params, suggest_xgb_params


def _make_trial() -> optuna.Trial:
    study = optuna.create_study(direction="maximize")
    return study.ask()


class TestSuggestRfParams:
    def test_returns_all_keys(self):
        trial = _make_trial()
        params = suggest_rf_params(trial)
        expected_keys = {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "class_weight",
            "random_state",
        }
        assert set(params.keys()) == expected_keys

    def test_random_state_is_fixed(self):
        trial = _make_trial()
        params = suggest_rf_params(trial)
        assert params["random_state"] == RANDOM_STATE

    def test_class_weight_is_balanced(self):
        trial = _make_trial()
        params = suggest_rf_params(trial)
        assert params["class_weight"] == "balanced"

    def test_values_are_in_bounds(self):
        trial = _make_trial()
        params = suggest_rf_params(trial)
        assert 100 <= params["n_estimators"] <= 500
        assert 4 <= params["max_depth"] <= 32
        assert 2 <= params["min_samples_split"] <= 20
        assert 1 <= params["min_samples_leaf"] <= 20


class TestSuggestXgbParams:
    def test_returns_all_keys(self):
        trial = _make_trial()
        params = suggest_xgb_params(trial)
        expected_keys = {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "reg_lambda",
            "reg_alpha",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "objective",
            "num_class",
            "random_state",
        }
        assert set(params.keys()) == expected_keys

    def test_fixed_params(self):
        trial = _make_trial()
        params = suggest_xgb_params(trial)
        assert params["objective"] == "multi:softprob"
        assert params["num_class"] == 3
        assert params["random_state"] == RANDOM_STATE

    def test_values_are_in_bounds(self):
        trial = _make_trial()
        params = suggest_xgb_params(trial)
        assert 100 <= params["n_estimators"] <= 500
        assert 3 <= params["max_depth"] <= 12
        assert 0.01 <= params["learning_rate"] <= 0.3
        assert 0.1 <= params["reg_lambda"] <= 10.0
        assert 0.0 <= params["reg_alpha"] <= 5.0
        assert 0.6 <= params["subsample"] <= 1.0
        assert 0.6 <= params["colsample_bytree"] <= 1.0
        assert 1 <= params["min_child_weight"] <= 10
