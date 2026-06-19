"""Tests for training pipeline functions."""

from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.build_features import load_or_build_features
from src.config import REGISTRY_NAME
from src.evaluate import evaluate_and_log, log_confusion_matrix, ordinal_mae
from src.export_model import register_best_model
from src.search_hyperparams import _clean_params, _reconstruct_params


class TestOrdinalMae:
    def test_perfect_prediction(self):
        y_true = np.array([0, 1, 2, 1, 0])
        y_pred = np.array([0, 1, 2, 1, 0])
        assert ordinal_mae(y_true, y_pred) == 0.0

    def test_off_by_one(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([1, 2, 1])
        result = ordinal_mae(y_true, y_pred)
        assert result == 1.0

    def test_off_by_two(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([2, 2, 2])
        assert ordinal_mae(y_true, y_pred) == 2.0

    def test_empty_arrays(self):
        y_true = np.array([])
        y_pred = np.array([])
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            result = ordinal_mae(y_true, y_pred)
        assert np.isnan(result)


class TestEvaluateAndLog:
    @pytest.fixture
    def model_and_data(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 10))
        y = rng.integers(0, 3, size=50)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        return model, X, y

    def test_returns_expected_keys(self, model_and_data):
        model, X, y = model_and_data
        with patch("src.evaluate.mlflow"):
            result = evaluate_and_log(model, X, y, "test_run", {"C": 1.0})
        expected_keys = {"accuracy", "macro_f1", "weighted_f1", "qwk", "mae"}
        assert set(result.keys()) == expected_keys

    def test_returns_float_values(self, model_and_data):
        model, X, y = model_and_data
        with patch("src.evaluate.mlflow"):
            result = evaluate_and_log(model, X, y, "test_run", {"C": 1.0})
        for v in result.values():
            assert isinstance(v, float)

    def test_calls_mlflow_log_metrics(self, model_and_data):
        model, X, y = model_and_data
        with patch("src.evaluate.mlflow") as mock_mlflow:
            evaluate_and_log(model, X, y, "test_run", {"C": 1.0})
        assert mock_mlflow.log_metrics.called


class TestLogConfusionMatrix:
    def test_creates_plot_and_logs(self):
        cm = np.array([[10, 2, 0], [1, 15, 3], [0, 2, 20]])
        with (
            patch("src.evaluate.plt") as mock_plt,
            patch("src.evaluate.mlflow") as mock_mlflow,
            patch("src.evaluate.os.unlink"),
            patch("src.evaluate.tempfile.NamedTemporaryFile") as mock_temp,
        ):
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            mock_file = MagicMock()
            mock_file.name = "/tmp/test.png"
            mock_temp.return_value.__enter__.return_value = mock_file
            log_confusion_matrix(cm, "test_run", "{'C': 1.0}")
        mock_plt.subplots.assert_called_once()
        mock_mlflow.log_artifact.assert_called_once()


class TestLoadOrBuildFeatures:
    def test_returns_tuple_of_arrays(self):
        df = pd.DataFrame(
            {
                "smiles_a": ["O"],
                "smiles_b": ["CCO"],
                "severity_label": [1],
            }
        )
        with (
            patch("src.build_features.Path.exists", return_value=False),
            patch("src.build_features.np.save"),
            patch("src.build_features._build_features") as mock_build,
        ):
            mock_build.return_value = np.zeros((1, 533), dtype=np.float64)
            X, y = load_or_build_features(df)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0] == 1

    def test_loads_from_cache_when_available(self):
        df = pd.DataFrame(
            {
                "smiles_a": ["O"],
                "smiles_b": ["CCO"],
                "severity_label": [1],
            }
        )
        X_cached = np.zeros((1, 533), dtype=np.float64)
        y_cached = np.array([1])
        with (
            patch("src.build_features.Path.exists", return_value=True),
            patch("src.build_features.np.load", side_effect=[X_cached, y_cached]),
        ):
            X, y = load_or_build_features(df)
        assert np.array_equal(X, X_cached)
        assert np.array_equal(y, y_cached)

    def test_rebuilds_on_cache_mismatch(self):
        df = pd.DataFrame(
            {
                "smiles_a": ["O", "CCO"],
                "smiles_b": ["CCO", "O"],
                "severity_label": [0, 1],
            }
        )
        X_cached = np.zeros((1, 533), dtype=np.float64)
        y_cached = np.array([1])
        with (
            patch("src.build_features.Path.exists", return_value=True),
            patch("src.build_features.np.load", side_effect=[X_cached, y_cached]),
            patch("src.build_features.np.save"),
            patch("src.build_features._build_features") as mock_build,
        ):
            mock_build.return_value = np.zeros((2, 533), dtype=np.float64)
            X, y = load_or_build_features(df)
        assert X.shape[0] == 2


class TestSearchHyperparamsHelpers:
    def test_clean_params_removes_skipped_keys(self):
        params = {"n_estimators": 100, "objective": "multi:softprob", "random_state": 42, "class_weight": "balanced"}
        result = _clean_params(params)
        assert "n_estimators" in result
        assert "objective" not in result
        assert "random_state" not in result
        assert "class_weight" not in result

    def test_clean_params_preserves_other_keys(self):
        params = {"max_depth": 8, "learning_rate": 0.1, "subsample": 0.8}
        result = _clean_params(params)
        assert result == params

    def test_reconstruct_params_rf(self):
        trial_params = {"rf_n_estimators": 100, "rf_max_depth": 8, "xgb_learning_rate": 0.1}
        result = _reconstruct_params(trial_params, "RandomForest")
        assert result == {"n_estimators": 100, "max_depth": 8}
        assert "learning_rate" not in result

    def test_reconstruct_params_xgb(self):
        trial_params = {"rf_n_estimators": 100, "xgb_learning_rate": 0.1, "xgb_max_depth": 6}
        result = _reconstruct_params(trial_params, "XGBoost")
        assert result == {"learning_rate": 0.1, "max_depth": 6}
        assert "n_estimators" not in result


class TestObjective:
    @pytest.fixture
    def trial_and_data(self):
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 10))
        y = rng.integers(0, 3, size=50)
        X_train, X_test = X[:40], X[10:]
        y_train, y_test = y[:40], y[10:]
        return trial, X_train, y_train, X_test, y_test

    def test_objective_random_forest(self, trial_and_data):
        trial, X_train, y_train, X_test, y_test = trial_and_data
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros(len(X_test))

        with (
            patch("src.search_hyperparams.RandomForestClassifier", return_value=mock_model),
            patch("src.search_hyperparams.cross_val_score", return_value=np.array([0.8, 0.82, 0.81])),
            patch("src.search_hyperparams.evaluate_and_log") as mock_eval,
            patch("src.search_hyperparams.mlflow"),
        ):
            mock_eval.return_value = {"macro_f1": 0.85, "accuracy": 0.84, "weighted_f1": 0.83, "qwk": 0.7, "mae": 0.2}
            result = _objective_with_family(trial, X_train, y_train, X_test, y_test, 10, "RandomForest")

        assert result == 0.85
        mock_model.fit.assert_called_once_with(X_train, y_train)

    def test_objective_xgboost(self, trial_and_data):
        trial, X_train, y_train, X_test, y_test = trial_and_data
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros(len(X_test))

        with (
            patch("src.search_hyperparams.XGBClassifier", return_value=mock_model),
            patch("src.search_hyperparams.cross_val_score", return_value=np.array([0.8, 0.82, 0.81])),
            patch("src.search_hyperparams.evaluate_and_log") as mock_eval,
            patch("src.search_hyperparams.mlflow"),
        ):
            mock_eval.return_value = {"macro_f1": 0.82, "accuracy": 0.80, "weighted_f1": 0.81, "qwk": 0.65, "mae": 0.25}
            result = _objective_with_family(trial, X_train, y_train, X_test, y_test, 10, "XGBoost")

        assert result == 0.82
        mock_model.fit.assert_called_once()


class TestSearchHyperparamsMain:
    def _make_mock_trial(self, number: int, value: float, family: str = "RandomForest") -> MagicMock:
        trial = MagicMock(spec=optuna.trial.FrozenTrial)
        trial.number = number
        trial.value = value
        trial.state = optuna.trial.TrialState.COMPLETE
        trial.user_attrs = {
            "model": MagicMock(),
            "family": family,
            "params": {"n_estimators": 100, "max_depth": 8},
            "run_id": f"run_{number}",
            "metrics": {"accuracy": 0.9, "macro_f1": value, "weighted_f1": 0.88, "qwk": 0.7, "mae": 0.2},
        }
        trial.params = {"rf_n_estimators": 100, "rf_max_depth": 8}
        return trial

    def test_main_runs_study_and_saves_results(self, tmp_path):
        mock_study = MagicMock()
        mock_trial = self._make_mock_trial(5, 0.85)
        mock_study.trials = [mock_trial]
        mock_study.best_trial = mock_trial

        X = np.zeros((100, 10))
        y = np.zeros(100, dtype=int)

        with (
            patch("src.search_hyperparams.np.load", side_effect=[X, y]),
            patch("src.search_hyperparams.train_test_split", return_value=(X[:80], X[80:], y[:80], y[80:])),
            patch("src.search_hyperparams.optuna.create_study", return_value=mock_study),
            patch("src.search_hyperparams.RandomForestClassifier"),
            patch("src.search_hyperparams.joblib.dump"),
            patch("src.search_hyperparams.json.dump"),
            patch("src.search_hyperparams.mlflow"),
            patch("src.search_hyperparams.register_best_model"),
            patch("src.search_hyperparams.logger"),
            patch("src.search_hyperparams.os.makedirs"),
        ):
            import src.search_hyperparams as shp

            shp.main()

        mock_study.optimize.assert_called_once()


def _objective_with_family(trial, x_train, y_train, x_test, y_test, n_features_raw, family):
    """Helper that calls the real objective() but forces the model_family suggestion."""
    original_suggest = trial.suggest_categorical

    def patched_suggest(name, choices):
        if name == "model_family":
            return family
        return original_suggest(name, choices)

    from src.search_hyperparams import objective

    with patch.object(trial, "suggest_categorical", patched_suggest):
        return objective(trial, x_train, y_train, x_test, y_test, n_features_raw)


class TestMain:
    def _make_mock_trial(self, number: int, value: float) -> MagicMock:
        trial = MagicMock(spec=optuna.trial.FrozenTrial)
        trial.number = number
        trial.value = value
        trial.state = optuna.trial.TrialState.COMPLETE
        trial.user_attrs = {
            "model": MagicMock(),
            "family": "RandomForest",
            "params": {"n_estimators": 100, "max_depth": 8},
            "run_id": f"run_{number}",
            "metrics": {"accuracy": 0.9, "macro_f1": value, "weighted_f1": 0.88, "qwk": 0.7, "mae": 0.2},
        }
        return trial

    def test_main_runs_pipeline(self):
        import src.train

        with (
            patch("src.train.build_features") as mock_build,
            patch("src.train.search_hyperparams") as mock_search,
        ):
            src.train.main()

        mock_build.assert_called_once()
        mock_search.assert_called_once()

    def test_main_saves_results_json(self):
        import src.train

        with (
            patch("src.train.build_features") as mock_build,
            patch("src.train.search_hyperparams") as mock_search,
        ):
            src.train.main()

        mock_build.assert_called_once()
        mock_search.assert_called_once()


class TestRegisterBestModel:
    def test_calls_mlflow_register(self):
        with (
            patch("src.export_model.mlflow.register_model") as mock_register,
            patch("src.export_model.mlflow.MlflowClient") as mock_client,
        ):
            mock_version = MagicMock()
            mock_version.version = "42"
            mock_register.return_value = mock_version
            register_best_model("run_123", "LogisticRegression", 0.85)
        mock_register.assert_called_once()
        mock_client.return_value.set_registered_model_alias.assert_called_once_with(REGISTRY_NAME, "production", "42")

    def test_logs_warning_on_failure(self):
        with (
            patch("src.export_model.mlflow.register_model", side_effect=Exception("fail")),
            patch("src.export_model.logger") as mock_logger,
        ):
            register_best_model("run_123", "LR", 0.85)
        mock_logger.warning.assert_called_once()
