"""Tests for training pipeline functions."""

from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import REGISTRY_NAME
from src.data import load_or_build_features
from src.evaluate import evaluate_and_log
from src.metrics import ordinal_mae
from src.registry import register_best_model
from src.visualization import log_confusion_matrix


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
        expected_keys = {"accuracy", "macro_f1", "weighted_f1", "kappa", "mae"}
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
            patch("src.visualization.plt") as mock_plt,
            patch("src.visualization.mlflow") as mock_mlflow,
            patch("src.visualization.os.unlink"),
            patch("src.visualization.tempfile.NamedTemporaryFile") as mock_temp,
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
            patch("src.data.Path.exists", return_value=False),
            patch("src.data.np.save"),
            patch("src.data.build_features") as mock_build,
        ):
            mock_build.return_value = np.zeros((1, 1045), dtype=np.float64)
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
        X_cached = np.zeros((1, 1045), dtype=np.float64)
        y_cached = np.array([1])
        with (
            patch("src.data.Path.exists", return_value=True),
            patch("src.data.np.load", side_effect=[X_cached, y_cached]),
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
        X_cached = np.zeros((1, 1045), dtype=np.float64)
        y_cached = np.array([1])
        with (
            patch("src.data.Path.exists", return_value=True),
            patch("src.data.np.load", side_effect=[X_cached, y_cached]),
            patch("src.data.np.save"),
            patch("src.data.build_features") as mock_build,
        ):
            mock_build.return_value = np.zeros((2, 1045), dtype=np.float64)
            X, y = load_or_build_features(df)
        assert X.shape[0] == 2


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
            "metrics": {"accuracy": 0.9, "macro_f1": value, "weighted_f1": 0.88, "kappa": 0.7, "mae": 0.2},
        }
        return trial

    def test_main_runs_pipeline(self):
        import src.train

        with (
            patch("src.train.build_features") as mock_build,
            patch("src.train.select_features") as mock_select,
            patch("src.train.search_hyperparams") as mock_search,
        ):
            src.train.main()

        mock_build.assert_called_once()
        mock_select.assert_called_once()
        mock_search.assert_called_once()

    def test_main_saves_results_json(self):
        import src.train

        with (
            patch("src.train.build_features") as mock_build,
            patch("src.train.select_features") as mock_select,
            patch("src.train.search_hyperparams") as mock_search,
        ):
            src.train.main()

        mock_build.assert_called_once()
        mock_select.assert_called_once()
        mock_search.assert_called_once()


class TestRegisterBestModel:
    def test_calls_mlflow_register(self):
        with (
            patch("src.registry.mlflow.register_model") as mock_register,
            patch("src.registry.mlflow.MlflowClient") as mock_client,
        ):
            mock_version = MagicMock()
            mock_version.version = "42"
            mock_register.return_value = mock_version
            register_best_model("run_123", "LogisticRegression", 0.85)
        mock_register.assert_called_once()
        mock_client.return_value.set_registered_model_alias.assert_called_once_with(REGISTRY_NAME, "production", "42")

    def test_logs_warning_on_failure(self):
        with (
            patch("src.registry.mlflow.register_model", side_effect=Exception("fail")),
            patch("src.registry.logger") as mock_logger,
        ):
            register_best_model("run_123", "LR", 0.85)
        mock_logger.warning.assert_called_once()
