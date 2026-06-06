"""Tests for training pipeline functions."""

import sys

sys.path.insert(0, "src")

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from train import (
    REGISTRY_NAME,
    evaluate_and_log,
    load_or_build_features,
    log_confusion_matrix,
    ordinal_mae,
    register_best_model,
)


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
        with patch("train.mlflow"):
            result = evaluate_and_log(model, X, y, "test_run", {"C": 1.0})
        expected_keys = {"accuracy", "macro_f1", "weighted_f1", "kappa", "mae"}
        assert set(result.keys()) == expected_keys

    def test_returns_float_values(self, model_and_data):
        model, X, y = model_and_data
        with patch("train.mlflow"):
            result = evaluate_and_log(model, X, y, "test_run", {"C": 1.0})
        for v in result.values():
            assert isinstance(v, float)

    def test_calls_mlflow_log_metrics(self, model_and_data):
        model, X, y = model_and_data
        with patch("train.mlflow") as mock_mlflow:
            evaluate_and_log(model, X, y, "test_run", {"C": 1.0})
        assert mock_mlflow.log_metrics.called


class TestLogConfusionMatrix:
    def test_creates_plot_and_logs(self):
        cm = np.array([[10, 2, 0], [1, 15, 3], [0, 2, 20]])
        with (
            patch("train.plt") as mock_plt,
            patch("train.mlflow") as mock_mlflow,
            patch("train.os.unlink"),
            patch("train.tempfile.NamedTemporaryFile") as mock_temp,
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
            patch("train.Path.exists", return_value=False),
            patch("train.np.save"),
            patch("train.build_features") as mock_build,
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
            patch("train.Path.exists", return_value=True),
            patch("train.np.load", side_effect=[X_cached, y_cached]),
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
            patch("train.Path.exists", return_value=True),
            patch("train.np.load", side_effect=[X_cached, y_cached]),
            patch("train.np.save"),
            patch("train.build_features") as mock_build,
        ):
            mock_build.return_value = np.zeros((2, 1045), dtype=np.float64)
            X, y = load_or_build_features(df)
        assert X.shape[0] == 2


class TestMain:
    def test_main_runs_pipeline(self):
        import src.train

        mock_df = pd.DataFrame(
            {
                "smiles_a": ["O", "CCO"],
                "smiles_b": ["CCO", "O"],
                "severity_label": [0, 1],
            }
        )
        mock_model_obj = MagicMock()
        mock_model_obj.predict.return_value = np.array([0, 1])
        mock_model_obj.predict_proba.return_value = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])

        def model_constructor(**kwargs):
            return mock_model_obj

        with (
            patch("src.train.os.makedirs"),
            patch("src.train.mlflow.set_experiment"),
            patch("src.train.pd.read_csv", return_value=mock_df),
            patch("src.train.load_or_build_features", return_value=(np.zeros((2, 1045)), np.array([0, 1]))),
            patch.object(src.train, "StandardScaler") as mock_scaler_cls,
            patch.object(src.train, "PCA") as mock_pca_cls,
            patch.object(
                src.train,
                "train_test_split",
                return_value=(
                    np.zeros((1, 1045)),
                    np.zeros((1, 1045)),
                    np.array([0]),
                    np.array([1]),
                ),
            ),
            patch("src.train.joblib.dump"),
            patch("src.train.cross_val_score", return_value=np.array([0.8, 0.9, 0.85])),
            patch(
                "src.train.evaluate_and_log",
                return_value={
                    "accuracy": 0.9,
                    "macro_f1": 0.85,
                    "weighted_f1": 0.88,
                    "kappa": 0.7,
                    "mae": 0.2,
                },
            ),
            patch("src.train.mlflow.models.infer_signature"),
            patch("src.train.mlflow.start_run") as mock_start_run,
            patch("src.train.mlflow.log_artifact"),
            patch("src.train.mlflow.sklearn.autolog"),
            patch("src.train.mlflow.sklearn.log_model"),
            patch("src.train.mlflow.register_model", side_effect=Exception("no registry")),
            patch("src.train.logger"),
            patch("src.train.json.dump"),
            patch.object(
                src.train,
                "MODEL_CONFIGS",
                {
                    "LogisticRegression": {
                        "model": model_constructor,
                        "fixed": {"solver": "lbfgs", "max_iter": 1000, "class_weight": "balanced"},
                        "grid": {"C": [1.0]},
                    },
                },
            ),
            patch.object(src.train, "get_param_grid", return_value=[{"C": 1.0}]),
        ):
            mock_scaler = MagicMock()
            mock_scaler_cls.return_value = mock_scaler
            mock_scaler.fit_transform.return_value = np.zeros((1, 1045))
            mock_scaler.transform.return_value = np.zeros((1, 1045))

            mock_pca = MagicMock()
            mock_pca_cls.return_value = mock_pca
            mock_pca.fit_transform.return_value = np.zeros((1, 50))
            mock_pca.transform.return_value = np.zeros((1, 50))
            mock_pca.explained_variance_ratio_ = np.array([0.1] * 50)

            mock_run = MagicMock()
            mock_run.info.run_id = "test_run_123"
            mock_start_run.return_value.__enter__.return_value = mock_run

            src.train.main()

    def test_main_saves_results_json(self):
        import src.train

        mock_df = pd.DataFrame(
            {
                "smiles_a": ["O"],
                "smiles_b": ["CCO"],
                "severity_label": [0],
            }
        )
        mock_model_obj = MagicMock()
        mock_model_obj.predict.return_value = np.array([0])
        mock_model_obj.predict_proba.return_value = np.array([[0.8, 0.1, 0.1]])

        def model_constructor(**kwargs):
            return mock_model_obj

        with (
            patch("src.train.os.makedirs"),
            patch("src.train.mlflow.set_experiment"),
            patch("src.train.pd.read_csv", return_value=mock_df),
            patch("src.train.load_or_build_features", return_value=(np.zeros((1, 1045)), np.array([0]))),
            patch.object(
                src.train,
                "train_test_split",
                return_value=(
                    np.zeros((1, 1045)),
                    np.zeros((1, 1045)),
                    np.array([0]),
                    np.array([0]),
                ),
            ),
            patch.object(src.train, "StandardScaler") as mock_scaler_cls,
            patch.object(src.train, "PCA") as mock_pca_cls,
            patch("src.train.joblib.dump"),
            patch("src.train.cross_val_score", return_value=np.array([0.9])),
            patch(
                "src.train.evaluate_and_log",
                return_value={
                    "accuracy": 0.9,
                    "macro_f1": 0.85,
                    "weighted_f1": 0.88,
                    "kappa": 0.7,
                    "mae": 0.2,
                },
            ),
            patch("src.train.mlflow.models.infer_signature"),
            patch("src.train.mlflow.start_run") as mock_start_run,
            patch("src.train.mlflow.log_artifact"),
            patch("src.train.mlflow.sklearn.autolog"),
            patch("src.train.mlflow.sklearn.log_model"),
            patch("src.train.mlflow.register_model", side_effect=Exception("no registry")),
            patch("src.train.logger"),
            patch("src.train.json.dump") as mock_json_dump,
            patch.object(
                src.train,
                "MODEL_CONFIGS",
                {
                    "LogisticRegression": {
                        "model": model_constructor,
                        "fixed": {"solver": "lbfgs", "max_iter": 1000, "class_weight": "balanced"},
                        "grid": {"C": [1.0]},
                    },
                },
            ),
            patch.object(src.train, "get_param_grid", return_value=[{"C": 1.0}]),
        ):
            mock_scaler = MagicMock()
            mock_scaler_cls.return_value = mock_scaler
            mock_scaler.fit_transform.return_value = np.zeros((1, 1045))
            mock_scaler.transform.return_value = np.zeros((1, 1045))

            mock_pca = MagicMock()
            mock_pca_cls.return_value = mock_pca
            mock_pca.fit_transform.return_value = np.zeros((1, 50))
            mock_pca.transform.return_value = np.zeros((1, 50))
            mock_pca.explained_variance_ratio_ = np.array([0.1] * 50)

            mock_run = MagicMock()
            mock_run.info.run_id = "test_run_123"
            mock_start_run.return_value.__enter__.return_value = mock_run

            src.train.main()

        mock_json_dump.assert_called_once()


class TestRegisterBestModel:
    def test_calls_mlflow_register(self):
        with (
            patch("train.mlflow.register_model") as mock_register,
            patch("train.mlflow.MlflowClient") as mock_client,
        ):
            mock_version = MagicMock()
            mock_version.version = "42"
            mock_register.return_value = mock_version
            register_best_model("run_123", "LogisticRegression", 0.85)
        mock_register.assert_called_once()
        mock_client.return_value.set_registered_model_alias.assert_called_once_with(REGISTRY_NAME, "production", "42")

    def test_logs_warning_on_failure(self):
        with (
            patch("train.mlflow.register_model", side_effect=Exception("fail")),
            patch("train.logger") as mock_logger,
        ):
            register_best_model("run_123", "LR", 0.85)
        mock_logger.warning.assert_called_once()
