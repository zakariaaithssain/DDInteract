"""Tests for model export."""

from unittest.mock import MagicMock, patch

import pandas as pd


class TestExportMain:
    def test_export_with_cached_features(self):
        mock_runs = pd.DataFrame(
            {
                "run_id": ["abc123"],
                "tags.mlflow.runName": ["LogisticRegression_C-0.1"],
                "metrics.macro_f1": [0.85],
            }
        )
        mock_model = MagicMock()
        with (
            patch("src.export_model.Path.exists", return_value=True),
            patch("src.export_model.mlflow.set_experiment"),
            patch("src.export_model.mlflow.search_runs", return_value=mock_runs),
            patch("src.export_model.mlflow.sklearn.load_model", return_value=mock_model),
            patch("src.export_model.joblib.load", return_value=mock_model),
            patch("src.export_model.joblib.dump"),
            patch("src.export_model.Path.mkdir"),
            patch("src.export_model.logger"),
        ):
            from src.export_model import main

            main()

    def test_export_builds_features_when_no_cache(self):
        mock_runs = pd.DataFrame(
            {
                "run_id": ["def456"],
                "tags.mlflow.runName": ["RandomForest_100_8"],
                "metrics.macro_f1": [0.82],
            }
        )

        mock_model = MagicMock()
        with (
            patch("src.export_model.Path.exists", return_value=False),
            patch("src.export_model.mlflow.set_experiment"),
            patch("src.export_model.mlflow.search_runs", return_value=mock_runs),
            patch("src.export_model.mlflow.sklearn.load_model", return_value=mock_model),
            patch("src.export_model.joblib.dump"),
            patch("src.export_model.Path.mkdir"),
            patch("src.export_model.logger"),
        ):
            from src.export_model import main

            main()

    def test_export_saves_model(self):
        mock_runs = pd.DataFrame(
            {
                "run_id": ["ghi789"],
                "tags.mlflow.runName": ["XGBoost"],
                "metrics.macro_f1": [0.88],
            }
        )
        mock_model = MagicMock()
        with (
            patch("src.export_model.Path.exists", return_value=True),
            patch("src.export_model.mlflow.set_experiment"),
            patch("src.export_model.mlflow.search_runs", return_value=mock_runs),
            patch("src.export_model.mlflow.sklearn.load_model", return_value=mock_model),
            patch("src.export_model.joblib.load", return_value=mock_model),
            patch("src.export_model.joblib.dump") as mock_dump,
            patch("src.export_model.Path.mkdir"),
            patch("src.export_model.logger"),
        ):
            from src.export_model import main

            main()

        assert mock_dump.call_count >= 1

    def test_export_handles_no_best_model(self):
        mock_runs = pd.DataFrame(
            {
                "run_id": pd.Series(dtype=str),
                "tags.mlflow.runName": pd.Series(dtype=str),
                "metrics.macro_f1": pd.Series(dtype=float),
            }
        )

        with (
            patch("src.export_model.Path.exists", return_value=False),
            patch("src.export_model.mlflow.set_experiment"),
            patch("src.export_model.mlflow.search_runs", return_value=mock_runs),
            patch("src.export_model.logger"),
        ):
            from src.export_model import main

            main()  # should return early without error

    def test_export_loads_local_best_model_when_available(self):
        mock_model = MagicMock()
        with (
            patch("src.export_model.Path.exists", return_value=True),
            patch("src.export_model.joblib.load", return_value=mock_model),
            patch("src.export_model.Path.mkdir"),
            patch("src.export_model.joblib.dump"),
            patch("src.export_model.logger"),
        ):
            from src.export_model import main

            main()
