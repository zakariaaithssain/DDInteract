"""Tests for the FastAPI prediction API."""

from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models import RANDOM_STATE


def _build_dummy_artifacts():
    rng = np.random.default_rng(42)
    X_dummy = rng.uniform(size=(100, 1045)).astype(np.float64)
    y_dummy = rng.integers(0, 3, size=100)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_dummy)
    pca = PCA(n_components=50, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_s)
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_pca, y_dummy)
    return scaler, pca, model


def _fake_reference_stats() -> dict:
    return {
        "fp_density_mean": 0.5,
        "fp_density_std": 0.1,
        "fp_density_p5": 0.3,
        "fp_density_p95": 0.7,
        "n_samples": 500,
        "reference_features": {
            "fp_a_density": [0.5] * 500,
            "fp_b_density": [0.5] * 500,
            "fp_diff_mean": [0.2] * 500,
            "fp_product_mean": [0.3] * 500,
            "tanimoto": [0.5] * 500,
        },
        "reference_densities": [0.5] * 500,
    }


@pytest.fixture(scope="module")
def client():
    scaler, pca, model = _build_dummy_artifacts()
    with patch("src.api.joblib.load", side_effect=[model, scaler, pca]):
        from src import api

        with TestClient(api.app) as c:
            yield c


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestIndex:
    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")


class TestPredict:
    def test_predict_valid_smiles(self, client):
        resp = client.post(
            "/predict",
            json={
                "smiles_a": "O",
                "smiles_b": "CCO",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["smiles_a"] == "O"
        assert data["smiles_b"] == "CCO"
        assert data["predicted_severity"] in ("Minor", "Moderate", "Major")
        assert set(data["probabilities"].keys()) == {"Minor", "Moderate", "Major"}
        for v in data["probabilities"].values():
            assert 0.0 <= v <= 1.0
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_probabilities_sum_to_one(self, client):
        resp = client.post(
            "/predict",
            json={
                "smiles_a": "c1ccccc1",
                "smiles_b": "CC(=O)O",
            },
        )
        data = resp.json()
        total = sum(data["probabilities"].values())
        assert abs(total - 1.0) < 0.01

    def test_predict_missing_field(self, client):
        resp = client.post("/predict", json={"smiles_a": "O"})
        assert resp.status_code == 422

    def test_predict_empty_strings(self, client):
        resp = client.post("/predict", json={"smiles_a": "", "smiles_b": ""})
        assert resp.status_code == 200


class TestDriftMonitoring:
    def test_drift_endpoint_returns_inactive_when_no_reference(self, client):
        from src import api

        api.reference_stats = None
        api.feature_buffer = []
        api.last_drift_result = None

        resp = client.get("/drift")
        assert resp.status_code == 200
        data = resp.json()
        assert data["monitoring_active"] is False
        assert data["samples_collected"] == 0
        assert data["last_result"] is None

    def test_drift_endpoint_shows_active_monitoring(self, client):
        from src import api

        api.reference_stats = _fake_reference_stats()
        api.feature_buffer = []
        api.last_drift_result = None

        resp = client.get("/drift")
        assert resp.status_code == 200
        data = resp.json()
        assert data["monitoring_active"] is True
        assert data["samples_collected"] == 0

    def test_predictions_accumulate_in_buffer(self, client):
        from src import api

        api.reference_stats = _fake_reference_stats()
        api.feature_buffer = []
        api.last_drift_result = None

        client.post("/predict", json={"smiles_a": "O", "smiles_b": "CCO"})
        assert len(api.feature_buffer) == 1

        client.post("/predict", json={"smiles_a": "CC", "smiles_b": "CO"})
        assert len(api.feature_buffer) == 2

    def test_drift_check_runs_at_interval(self, client, monkeypatch):
        from src import api

        api.reference_stats = _fake_reference_stats()
        api.feature_buffer = []
        api.last_drift_result = None
        monkeypatch.setattr("src.api.DRIFT_CHECK_INTERVAL", 2)

        client.post("/predict", json={"smiles_a": "O", "smiles_b": "CCO"})
        assert len(api.feature_buffer) == 1
        assert api.last_drift_result is None

        # second prediction hits the threshold
        client.post("/predict", json={"smiles_a": "CC", "smiles_b": "CO"})
        assert len(api.feature_buffer) == 0  # buffer reset
        assert api.last_drift_result is not None
