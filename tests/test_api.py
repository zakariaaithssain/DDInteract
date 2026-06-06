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
