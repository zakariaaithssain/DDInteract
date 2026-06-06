import sys

sys.path.insert(0, "src")

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from numpy.typing import NDArray
from pydantic import BaseModel
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from features import build_features
from logger import logger

CLASS_NAMES: list[str] = ["Minor", "Moderate", "Major"]

app = FastAPI(title="DDI Severity Predictor")
model: BaseEstimator | None = None
scaler: StandardScaler | None = None
pca: PCA | None = None


class PredictRequest(BaseModel):
    """Prediction request containing a pair of SMILES strings."""

    smiles_a: str
    smiles_b: str


class PredictResponse(BaseModel):
    """Prediction response with severity class and probabilities."""

    smiles_a: str
    smiles_b: str
    predicted_severity: str
    probabilities: dict[str, float]
    confidence: float


@app.on_event("startup")
def load_artifacts() -> None:
    """Load the trained model, scaler, and PCA from disk on startup."""
    global model, scaler, pca
    logger.info("Loading model artifacts")
    model = joblib.load("models/model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    pca = joblib.load("models/pca.joblib")
    logger.info("Artifacts loaded successfully")


@app.get("/")
def index() -> FileResponse:
    """Serve the frontend HTML page."""
    return FileResponse("src/static/index.html")


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Predict DDI severity for a pair of SMILES strings.

    Args:
        req: PredictRequest with smiles_a and smiles_b.

    Returns:
        PredictResponse with predicted severity and probabilities.
    """
    df = pd.DataFrame({"smiles_a": [req.smiles_a], "smiles_b": [req.smiles_b]})
    assert scaler is not None and pca is not None and model is not None
    X: NDArray[np.float64] = build_features(df)
    X_s: NDArray[np.float64] = scaler.transform(X)
    X_pca: NDArray[np.float64] = pca.transform(X_s)

    probs: NDArray[np.float64] = model.predict_proba(X_pca)[0]
    label_idx: int = int(np.argmax(probs))
    prob_dict: dict[str, float] = {cls: round(float(p), 4) for cls, p in zip(CLASS_NAMES, probs)}

    logger.info(
        "Prediction: %s + %s → %s (conf=%.2f)",
        req.smiles_a,
        req.smiles_b,
        CLASS_NAMES[label_idx],
        probs[label_idx],
    )

    return PredictResponse(
        smiles_a=req.smiles_a,
        smiles_b=req.smiles_b,
        predicted_severity=CLASS_NAMES[label_idx],
        probabilities=prob_dict,
        confidence=round(float(probs[label_idx]), 4),
    )
