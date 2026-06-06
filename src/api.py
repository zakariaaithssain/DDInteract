import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

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

from src.config import BEST_MODEL_PATH, CLASS_NAMES, DRIFT_REFERENCE_PATH, MODEL_PATH, PCA_PATH, SCALER_PATH
from src.drift import detect_drift, save_report
from src.features import build_features
from src.logger import logger

DRIFT_CHECK_INTERVAL: int = 100

model: BaseEstimator | None = None
scaler: StandardScaler | None = None
pca: PCA | None = None

reference_stats: dict[str, Any] | None = None
feature_buffer: list[np.ndarray] = []
last_drift_result: dict[str, Any] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model artifacts and drift reference stats on startup.

    Args:
        app: The FastAPI application instance.
    """
    global model, scaler, pca, reference_stats, feature_buffer, last_drift_result

    logger.info("Loading model artifacts")
    try:
        model = joblib.load(BEST_MODEL_PATH)
    except FileNotFoundError:
        model = joblib.load(MODEL_PATH)  # fallback to export-model output
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    logger.info("Artifacts loaded successfully")

    logger.info("Loading drift reference stats")
    try:
        with open(DRIFT_REFERENCE_PATH) as f:
            reference_stats = json.load(f)
        logger.info("Drift reference loaded (%d samples)", reference_stats["n_samples"])
    except FileNotFoundError, json.JSONDecodeError:
        logger.warning("Drift reference not found or invalid — drift detection disabled")

    feature_buffer = []
    last_drift_result = None
    yield


app = FastAPI(title="DDI Severity Predictor", lifespan=lifespan)


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


class DriftStatus(BaseModel):
    """Drift detection status."""

    monitoring_active: bool
    samples_collected: int
    check_interval: int
    last_result: dict[str, Any] | None


@app.get("/")
def index() -> FileResponse:
    """Serve the frontend HTML page."""
    return FileResponse("src/static/index.html")


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


def _run_drift_check() -> dict[str, Any]:
    """Run drift detection on buffered features and reset the buffer.

    Returns:
        Drift detection result dict.
    """
    global feature_buffer, last_drift_result, reference_stats

    if not feature_buffer or reference_stats is None:
        return {"drift_detected": False, "reason": "insufficient_data"}

    X_new = np.array(feature_buffer)
    result = detect_drift(X_new, reference_stats)
    if result.get("drift_detected"):
        save_report(result)
    feature_buffer = []
    last_drift_result = result
    return result


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Predict DDI severity for a pair of SMILES strings.

    Args:
        req: PredictRequest with smiles_a and smiles_b.

    Returns:
        PredictResponse with predicted severity and probabilities.
    """
    global feature_buffer, reference_stats

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

    if reference_stats is not None:
        feature_buffer.append(X[0])
        if len(feature_buffer) >= DRIFT_CHECK_INTERVAL:
            _run_drift_check()

    return PredictResponse(
        smiles_a=req.smiles_a,
        smiles_b=req.smiles_b,
        predicted_severity=CLASS_NAMES[label_idx],
        probabilities=prob_dict,
        confidence=round(float(probs[label_idx]), 4),
    )


@app.get("/drift", response_model=DriftStatus)
def drift_status() -> DriftStatus:
    """Return current drift detection status."""
    return DriftStatus(
        monitoring_active=reference_stats is not None,
        samples_collected=len(feature_buffer),
        check_interval=DRIFT_CHECK_INTERVAL,
        last_result=last_drift_result,
    )
