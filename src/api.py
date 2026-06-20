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

from src.config import BEST_MODEL_PATH, CLASS_NAMES, DRIFT_BUFFER_PATH, DRIFT_REFERENCE_PATH, logger
from src.drift import detect_drift, save_report
from src.features import build_features

DRIFT_CHECK_INTERVAL: int = 100

model: BaseEstimator | None = None

reference_stats: dict[str, Any] | None = None
feature_buffer: list[np.ndarray] = []
class_buffer: list[str] = []
confidence_buffer: list[float] = []
last_drift_result: dict[str, Any] | None = None


def _load_buffer() -> list[np.ndarray]:
    """Load persisted drift buffer from disk."""
    try:
        data = np.load(DRIFT_BUFFER_PATH)
        if data.ndim == 1:
            return [data]
        return [data[i] for i in range(len(data))]
    except FileNotFoundError:
        return []


def _save_buffer(buffer: list[np.ndarray]) -> None:
    """Persist drift buffer to disk."""
    if buffer:
        np.save(DRIFT_BUFFER_PATH, np.array(buffer))
    else:
        import os

        try:
            os.remove(DRIFT_BUFFER_PATH)
        except FileNotFoundError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model artifact, drift reference stats, and persisted buffer on startup."""
    global model, reference_stats, feature_buffer, last_drift_result

    logger.info("Loading model artifact")
    try:
        model = joblib.load(BEST_MODEL_PATH)
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.error("Model file not found at %s — predictions will fail", BEST_MODEL_PATH)
        model = None

    logger.info("Loading drift reference stats")
    try:
        with open(DRIFT_REFERENCE_PATH) as f:
            reference_stats = json.load(f)
        logger.info("Drift reference loaded (%d samples)", reference_stats["n_samples"])
    except FileNotFoundError, json.JSONDecodeError:
        logger.warning("Drift reference not found or invalid — drift detection disabled")

    feature_buffer = _load_buffer()
    if feature_buffer:
        logger.info("Restored drift buffer from disk (%d samples)", len(feature_buffer))
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
    """Drift detection status with per-type breakdown."""

    monitoring_active: bool
    samples_collected: int
    check_interval: int
    last_result: dict[str, Any] | None
    label_counts: dict[str, int] | None = None
    mean_confidence: float | None = None


@app.get("/")
def index() -> FileResponse:
    """Serve the frontend HTML page."""
    return FileResponse("src/static/index.html")


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


def _run_drift_check() -> dict[str, Any]:
    """Run all three drift checks on buffered data and reset buffers."""
    global feature_buffer, class_buffer, confidence_buffer, last_drift_result, reference_stats

    if not feature_buffer or reference_stats is None:
        return {"drift_detected": False, "reason": "insufficient_data"}

    X_new = np.array(feature_buffer)
    result = detect_drift(X_new, class_buffer, confidence_buffer, reference_stats)
    if result.get("drift_detected"):
        save_report(result)
    feature_buffer = []
    class_buffer = []
    confidence_buffer = []
    _save_buffer(feature_buffer)
    last_drift_result = result
    return result


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Predict DDI severity for a pair of SMILES strings."""
    global feature_buffer, class_buffer, confidence_buffer, reference_stats

    if model is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Model not loaded — train the pipeline first")

    df = pd.DataFrame({"smiles_a": [req.smiles_a], "smiles_b": [req.smiles_b]})
    X: NDArray[np.float64] = build_features(df)

    probs: NDArray[np.float64] = model.predict_proba(X)[0]
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
        class_buffer.append(CLASS_NAMES[label_idx])
        confidence_buffer.append(float(probs[label_idx]))
        _save_buffer(feature_buffer)
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
    """Return current drift detection status with per-type breakdown."""
    label_counts = None
    mean_confidence = None
    if class_buffer:
        label_counts = {cls: class_buffer.count(cls) for cls in CLASS_NAMES}
    if confidence_buffer:
        mean_confidence = round(float(np.mean(confidence_buffer)), 4)
    return DriftStatus(
        monitoring_active=reference_stats is not None,
        samples_collected=len(feature_buffer),
        check_interval=DRIFT_CHECK_INTERVAL,
        last_result=last_drift_result,
        label_counts=label_counts,
        mean_confidence=mean_confidence,
    )
