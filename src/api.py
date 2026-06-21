import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from numpy.typing import NDArray
from pydantic import BaseModel
from sklearn.base import BaseEstimator

from src.config import BEST_MODEL_PATH, CLASS_NAMES, DRIFT_BUFFER_PATH, DRIFT_REFERENCE_PATH, logger
from src.drift import detect_drift, save_report
from src.features import build_features
from src.threshold_optimizer import apply_thresholds, load_thresholds

DRIFT_CHECK_INTERVAL: int = 100

DRIFT_HTML: str = Path("src/static/drift.html").read_text()

model: BaseEstimator | None = None
thresholds: dict[str, float] | None = None

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
    global model, reference_stats, feature_buffer, last_drift_result, thresholds

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

    logger.info("Loading prediction thresholds")
    thresholds = load_thresholds()
    if thresholds is not None:
        logger.info("Thresholds loaded: t_major=%.3f, t_minor=%.3f", thresholds["t_major"], thresholds["t_minor"])
    else:
        logger.info("No thresholds found — falling back to argmax")
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


class DriftCheckStatus(BaseModel):
    """Status of an individual drift check (covariate, label, or concept)."""

    status: str = "pending"  # "pending", "checked", "insufficient_data"
    drift_detected: bool | None = None
    details: dict[str, Any] | None = None


class DriftStatus(BaseModel):
    """Drift detection status with per-type breakdown."""

    monitoring_active: bool
    samples_collected: int
    check_interval: int
    covariate_shift: DriftCheckStatus
    label_shift: DriftCheckStatus
    concept_drift: DriftCheckStatus
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
    pred_label, prob_dict, confidence = apply_thresholds(probs, thresholds)

    logger.info(
        "Prediction: %s + %s → %s (conf=%.2f)",
        req.smiles_a,
        req.smiles_b,
        pred_label,
        confidence,
    )

    if reference_stats is not None:
        feature_buffer.append(X[0])
        class_buffer.append(pred_label)
        confidence_buffer.append(confidence)
        _save_buffer(feature_buffer)
        if len(feature_buffer) >= DRIFT_CHECK_INTERVAL:
            _run_drift_check()

    return PredictResponse(
        smiles_a=req.smiles_a,
        smiles_b=req.smiles_b,
        predicted_severity=pred_label,
        probabilities=prob_dict,
        confidence=confidence,
    )


@app.get("/drift")
def drift_gui() -> HTMLResponse:
    """Serve the drift monitoring GUI."""
    return HTMLResponse(DRIFT_HTML)


@app.get("/drift/data")
def drift_status() -> DriftStatus:
    label_counts = None
    mean_confidence = None
    if class_buffer:
        label_counts = {cls: class_buffer.count(cls) for cls in CLASS_NAMES}
    if confidence_buffer:
        mean_confidence = round(float(np.mean(confidence_buffer)), 4)

    covariate = DriftCheckStatus(status="pending")
    label = DriftCheckStatus(status="pending")
    concept = DriftCheckStatus(status="pending")

    if last_drift_result is not None:
        if "covariate_shift" in last_drift_result:
            cov = last_drift_result["covariate_shift"]
            covariate = DriftCheckStatus(
                status="checked",
                drift_detected=cov.get("drift_detected"),
                details={k: v for k, v in cov.items() if k != "drift_detected"},
            )
        if "label_shift" in last_drift_result:
            lab = last_drift_result["label_shift"]
            label = DriftCheckStatus(
                status="checked",
                drift_detected=lab.get("drift_detected"),
                details={k: v for k, v in lab.items() if k != "drift_detected"},
            )
        if "concept_drift" in last_drift_result:
            con = last_drift_result["concept_drift"]
            concept = DriftCheckStatus(
                status="checked",
                drift_detected=con.get("drift_detected"),
                details={k: v for k, v in con.items() if k != "drift_detected"},
            )

    return DriftStatus(
        monitoring_active=reference_stats is not None,
        samples_collected=len(feature_buffer),
        check_interval=DRIFT_CHECK_INTERVAL,
        covariate_shift=covariate,
        label_shift=label,
        concept_drift=concept,
        label_counts=label_counts,
        mean_confidence=mean_confidence,
    )
