import sys
sys.path.insert(0, "src")

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from features import build_features
from logger import logger

CLASS_NAMES = ["Minor", "Moderate", "Major"]

app = FastAPI(title="DDI Severity Predictor")
model = scaler = pca = None


class PredictRequest(BaseModel):
    smiles_a: str
    smiles_b: str


class PredictResponse(BaseModel):
    smiles_a: str
    smiles_b: str
    predicted_severity: str
    probabilities: dict[str, float]
    confidence: float


@app.on_event("startup")
def load_artifacts():
    global model, scaler, pca
    logger.info("Loading model artifacts")
    model = joblib.load("models/model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    pca = joblib.load("models/pca.joblib")
    logger.info("Artifacts loaded successfully")


@app.get("/")
def index():
    return FileResponse("src/static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = pd.DataFrame({"smiles_a": [req.smiles_a], "smiles_b": [req.smiles_b]})
    X = build_features(df)
    X_s = scaler.transform(X)
    X_pca = pca.transform(X_s)

    probs = model.predict_proba(X_pca)[0]
    label_idx = int(np.argmax(probs))
    prob_dict = {cls: round(float(p), 4) for cls, p in zip(CLASS_NAMES, probs)}

    logger.info("Prediction: %s + %s → %s (conf=%.2f)", req.smiles_a, req.smiles_b, CLASS_NAMES[label_idx], probs[label_idx])

    return PredictResponse(
        smiles_a=req.smiles_a,
        smiles_b=req.smiles_b,
        predicted_severity=CLASS_NAMES[label_idx],
        probabilities=prob_dict,
        confidence=round(float(probs[label_idx]), 4),
    )
