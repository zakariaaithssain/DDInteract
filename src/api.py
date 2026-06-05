import sys
sys.path.insert(0, "src")

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from features import build_features

CLASS_NAMES = ["Minor", "Moderate", "Major"]

app = FastAPI(title="DDI Severity Predictor")

model = None
scaler = None
pca = None


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
    model = joblib.load("models/model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    pca = joblib.load("models/pca.joblib")


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

    return PredictResponse(
        smiles_a=req.smiles_a,
        smiles_b=req.smiles_b,
        predicted_severity=CLASS_NAMES[label_idx],
        probabilities=prob_dict,
        confidence=round(float(probs[label_idx]), 4),
    )
