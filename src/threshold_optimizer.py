import json

import numpy as np
import optuna
from sklearn.metrics import precision_score, recall_score

from src.config import CLASS_NAMES, THRESHOLDS_PATH, logger


def evaluate_clinical_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    prec_minor = precision_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)
    rec_major = recall_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)
    rec_mod = recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)
    return 0.40 * prec_minor + 0.40 * rec_major + 0.20 * rec_mod


def predict_with_thresholds(y_probs: np.ndarray, t_major: float, t_minor: float) -> np.ndarray:
    preds = np.full(len(y_probs), 1, dtype=int)
    is_major = y_probs[:, 2] >= t_major
    preds[is_major] = 2
    is_minor = (y_probs[:, 0] >= t_minor) & (~is_major)
    preds[is_minor] = 0
    return preds


def optimize_thresholds(y_true: np.ndarray, y_probs: np.ndarray, n_trials: int = 200) -> dict[str, float]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        t_major = trial.suggest_float("t_major", 0.15, 0.60)
        t_minor = trial.suggest_float("t_minor", 0.40, 0.90)
        y_pred = predict_with_thresholds(y_probs, t_major, t_minor)
        return evaluate_clinical_score(y_true, y_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    logger.info(
        "Best clinical score=%.4f at thresholds: t_major=%.3f, t_minor=%.3f",
        study.best_value,
        study.best_params["t_major"],
        study.best_params["t_minor"],
    )
    return study.best_params


def save_thresholds(thresholds: dict[str, float]) -> None:
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(thresholds, f, indent=2)
    logger.info("Thresholds saved to %s", THRESHOLDS_PATH)


def load_thresholds() -> dict[str, float] | None:
    try:
        with open(THRESHOLDS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def apply_thresholds(probs: np.ndarray, thresholds: dict[str, float] | None) -> tuple[str, dict[str, float], float]:
    if thresholds is None:
        label_idx = int(np.argmax(probs))
        prob_dict = {cls: round(float(p), 4) for cls, p in zip(CLASS_NAMES, probs)}
        return CLASS_NAMES[label_idx], prob_dict, round(float(probs[label_idx]), 4)

    t_major = thresholds.get("t_major", 0.5)
    t_minor = thresholds.get("t_minor", 0.8)
    pred_idx = int(predict_with_thresholds(probs.reshape(1, -1), t_major, t_minor)[0])
    pred_label = CLASS_NAMES[pred_idx]
    prob_dict = {cls: round(float(p), 4) for cls, p in zip(CLASS_NAMES, probs)}
    confidence = round(float(probs[pred_idx]), 4)
    return pred_label, prob_dict, confidence
