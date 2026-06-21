import os
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import CLASS_NAMES

matplotlib.use("Agg")


# --- Metric functions ---


def ordinal_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


# --- Visualization ---


def log_confusion_matrix(cm: np.ndarray, run_name: str, params_str: str, output_dir: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)  # type: ignore[attr-defined]
    ax.set_title(f"Confusion Matrix — {run_name}\n{params_str}", fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black"
            )
    cm_path = os.path.join(output_dir, "confusion_matrix.png") if output_dir else "/tmp/confusion_matrix.png"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(cm_path, bbox_inches="tight")
    mlflow.log_artifact(cm_path, artifact_path="confusion_matrices")
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)


# --- Evaluation ---


def evaluate_and_log(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    run_name: str,
    params: dict[str, Any],
    output_dir: str | None = None,
) -> dict[str, float]:
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, labels=[0, 1, 2])
    macro_f1 = np.mean(f1)
    _, _, weighted_f1, _ = precision_recall_fscore_support(y_test, preds, labels=[0, 1, 2], average="weighted")
    qwk = quadratic_weighted_kappa(y_test, preds)
    mae = ordinal_mae(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])

    for i, cls in enumerate(CLASS_NAMES):
        mlflow.log_metrics(
            {
                f"{cls}_precision": prec[i],
                f"{cls}_recall": rec[i],
                f"{cls}_f1": f1[i],
            }
        )
    mlflow.log_metrics(
        {
            "test_accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "qwk": qwk,
            "mae": mae,
        }
    )

    log_confusion_matrix(cm, run_name, str(params), output_dir)

    report = classification_report(y_test, preds, target_names=CLASS_NAMES, digits=4)
    mlflow.log_text(report, "classification_report.txt")
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
            f.write(report)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "qwk": qwk,
        "mae": mae,
        "Minor_precision": prec[0],
        "Minor_recall": rec[0],
        "Minor_f1": f1[0],
        "Moderate_precision": prec[1],
        "Moderate_recall": rec[1],
        "Moderate_f1": f1[1],
        "Major_precision": prec[2],
        "Major_recall": rec[2],
        "Major_f1": f1[2],
    }
