from typing import Any

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
from src.metrics import ordinal_mae
from src.visualization import log_confusion_matrix


def evaluate_and_log(
    model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray, run_name: str, params: dict[str, Any]
) -> dict[str, float]:
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, labels=[0, 1, 2])
    macro_f1 = np.mean(f1)
    _, _, weighted_f1, _ = precision_recall_fscore_support(y_test, preds, labels=[0, 1, 2], average="weighted")
    kappa = cohen_kappa_score(y_test, preds)
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
            "cohen_kappa": kappa,
            "mae": mae,
        }
    )

    log_confusion_matrix(cm, run_name, str(params))

    report = classification_report(y_test, preds, target_names=CLASS_NAMES, digits=4)
    mlflow.log_text(report, "classification_report.txt")

    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "kappa": kappa, "mae": mae}
