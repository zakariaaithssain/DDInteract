import os
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from src.config import CLASS_NAMES

matplotlib.use("Agg")


def log_confusion_matrix(cm: np.ndarray, run_name: str, params_str: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
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
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fig.savefig(f.name, bbox_inches="tight")
        mlflow.log_artifact(f.name, artifact_path="confusion_matrices")
    plt.close(fig)
    os.unlink(f.name)
