"""Clinical cost-weighted scoring for DDI severity classification.

Severity classes: Minor=0, Moderate=1, Major=2.

The cost matrix encodes asymmetric risk: under-triage (predicting a lower
severity than the truth) is penalized far more than over-triage, since
under-triage hides a real interaction from clinical review while
over-triage only causes an extra (safe) alert.

NOTE: the values below are illustrative defaults, not clinically validated.
Replace them with numbers reviewed by a pharmacist/clinical stakeholder
before relying on this in production, and keep the reasoning documented
so the choice is auditable.
"""

import numpy as np
from sklearn.metrics import confusion_matrix

# rows = true label, cols = predicted label. order: Minor=0, Moderate=1, Major=2
COST_MATRIX = np.array(
    [
        [0, 1, 2],  # true Minor:    over-triage to Moderate/Major (mild cost)
        [2, 0, 1],  # true Moderate: under-triage to Minor costs more than over-triage to Major
        [5, 3, 0],  # true Major:    under-triage — the costliest cells in the matrix
    ],
    dtype=float,
)

_CLASS_NAMES = ["Minor", "Moderate", "Major"]


def expected_cost(y_true: np.ndarray, y_pred: np.ndarray, cost_matrix: np.ndarray = COST_MATRIX) -> float:
    """Mean misclassification cost per sample. Lower is better — pass direction='minimize' to Optuna."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return float((cm * cost_matrix).sum() / cm.sum())


def cost_breakdown(y_true: np.ndarray, y_pred: np.ndarray, cost_matrix: np.ndarray = COST_MATRIX) -> dict[str, float]:
    """Per-true-class contribution to total cost — useful for MLflow logging/debugging."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    weighted = cm * cost_matrix
    total = cm.sum()
    return {f"cost_from_true_{_CLASS_NAMES[i]}": float(weighted[i].sum()) / total for i in range(3)}
