"""
evaluation/metrics.py
─────────────────────────────────────────────────────────────────────────────
Classification metrics for binary imbalanced problems.

All functions accept plain numpy arrays and return either a scalar or a dict.
No model-specific logic lives here — this module is intentionally stateless.

Recommended primary metric for threshold-independent evaluation: PR-AUC.
Accuracy is deliberately excluded: it is misleading under class imbalance.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute a full suite of binary classification metrics.

    Parameters
    ----------
    y_true    : Ground-truth binary labels (0 / 1).
    y_prob    : Predicted probabilities for the positive class.
    threshold : Decision threshold applied to y_prob.

    Returns
    -------
    dict with keys:
        pr_auc      — area under the Precision-Recall curve (primary metric)
        roc_auc     — area under the ROC curve
        f1          — F1 score at *threshold*
        precision   — precision at *threshold*
        recall      — recall (sensitivity) at *threshold*
        specificity — true negative rate at *threshold*
        mcc         — Matthews Correlation Coefficient
        g_mean      — geometric mean of sensitivity and specificity
        threshold   — the threshold used
    """
    y_pred = (y_prob >= threshold).astype(int)

    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    g_mean      = float(np.sqrt(sensitivity * specificity))

    return {
        "pr_auc":      float(average_precision_score(y_true, y_prob)),
        "roc_auc":     float(roc_auc_score(y_true, y_prob)),
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":      float(sensitivity),
        "specificity": float(specificity),
        "mcc":         float(matthews_corrcoef(y_true, y_pred)),
        "g_mean":      g_mean,
        "threshold":   float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def evaluate_at_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Evaluate metrics at multiple thresholds.

    Useful for threshold sweeps and for building calibration curves.

    Parameters
    ----------
    y_true     : Ground-truth binary labels.
    y_prob     : Predicted probabilities for the positive class.
    thresholds : Array of thresholds to evaluate.  Defaults to 100 points
                 in [0, 1].

    Returns
    -------
    List of metric dicts, one per threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 100)
    return [evaluate(y_true, y_prob, t) for t in thresholds]