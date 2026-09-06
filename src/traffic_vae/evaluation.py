"""Calibration and held-out binary detection metrics."""
import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve,
    precision_recall_fscore_support, roc_auc_score, auc,
)


def calibrate(scores: np.ndarray, target_fpr: float) -> float:
    if not 0 < target_fpr < 1 or len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("Calibration requires finite scores and 0 < target_fpr < 1.")
    return float(np.quantile(scores, 1 - target_fpr, method="higher"))


def evaluate(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Evaluation requires both normal and attack examples.")
    if len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite and match label count.")
    predictions = scores > threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    curve_precision, curve_recall, _ = precision_recall_curve(labels, scores)
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "false_positive_rate": float(fp / (fp + tn)),
        "average_precision": float(average_precision_score(labels, scores)),
        "pr_auc_trapezoidal": float(auc(curve_recall, curve_precision)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "attack_prevalence": float(labels.mean()),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
        "threshold": float(threshold), "test_rows": len(labels),
    }
