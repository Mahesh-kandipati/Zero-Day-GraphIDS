"""
Zero-Day Evaluation Framework and Baseline Comparison Engine.
Implements Leave-One-Attack-Family-Out protocol, ROC/PR curves, confusion matrices,
and comparative benchmarks against Isolation Forest and raw-feature baselines.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)
from sklearn.ensemble import IsolationForest

def evaluate_detector_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anomaly_scores: np.ndarray
) -> Dict[str, Any]:
    """
    Computes complete forensic classification metrics with safeguards
    for single-class test fixtures and zero-division edge cases.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    anomaly_scores = np.asarray(anomaly_scores, dtype=float)

    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else 0.0
    rec = float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else 0.0

    # ROC-AUC & PR-AUC
    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, anomaly_scores))
        except Exception:
            roc_auc = 0.5
        try:
            precisions, recalls, _ = precision_recall_curve(y_true, anomaly_scores)
            pr_auc = float(auc(recalls, precisions))
        except Exception:
            pr_auc = 0.5
    else:
        roc_auc = 1.0 if np.all(y_pred == y_true) else 0.5
        pr_auc = 1.0 if np.all(y_pred == y_true) else 0.5

    # Confusion Matrix: TN, FP, FN, TP
    if len(unique_classes) > 1:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    else:
        # Fallback when test contains only one class
        if unique_classes[0] == 0:
            tn = int(np.sum(y_pred == 0))
            fp = int(np.sum(y_pred == 1))
            fn = 0
            tp = 0
        else:
            tn = 0
            fp = 0
            fn = int(np.sum(y_pred == 0))
            tp = int(np.sum(y_pred == 1))

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
        'pr_auc': round(pr_auc, 4),
        'false_positive_rate': round(fpr, 4),
        'true_positive_rate': round(tpr, 4),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }

def run_baseline_comparison(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    proposed_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compares the proposed Self-Supervised GraphIDS against standard IDS baselines:
    1. Isolation Forest on raw tabular features.
    2. Raw Feature Centroid Distance baseline.
    """
    y_test_binary = np.array([0 if str(l).strip().upper() in ('BENIGN', 'NORMAL', '0') else 1 for l in y_test])

    # 1. Isolation Forest Baseline
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X_train_raw)
    iso_scores = -iso_forest.score_samples(X_test_raw)  # Higher = more anomalous
    iso_preds = np.where(iso_forest.predict(X_test_raw) == -1, 1, 0)
    iso_metrics = evaluate_detector_predictions(y_test_binary, iso_preds, iso_scores)

    # 2. Raw Centroid Baseline
    centroid = np.mean(X_train_raw, axis=0)
    raw_dists = np.linalg.norm(X_test_raw - centroid, axis=1)
    raw_thresh = np.percentile(raw_dists, 95)
    raw_preds = (raw_dists > raw_thresh).astype(int)
    raw_metrics = evaluate_detector_predictions(y_test_binary, raw_preds, raw_dists)

    return {
        'proposed_graphids': proposed_metrics,
        'isolation_forest': iso_metrics,
        'raw_centroid_distance': raw_metrics
    }
