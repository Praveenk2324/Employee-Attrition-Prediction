import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

import logging

logger = logging.getLogger(__name__)

def find_best_threshold(y_true: np.ndarray, probs:np.ndarray) -> float:
    """
    Scan thresholds 0.3 - 0.8 and return the one with highest F1.
    In HR attrition, recall on positives is usually more important than precision.
    """
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.30, 0.81, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(float(best_t), 2)


def evaluate(pipeline, X_test: pd.DataFrame, y_test:pd.Series, threshold:float,metrics_path: str) -> dict:
    """
    Runs full evaluation on the test set.
    Returns metrics dict and writes metrics.json for DVC to track.
    """

    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    auc = roc_auc_score(y_test, probs)
    f1 = f1_score(y_test, preds, zero_division=0)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    best_t = find_best_threshold(y_test.values, probs)

    metrics = {
        "roc_auc": round(auc, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "threshold_used": threshold,
        "best_threshold": best_t
    }

    logger.info("\n" + classification_report(y_test, preds, target_names=["Stay", "Leave"]))
    logger.info(f"ROC-AUC:   {auc:.4f}")
    logger.info(f"F1:        {f1:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"Best threshold (by F1): {best_t}")
    logger.info(f"Confusion matrix:\n{confusion_matrix(y_test, preds)}")

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    return metrics