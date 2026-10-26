"""Evaluation utilities for the Online Shopper classification experiments.

This module keeps the metric logic separate from the model-building code so the
same evaluation can be reused across feature configurations and imbalance
strategies.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


POSITIVE_LABEL = True


def get_model_scores(model, X):
    """Return a 1-D score array for ROC-AUC evaluation.

    For scikit-learn classifiers, the score should be the probability of the
    positive class (True / purchase). If a model does not expose probabilities,
    we fall back to decision_function() or the raw predictions.
    """
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return np.asarray(probabilities[:, 1], dtype=float)
        return np.asarray(probabilities.ravel(), dtype=float)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if hasattr(scores, "shape") and len(scores.shape) > 1 and scores.shape[1] > 1:
            return np.asarray(scores[:, 1], dtype=float)
        return np.asarray(scores, dtype=float)

    return np.asarray(model.predict(X), dtype=float)


def evaluate_predictions(y_true, y_pred, y_score):
    """Compute the key metrics for the held-out test set."""
    y_true_arr = np.asarray(y_true, dtype=bool).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=bool).ravel()
    y_score_arr = np.asarray(y_score, dtype=float).ravel()

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(
            precision_score(
                y_true_arr,
                y_pred_arr,
                pos_label=POSITIVE_LABEL,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true_arr,
                y_pred_arr,
                pos_label=POSITIVE_LABEL,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true_arr,
                y_pred_arr,
                pos_label=POSITIVE_LABEL,
                zero_division=0,
            )
        ),
        "roc_auc": float(roc_auc_score(y_true_arr.astype(int), y_score_arr)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return metrics
