"""Reusable model constructors for the Online Shopper experiment pipeline.

This file is intentionally small and explicit so the experiments remain easy
for beginners to follow. The models are created with the same training-only and
feature-selection rules used elsewhere in the project.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42


def build_model(model_name: str, **model_options: Any):
    """Return a fitted or unfitted estimator for one of the agreed models.

    Parameters
    ----------
    model_name : str
        One of: logistic_regression, decision_tree, random_forest, svm
    model_options : dict
        Extra keywords such as class_weight or probability settings.

    Returns
    -------
    sklearn estimator
        An unfitted model instance ready for .fit(...).
    """
    name = model_name.strip().lower().replace(" ", "_")

    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="liblinear",
            **model_options,
        )

    if name == "decision_tree":
        return DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            **model_options,
        )

    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **model_options,
        )

    if name == "svm":
        # probability=True ensures ROC-AUC can be computed from class scores even
        # with a non-probabilistic SVM. This is compatible with the project goal
        # of evaluating ROC-AUC on the held-out test set.
        return SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE,
            **model_options,
        )

    raise ValueError(
        f"Unsupported model name: {model_name!r}. "
        "Choose from logistic_regression, decision_tree, random_forest, svm."
    )
