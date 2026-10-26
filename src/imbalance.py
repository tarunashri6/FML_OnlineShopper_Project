"""Class-imbalance strategies for the Online Shoppers project.

SMOTE is applied to training data only. Because the shared processed matrix
contains one-hot categorical columns, ordinary SMOTE may create fractional
values in those columns. This limitation should be considered when interpreting
the resulting synthetic samples.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy import sparse


RANDOM_STATE = 42

IMBALANCE_CONFIGURATIONS = ("none", "smote", "class_weight")


def apply_imbalance_strategy(X_train, y_train, strategy="none"):
    """Prepare training data for one of the agreed imbalance configurations.

    Parameters
    ----------
    X_train : array-like or sparse matrix
        Official processed training features.
    y_train : array-like or pandas Series
        Training labels, where True indicates a purchase.
    strategy : str
        One of 'none', 'smote', or 'class_weight'.

    Returns
    -------
    X_resampled, y_resampled, model_options
        Training data and model options. For class_weight, the data remains
        unchanged and model_options contains class_weight='balanced'.
    """
    if strategy not in IMBALANCE_CONFIGURATIONS:
        raise ValueError(
            f"Unknown strategy {strategy!r}. "
            f"Choose from {IMBALANCE_CONFIGURATIONS}."
        )

    y = np.asarray(y_train).ravel()

    if X_train.shape[0] != len(y):
        raise ValueError("Training feature rows and target labels do not match.")

    if strategy == "none":
        return X_train, y, {}

    if strategy == "class_weight":
        return X_train, y, {"class_weight": "balanced"}

    sampler = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = sampler.fit_resample(X_train, y)

    return X_resampled, np.asarray(y_resampled).ravel(), {}


def get_model_options(strategy="none"):
    """Return model-fitting options for a strategy without resampling data."""
    if strategy not in IMBALANCE_CONFIGURATIONS:
        raise ValueError(
            f"Unknown strategy {strategy!r}. "
            f"Choose from {IMBALANCE_CONFIGURATIONS}."
        )

    if strategy == "class_weight":
        return {"class_weight": "balanced"}

    return {}


def summarize_class_distribution(y, label="dataset"):
    """Return a small summary of class counts and proportions."""
    values = np.asarray(y).ravel()
    counts = pd.Series(values).value_counts().sort_index()
    total = len(values)

    return pd.DataFrame({
        "dataset": label,
        "class": counts.index,
        "count": counts.values,
        "proportion": counts.values / total if total else [],
    })
