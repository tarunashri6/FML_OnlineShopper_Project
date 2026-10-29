"""Class-imbalance strategies for the Online Shoppers project.

Provides SMOTENC (SMOTE for Nominal and Continuous features) applied to raw
categorical features before one-hot encoding to avoid fractional categorical
dummy corruptions, categorical-safe resampling on processed matrices, and
scikit-learn / imbalanced-learn pipeline construction for strictly in-fold,
leakage-free cross-validation.
"""

from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import sparse


RANDOM_STATE = 42

IMBALANCE_CONFIGURATIONS = ("none", "smote", "smotenc", "class_weight")

RAW_CATEGORICAL_COLUMNS = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

RAW_NUMERICAL_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]


def resample_raw_with_smotenc(
    X_raw: pd.DataFrame,
    y_raw: pd.Series | np.ndarray,
    categorical_cols: list[str] | None = None,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Resample raw tabular data using SMOTENC before one-hot encoding.

    Categorical features are preserved with valid discrete categories instead
    of fractional dummy interpolations.
    """
    if categorical_cols is None:
        categorical_cols = [c for c in RAW_CATEGORICAL_COLUMNS if c in X_raw.columns]

    categorical_indices = [X_raw.columns.get_loc(c) for c in categorical_cols]
    sampler = SMOTENC(categorical_features=categorical_indices, random_state=random_state)
    X_resampled, y_resampled = sampler.fit_resample(X_raw, y_raw)
    return X_resampled, np.asarray(y_resampled).ravel()


def snap_categorical_dummies(X_matrix, feature_names: list[str] | None = None):
    """Ensure one-hot categorical dummy columns contain strictly discrete {0.0, 1.0}."""
    is_sparse = sparse.issparse(X_matrix)
    arr = X_matrix.toarray() if is_sparse else np.array(X_matrix, copy=True)

    if feature_names is not None:
        cat_indices = [i for i, name in enumerate(feature_names) if name.startswith("categorical__")]
    else:
        # Infer binary/dummy columns (columns whose non-zero entries are nominally 0 or 1)
        cat_indices = [
            i for i in range(arr.shape[1])
            if np.all((arr[:, i] >= 0.0) & (arr[:, i] <= 1.0)) and np.unique(arr[:, i]).size <= 3
        ]

    for idx in cat_indices:
        arr[:, idx] = np.where(arr[:, idx] >= 0.5, 1.0, 0.0)

    return sparse.csr_matrix(arr) if is_sparse else arr


def apply_imbalance_strategy(
    X_train,
    y_train,
    strategy="none",
    feature_names: list[str] | None = None,
    categorical_cols: list[str] | None = None,
):
    """Prepare training data for one of the agreed imbalance configurations.

    If X_train is a raw pandas DataFrame, SMOTENC is used on the raw categories.
    If X_train is a processed matrix, SMOTE is applied followed by categorical
    dummy snapping to avoid fractional one-hot corruptions.
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

    # If raw DataFrame is provided, use SMOTENC directly
    if isinstance(X_train, pd.DataFrame):
        X_resampled, y_resampled = resample_raw_with_smotenc(
            X_train, y, categorical_cols=categorical_cols, random_state=RANDOM_STATE
        )
        return X_resampled, y_resampled, {}

    # Otherwise, apply SMOTE and snap fractional one-hot dummy values
    sampler = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = sampler.fit_resample(X_train, y)
    X_resampled = snap_categorical_dummies(X_resampled, feature_names=feature_names)

    return X_resampled, np.asarray(y_resampled).ravel(), {}


def build_resampling_pipeline(
    model,
    strategy="none",
    preprocessor=None,
    feature_indices=None,
    categorical_cols: list[str] | None = None,
    raw_columns: list[str] | None = None,
    random_state: int = RANDOM_STATE,
) -> ImbPipeline:
    """Build an imbalanced-learn Pipeline ensuring resampling runs strictly inside training folds."""
    from sklearn.preprocessing import FunctionTransformer

    steps = []

    if strategy in ("smote", "smotenc"):
        if categorical_cols is None:
            categorical_cols = RAW_CATEGORICAL_COLUMNS
        if raw_columns is not None:
            cat_indices = [raw_columns.index(c) for c in categorical_cols if c in raw_columns]
        else:
            cat_indices = list(range(len(categorical_cols)))
        steps.append(("resample", SMOTENC(categorical_features=cat_indices, random_state=random_state)))

    if preprocessor is not None:
        steps.append(("preprocessor", preprocessor))

    if feature_indices is not None:
        steps.append(("feature_selector", FunctionTransformer(lambda X: X[:, feature_indices])))

    steps.append(("model", model))
    return ImbPipeline(steps)


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
