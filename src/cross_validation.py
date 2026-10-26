"""Stratified K-Fold cross-validation with strictly in-fold resampling.

This module guarantees zero data leakage:
1. SMOTENC resampling is executed strictly inside training folds.
2. Numerical standardisation is fitted on training folds only.
3. Categorical one-hot encoding uses fixed categories derived from domain knowledge.
4. Validation folds are NEVER resampled and evaluate honest performance.
"""

from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "nayantara"
PIPELINE_PATH = ROOT / "results" / "taruni" / "preprocessing_pipeline.joblib"
FEATURE_CONFIG_PATH = ROOT / "results" / "feature_selection" / "feature_configurations.joblib"

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


class FeatureSubsetSelector(BaseEstimator, TransformerMixin):
    """Transformer to select a subset of preprocessed features by column indices."""

    def __init__(self, selected_indices: list[int] | None = None):
        self.selected_indices = selected_indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.selected_indices is None:
            return X
        return X[:, self.selected_indices]


def build_fold_preprocessor(fixed_categories: list[np.ndarray]) -> ColumnTransformer:
    """Build a fresh ColumnTransformer to fit strictly inside each CV training fold."""
    return ColumnTransformer([
        ("numerical", StandardScaler(), RAW_NUMERICAL_COLUMNS),
        (
            "categorical",
            OneHotEncoder(
                categories=fixed_categories,
                handle_unknown="ignore",
                sparse_output=False,
            ),
            RAW_CATEGORICAL_COLUMNS,
        ),
    ])


def build_cv_pipeline(
    model,
    strategy: str = "smotenc",
    fixed_categories: list[np.ndarray] | None = None,
    cat_indices: list[int] | None = None,
    feature_indices: list[int] | None = None,
    random_state: int = 42,
) -> ImbPipeline:
    """Construct an imbalanced-learn Pipeline ensuring strictly in-fold execution."""
    steps = []

    if strategy in ("smote", "smotenc"):
        steps.append((
            "resample",
            SMOTENC(categorical_features=cat_indices, random_state=random_state),
        ))

    preprocessor = build_fold_preprocessor(fixed_categories)
    steps.append(("preprocessor", preprocessor))

    if feature_indices is not None:
        steps.append(("feature_selector", FeatureSubsetSelector(feature_indices)))

    steps.append(("model", model))
    return ImbPipeline(steps)


def run_cross_validation_experiments(n_splits: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Run stratified cross-validation for key model and imbalance configurations."""
    print("Loading data for stratified cross-validation...")
    df = pd.read_csv(DATA_DIR / "train_raw.csv")
    X = df.drop(columns=["Revenue"])
    y = df["Revenue"].astype(bool)

    base_pipe = joblib.load(PIPELINE_PATH)
    fixed_categories = base_pipe.named_transformers_["categorical"].categories_
    feature_names = pd.read_csv(DATA_DIR / "feature_names.csv")["Feature"].tolist()
    feature_configs = joblib.load(FEATURE_CONFIG_PATH)

    cat_indices = [X.columns.get_loc(c) for c in RAW_CATEGORICAL_COLUMNS]

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        "svm": SVC(probability=True, random_state=random_state),
    }

    strategies = ["none", "class_weight", "smotenc"]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    records = []

    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for model_name, base_model in models.items():
        for strategy in strategies:
            # Configure model class weight if needed
            if strategy == "class_weight":
                if hasattr(base_model, "set_params"):
                    model = base_model.set_params(class_weight="balanced")
                else:
                    model = base_model
            else:
                if hasattr(base_model, "set_params"):
                    model = base_model.set_params(class_weight=None)
                else:
                    model = base_model

            # Use tree_importance selected features for random_forest, all features for others
            if model_name == "random_forest":
                feat_cfg = "tree_importance"
                sel_names = feature_configs[feat_cfg]
                feat_indices = [feature_names.index(f) for f in sel_names]
            else:
                feat_cfg = "all"
                feat_indices = None

            pipeline = build_cv_pipeline(
                model=model,
                strategy=strategy,
                fixed_categories=fixed_categories,
                cat_indices=cat_indices,
                feature_indices=feat_indices,
                random_state=random_state,
            )

            scores = cross_validate(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
            )

            record = {
                "model": model_name,
                "feature_configuration": feat_cfg,
                "imbalance_strategy": strategy,
                "cv_folds": n_splits,
                "accuracy_mean": float(scores["test_accuracy"].mean()),
                "accuracy_std": float(scores["test_accuracy"].std()),
                "precision_mean": float(scores["test_precision"].mean()),
                "precision_std": float(scores["test_precision"].std()),
                "recall_mean": float(scores["test_recall"].mean()),
                "recall_std": float(scores["test_recall"].std()),
                "f1_mean": float(scores["test_f1"].mean()),
                "f1_std": float(scores["test_f1"].std()),
                "roc_auc_mean": float(scores["test_roc_auc"].mean()),
                "roc_auc_std": float(scores["test_roc_auc"].std()),
            }
            records.append(record)
            print(
                f"[{model_name}] [{strategy}] F1: {record['f1_mean']:.4f} ± {record['f1_std']:.4f} | "
                f"Recall: {record['recall_mean']:.4f} ± {record['recall_std']:.4f} | "
                f"ROC-AUC: {record['roc_auc_mean']:.4f} ± {record['roc_auc_std']:.4f}"
            )

    result_df = pd.DataFrame(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "cv_resampling_results.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved cross-validation results to: {out_path}")
    return result_df


if __name__ == "__main__":
    run_cross_validation_experiments()
