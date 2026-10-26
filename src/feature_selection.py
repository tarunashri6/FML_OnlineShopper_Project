
"""Training-only feature selection for the Online Shoppers project."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "results" / "feature_selection"

RANDOM_STATE = 42
CORRELATION_THRESHOLD = 0.95
RFE_FEATURE_COUNT = 40
TREE_FEATURE_COUNT = 40


def load_training_data():
    """Load the official processed training matrix, labels, and feature names."""
    X = load_npz(DATA_DIR / "X_train_processed.npz")
    y = pd.read_csv(DATA_DIR / "y_train.csv")["Revenue"].astype(bool)
    feature_names = pd.read_csv(DATA_DIR / "feature_names.csv")["Feature"].tolist()

    if X.shape[0] != len(y):
        raise ValueError("Training matrix rows do not match target labels.")
    if X.shape[1] != len(feature_names):
        raise ValueError("Training matrix columns do not match feature names.")

    return X, y, feature_names


def select_all(feature_names):
    """Return all features as the baseline configuration."""
    return list(feature_names)


def select_by_correlation(X, y, feature_names):
    """Remove highly correlated numeric columns; retain one-hot columns.

    Correlations are calculated on the official training matrix only.
    One-hot categorical columns are retained to avoid indiscriminate
    removal of category indicators.
    """
    dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    frame = pd.DataFrame(dense, columns=feature_names)

    numeric_indices = [
        i for i, name in enumerate(feature_names)
        if name.startswith("numerical__")
    ]
    numeric_names = [feature_names[i] for i in numeric_indices]

    if len(numeric_names) < 2:
        return list(feature_names), pd.DataFrame(
            columns=["feature_removed", "correlated_with", "correlation"]
        )

    corr = frame[numeric_names].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    removed = []

    for column in upper.columns:
        high_corr = upper[column][upper[column] > CORRELATION_THRESHOLD]
        if not high_corr.empty:
            partner = high_corr.idxmax()
            removed.append({
                "feature_removed": column,
                "correlated_with": partner,
                "correlation": float(high_corr.max()),
            })

    removed_names = {item["feature_removed"] for item in removed}
    selected = [name for name in feature_names if name not in removed_names]
    return selected, pd.DataFrame(removed)


def select_with_rfe(X, y, feature_names):
    """Select features with RFE and logistic regression."""
    count = min(RFE_FEATURE_COUNT, X.shape[1])
    estimator = LogisticRegression(
        max_iter=2000,
        class_weight=None,
        random_state=RANDOM_STATE,
    )
    selector = RFE(estimator=estimator, n_features_to_select=count, step=0.1)
    selector.fit(X, y)

    selected = [
        name for name, keep in zip(feature_names, selector.support_) if keep
    ]
    ranking = pd.DataFrame({
        "feature": feature_names,
        "selected": selector.support_,
        "ranking": selector.ranking_,
    }).sort_values(["selected", "ranking"], ascending=[False, True])

    return selected, ranking


def select_with_tree_importance(X, y, feature_names):
    """Rank features using an Extra Trees classifier trained on training data."""
    estimator = ExtraTreesClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight=None,
        n_jobs=-1,
    )
    estimator.fit(X, y)

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": estimator.feature_importances_,
    }).sort_values("importance", ascending=False)

    count = min(TREE_FEATURE_COUNT, len(feature_names))
    selected = importance.head(count)["feature"].tolist()
    return selected, importance


def run_feature_selection():
    """Generate the four named feature configurations and their metadata."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, feature_names = load_training_data()

    configurations = {
        "all": select_all(feature_names),
    }

    correlation_features, correlation_details = select_by_correlation(
        X, y, feature_names
    )
    configurations["correlation"] = correlation_features
    correlation_details = correlation_details.reindex(
    columns=["feature_removed", "correlated_with", "correlation"]
    )
    correlation_details.to_csv(
    OUTPUT_DIR / "correlation_removed_features.csv",
    index=False,
    )

    rfe_features, rfe_details = select_with_rfe(X, y, feature_names)
    configurations["rfe"] = rfe_features
    rfe_details.to_csv(OUTPUT_DIR / "rfe_feature_ranking.csv", index=False)

    tree_features, tree_details = select_with_tree_importance(
        X, y, feature_names
    )
    configurations["tree_importance"] = tree_features
    tree_details.to_csv(
        OUTPUT_DIR / "tree_feature_importance.csv", index=False
    )

    for name, selected in configurations.items():
        pd.DataFrame({"Feature": selected}).to_csv(
            OUTPUT_DIR / f"{name}_features.csv", index=False
        )

    summary = pd.DataFrame([
        {"configuration": name, "feature_count": len(selected)}
        for name, selected in configurations.items()
    ])
    summary.to_csv(OUTPUT_DIR / "feature_selection_summary.csv", index=False)

    joblib.dump(
        configurations,
        OUTPUT_DIR / "feature_configurations.joblib",
    )

    print("Feature selection completed.")
    print(summary.to_string(index=False))
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_feature_selection()
