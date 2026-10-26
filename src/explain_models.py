"""SHAP explainability for the selected model configurations.

This script uses the actual best-F1 and best-recall configurations from the
saved CSV, retrains those models on the official train split only, and explains
why the model predicts purchase (Revenue=True). It does not modify the original
experiment results or create outputs outside the Nayantara folders.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.sparse import load_npz

try:
    from src.imbalance import apply_imbalance_strategy
    from src.model_builders import build_model
except ModuleNotFoundError:  # pragma: no cover - direct execution fallback
    from imbalance import apply_imbalance_strategy
    from model_builders import build_model

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_PATH = ROOT / "results" / "nayantara" / "full_experiment_results.csv"
FEATURE_CONFIG_PATH = ROOT / "results" / "feature_selection" / "feature_configurations.joblib"
RESULTS_DIR = ROOT / "results" / "nayantara"
FIGURES_DIR = ROOT / "figures" / "nayantara"


def load_best_configs():
    """Return the actual best-F1 and best-recall rows from the saved CSV."""
    df = pd.read_csv(RESULTS_PATH)
    if len(df) != 48:
        raise ValueError(f"Expected 48 experiment rows, found {len(df)}.")
    best_f1_row = df.sort_values("f1", ascending=False).iloc[0]
    best_recall_row = df.sort_values("recall", ascending=False).iloc[0]
    return df, best_f1_row, best_recall_row


def load_official_data():
    """Load the project’s official train/test split and feature names."""
    X_train = load_npz(DATA_DIR / "X_train_processed.npz")
    X_test = load_npz(DATA_DIR / "X_test_processed.npz")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["Revenue"].astype(bool)
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["Revenue"].astype(bool)
    feature_names = pd.read_csv(DATA_DIR / "feature_names.csv")["Feature"].tolist()
    return X_train, X_test, y_train, y_test, feature_names


def select_feature_subset(X, feature_names, selected_features):
    """Select a subset of feature columns while preserving stable ordering."""
    missing = [name for name in selected_features if name not in feature_names]
    if missing:
        raise ValueError(f"Selected features are missing from feature_names: {missing}")
    indices = [feature_names.index(name) for name in selected_features]
    if hasattr(X, "toarray"):
        return X[:, indices].toarray()
    return X[:, indices]


def get_model_and_data(row):
    """Train the identified model with the same official split and config."""
    X_train, X_test, y_train, y_test, feature_names = load_official_data()
    feature_configurations = joblib.load(FEATURE_CONFIG_PATH)
    selected_features = feature_configurations[row["feature_configuration"]]

    X_train_selected = select_feature_subset(X_train, feature_names, selected_features)
    X_test_selected = select_feature_subset(X_test, feature_names, selected_features)

    X_train_resampled, y_train_resampled, model_options = apply_imbalance_strategy(
        X_train_selected,
        y_train.to_numpy(),
        strategy=row["imbalance_strategy"],
    )

    model = build_model(row["model"], **model_options)
    model.fit(X_train_resampled, y_train_resampled)
    return model, X_train_resampled, X_test_selected, y_test.to_numpy(), selected_features


def positive_class_shap_values(shap_values):
    """Return SHAP values for the purchase class (True / class 1)."""
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[1], dtype=float)
    elif hasattr(shap_values, "values"):
        arr = np.asarray(shap_values.values, dtype=float)
    else:
        arr = np.asarray(shap_values, dtype=float)

    if arr.ndim == 3:
        if arr.shape[-1] == 2:
            return arr[:, :, 1]
        return arr[:, :, 0]
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    raise ValueError(f"Unsupported SHAP structure for purchase-class explanation: {arr.shape}")


def make_feature_ranking(values, feature_names):
    """Create a ranked feature table from absolute SHAP values."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        arr = arr[:, :, 1]
    mean_abs = np.abs(arr).mean(axis=0)
    return pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)


def save_global_plot(values, feature_names, path, title):
    """Save a global feature-importance plot based on SHAP values."""
    ranking = make_feature_ranking(values, feature_names).head(10)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=ranking, x="mean_abs_shap", y="feature", palette="viridis")
    plt.title(title)
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_local_plot(values, feature_names, path, title):
    """Save a local explanation for the most influential sample."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        arr = arr[:, :, 1]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D SHAP value matrix for local plotting, got {arr.shape}.")
    if arr.shape[0] == 0:
        raise ValueError("No SHAP values were generated for the selected sample.")

    sample_idx = int(np.argmax(np.abs(arr).sum(axis=1)))
    local_values = arr[sample_idx]
    ranking = pd.DataFrame({
        "feature": feature_names,
        "shap_value": local_values,
    }).sort_values("shap_value", key=lambda s: s.abs(), ascending=False).head(10)

    plt.figure(figsize=(10, 7))
    sns.barplot(data=ranking, x="shap_value", y="feature", palette="coolwarm")
    plt.title(title)
    plt.xlabel("SHAP value for purchase class")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def explain_tree_model(model, X_train_resampled, X_test_selected, feature_names, label_prefix):
    """Use TreeExplainer for tree models and save SHAP summary/local plots."""
    background = X_train_resampled[: min(50, len(X_train_resampled))]
    explainer = shap.TreeExplainer(model, data=background, feature_names=feature_names)
    sample = X_test_selected[: min(10, len(X_test_selected))]
    raw_values = explainer.shap_values(sample, check_additivity=False)
    values = positive_class_shap_values(raw_values)

    feature_ranking = make_feature_ranking(values, feature_names)
    global_path = FIGURES_DIR / f"{label_prefix}_shap_global_summary.png"
    local_path = FIGURES_DIR / f"{label_prefix}_shap_local_example.png"
    save_global_plot(values, feature_names, global_path, f"Global SHAP importance for {label_prefix}")
    save_local_plot(values, feature_names, local_path, f"Local SHAP explanation for {label_prefix}")
    return feature_ranking, global_path, local_path


def explain_kernel_model(model, X_train_resampled, X_test_selected, feature_names, label_prefix):
    """Use KernelExplainer for non-tree models with a small background sample."""
    background = X_train_resampled[: min(50, len(X_train_resampled))]
    sample = X_test_selected[: min(10, len(X_test_selected))]
    explainer = shap.KernelExplainer(model.predict_proba, background, feature_names=feature_names)
    raw_values = explainer.shap_values(sample, silent=True)
    values = positive_class_shap_values(raw_values)

    feature_ranking = make_feature_ranking(values, feature_names)
    global_path = FIGURES_DIR / f"{label_prefix}_shap_global_summary.png"
    local_path = FIGURES_DIR / f"{label_prefix}_shap_local_example.png"
    save_global_plot(values, feature_names, global_path, f"Global SHAP importance for {label_prefix}")
    save_local_plot(values, feature_names, local_path, f"Local SHAP explanation for {label_prefix}")
    return feature_ranking, global_path, local_path


def explain_configuration(row, label_prefix):
    """Fit the selected model and generate SHAP explanations for that configuration."""
    model, X_train_resampled, X_test_selected, _, selected_features = get_model_and_data(row)
    if row["model"] in {"random_forest", "decision_tree"}:
        feature_ranking, global_path, local_path = explain_tree_model(
            model,
            X_train_resampled,
            X_test_selected,
            selected_features,
            label_prefix,
        )
    else:
        feature_ranking, global_path, local_path = explain_kernel_model(
            model,
            X_train_resampled,
            X_test_selected,
            selected_features,
            label_prefix,
        )

    summary = {
        "model": row["model"],
        "feature_configuration": row["feature_configuration"],
        "imbalance_strategy": row["imbalance_strategy"],
        "positive_class": "Revenue=True",
        "positive_class_note": "SHAP values are selected for class 1, which represents the purchase class True.",
        "top_features": feature_ranking.head(10).to_dict(orient="records"),
    }

    summary_path = RESULTS_DIR / f"{label_prefix}_shap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, global_path, local_path


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_PATH)
    if len(df) != 48:
        raise ValueError(f"Expected 48 experiment rows, found {len(df)} rows.")

    best_f1_row = df.sort_values("f1", ascending=False).iloc[0]
    best_recall_row = df.sort_values("recall", ascending=False).iloc[0]

    summary_f1, global_f1_path, local_f1_path = explain_configuration(best_f1_row, "best_f1")
    summary_recall, global_recall_path, local_recall_path = explain_configuration(best_recall_row, "best_recall")

    print("Best F1 configuration:")
    print(best_f1_row[["model", "feature_configuration", "imbalance_strategy", "f1", "recall", "roc_auc"]].to_dict())
    print("Top features:")
    print(json.dumps(summary_f1["top_features"][:5], indent=2))

    print("\nBest Recall configuration:")
    print(best_recall_row[["model", "feature_configuration", "imbalance_strategy", "f1", "recall", "roc_auc"]].to_dict())
    print("Top features:")
    print(json.dumps(summary_recall["top_features"][:5], indent=2))

    print("\nSaved SHAP files:")
    print(f" - {RESULTS_DIR / 'best_f1_shap_summary.json'}")
    print(f" - {global_f1_path}")
    print(f" - {local_f1_path}")
    print(f" - {RESULTS_DIR / 'best_recall_shap_summary.json'}")
    print(f" - {global_recall_path}")
    print(f" - {local_recall_path}")
