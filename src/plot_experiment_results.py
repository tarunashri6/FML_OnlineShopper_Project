"""Create comparison plots for the full experiment grid.

This module reads the saved Nayantara CSV, verifies the result integrity, and
creates clear visual comparisons for F1, purchase recall, ROC-AUC, and the best
configurations. It does not modify the experiment results file itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "nayantara" / "full_experiment_results.csv"
OUTPUT_DIR = ROOT / "figures" / "nayantara"


def verify_results(df: pd.DataFrame) -> None:
    """Check that the CSV contains exactly 48 unique combinations and no missing metrics."""
    if len(df) != 48:
        raise ValueError(f"Expected 48 rows, found {len(df)} rows.")

    unique_configs = df[["model", "feature_configuration", "imbalance_strategy"]].drop_duplicates()
    if len(unique_configs) != 48:
        raise ValueError(
            "Expected 48 unique configurations, found "
            f"{len(unique_configs)} unique combinations."
        )

    required = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    missing = [metric for metric in required if metric not in df.columns or df[metric].isna().any()]
    if missing:
        raise ValueError(f"Missing or null metric values: {missing}")


def select_best_configs(df: pd.DataFrame):
    """Return the best F1 and best recall rows from the actual CSV."""
    best_f1 = df.sort_values("f1", ascending=False).iloc[0]
    best_recall = df.sort_values("recall", ascending=False).iloc[0]
    return best_f1, best_recall


def plot_f1_by_model_and_imbalance(df: pd.DataFrame):
    """Compare F1 scores across models and imbalance strategies."""
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=df,
        x="model",
        y="f1",
        hue="imbalance_strategy",
        palette="Set2",
    )
    plt.title("F1 score by model and imbalance strategy")
    plt.xlabel("Model")
    plt.ylabel("F1 score")
    plt.ylim(0, 1)
    plt.legend(title="Imbalance strategy")
    plt.tight_layout()
    path = OUTPUT_DIR / "f1_by_model_and_imbalance.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_recall_by_configuration(df: pd.DataFrame):
    """Compare purchase recall across all configurations."""
    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=df,
        x="feature_configuration",
        y="recall",
        hue="model",
        palette="tab10",
    )
    plt.title("Purchase recall across all feature configurations")
    plt.xlabel("Feature configuration")
    plt.ylabel("Recall for purchase class (True)")
    plt.ylim(0, 1)
    plt.legend(title="Model")
    plt.tight_layout()
    path = OUTPUT_DIR / "recall_by_configuration.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_roc_auc_by_model(df: pd.DataFrame):
    """Compare ROC-AUC across models irrespective of feature configuration."""
    model_summary = df.groupby("model", as_index=False)["roc_auc"].mean()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=model_summary, x="model", y="roc_auc", palette="viridis", hue="model", dodge=False)
    plt.title("Average ROC-AUC by model")
    plt.xlabel("Model")
    plt.ylabel("Mean ROC-AUC")
    plt.ylim(0.5, 1.0)
    plt.legend([], frameon=False)
    plt.tight_layout()
    path = OUTPUT_DIR / "roc_auc_by_model.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_best_f1_confusion_matrix(df: pd.DataFrame):
    """Plot the confusion matrix for the best F1 configuration from the CSV."""
    best_f1 = df.sort_values("f1", ascending=False).iloc[0]
    model = best_f1["model"]
    feat = best_f1["feature_configuration"]
    strategy = best_f1["imbalance_strategy"]

    cm = pd.DataFrame(
        [[int(best_f1["tn"]), int(best_f1["fp"])], [int(best_f1["fn"]), int(best_f1["tp"]) ]],
        index=["Actual False", "Actual True"],
        columns=["Predicted False", "Predicted True"],
    )

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title(f"Confusion matrix: {model} | {feat} | {strategy} (best F1)")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    path = OUTPUT_DIR / "best_f1_confusion_matrix.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path, best_f1


def plot_best_recall_vs_best_f1(df: pd.DataFrame):
    """Compare the best recall and best F1 runs on the same metrics plot."""
    best_f1, best_recall = select_best_configs(df)
    selected = pd.DataFrame([
        {
            "label": "Best F1",
            "model": best_f1["model"],
            "feature_configuration": best_f1["feature_configuration"],
            "imbalance_strategy": best_f1["imbalance_strategy"],
            "f1": best_f1["f1"],
            "recall": best_f1["recall"],
            "roc_auc": best_f1["roc_auc"],
        },
        {
            "label": "Best Recall",
            "model": best_recall["model"],
            "feature_configuration": best_recall["feature_configuration"],
            "imbalance_strategy": best_recall["imbalance_strategy"],
            "f1": best_recall["f1"],
            "recall": best_recall["recall"],
            "roc_auc": best_recall["roc_auc"],
        },
    ])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["f1", "recall", "roc_auc"]
    labels = ["F1", "Purchase recall", "ROC-AUC"]

    for ax, metric, label in zip(axes, metrics, labels):
        sns.barplot(data=selected, x="label", y=metric, palette="Set2", hue="label", dodge=False, ax=ax)
        ax.set_title(f"{label} comparison")
        ax.set_xlabel("Configuration")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.05)
        ax.legend([], frameon=False)

    fig.suptitle("Best F1 vs Best Recall configuration")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUTPUT_DIR / "best_f1_vs_best_recall.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path, selected


def main():
    """Create all comparison plots and store them under figures/nayantara/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    verify_results(df)

    paths = {
        "f1_by_model_and_imbalance": plot_f1_by_model_and_imbalance(df),
        "recall_by_configuration": plot_recall_by_configuration(df),
        "roc_auc_by_model": plot_roc_auc_by_model(df),
    }

    best_f1_matrix_path, best_f1 = plot_best_f1_confusion_matrix(df)
    paths["best_f1_confusion_matrix"] = best_f1_matrix_path

    best_compare_path, _ = plot_best_recall_vs_best_f1(df)
    paths["best_recall_vs_best_f1"] = best_compare_path

    # Save a small metadata file with the selected best configurations.
    best_f1_row, best_recall_row = select_best_configs(df)
    metadata = {
        "best_f1": {
            "model": best_f1_row["model"],
            "feature_configuration": best_f1_row["feature_configuration"],
            "imbalance_strategy": best_f1_row["imbalance_strategy"],
            "f1": float(best_f1_row["f1"]),
            "recall": float(best_f1_row["recall"]),
            "roc_auc": float(best_f1_row["roc_auc"]),
        },
        "best_recall": {
            "model": best_recall_row["model"],
            "feature_configuration": best_recall_row["feature_configuration"],
            "imbalance_strategy": best_recall_row["imbalance_strategy"],
            "f1": float(best_recall_row["f1"]),
            "recall": float(best_recall_row["recall"]),
            "roc_auc": float(best_recall_row["roc_auc"]),
        },
    }
    (OUTPUT_DIR / "best_configuration_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Validated CSV: 48 unique configurations, no missing metrics.")
    print("Plots saved to:")
    for key, path in paths.items():
        print(f" - {key}: {path}")


if __name__ == "__main__":
    main()
