
"""Create feature-selection figures for the Online Shoppers project."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import load_npz


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "feature_selection"
FIGURES_DIR = ROOT / "figures" / "dishita"


def plot_tree_importance():
    importance_path = RESULTS_DIR / "tree_feature_importance.csv"
    importance = pd.read_csv(importance_path).head(15)
    importance = importance.sort_values("importance")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))
    plt.barh(importance["feature"], importance["importance"])
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title("Top 15 Features — Extra Trees Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tree_feature_importance_top15.png", dpi=200)
    plt.close()


def plot_numeric_correlations():
    X = load_npz(DATA_DIR / "X_train_processed.npz")
    feature_names = pd.read_csv(
        DATA_DIR / "feature_names.csv"
    )["Feature"].tolist()

    numeric_indices = [
        i for i, name in enumerate(feature_names)
        if name.startswith("numerical__")
    ]
    numeric_names = [feature_names[i] for i in numeric_indices]

    numeric_data = X[:, numeric_indices].toarray()
    frame = pd.DataFrame(numeric_data, columns=numeric_names)
    correlation = frame.corr()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(correlation, vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(len(numeric_names)))
    axis.set_xticklabels(numeric_names, rotation=90)
    axis.set_yticks(range(len(numeric_names)))
    axis.set_yticklabels(numeric_names)
    axis.set_title("Numerical Feature Correlation Matrix")
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "numeric_feature_correlation.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    plot_tree_importance()
    plot_numeric_correlations()
    print("Feature-selection figures saved to:", FIGURES_DIR)
