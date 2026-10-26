"""Runner for a single validation experiment.

This is intentionally scoped to one configuration to validate the interfaces and
model-processing flow before running all 48 experiments.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import load_npz

try:
    from src.evaluation import evaluate_predictions, get_model_scores
    from src.imbalance import apply_imbalance_strategy
    from src.model_builders import build_model
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from evaluation import evaluate_predictions, get_model_scores
    from imbalance import apply_imbalance_strategy
    from model_builders import build_model


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
FEATURE_CONFIG_PATH = ROOT / "results" / "feature_selection" / "feature_configurations.joblib"
OUTPUT_DIR = ROOT / "results" / "nayantara"
MODEL_NAMES = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "svm",
]
FEATURE_CONFIGS = ["all", "correlation", "rfe", "tree_importance"]
IMBALANCE_STRATEGIES = ["none", "smote", "class_weight"]


def load_official_data():
    """Load the processed train/test arrays and labels from the project spec."""
    X_train = load_npz(DATA_DIR / "X_train_processed.npz")
    X_test = load_npz(DATA_DIR / "X_test_processed.npz")

    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["Revenue"].astype(bool)
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["Revenue"].astype(bool)
    feature_names = pd.read_csv(DATA_DIR / "feature_names.csv")["Feature"].tolist()

    if X_train.shape[0] != len(y_train):
        raise ValueError("Training matrix rows do not match the training labels.")
    if X_test.shape[0] != len(y_test):
        raise ValueError("Test matrix rows do not match the test labels.")
    if X_train.shape[1] != len(feature_names):
        raise ValueError("Training matrix columns do not match the saved feature names.")

    return X_train, X_test, y_train, y_test, feature_names


def select_feature_subset(X, feature_names, selected_features):
    """Select the requested feature columns while preserving sparse format."""
    missing = [name for name in selected_features if name not in feature_names]
    if missing:
        raise ValueError(f"Selected features not found in feature_names: {missing}")

    indices = [feature_names.index(name) for name in selected_features]

    if hasattr(X, "__getitem__"):
        return X[:, indices]

    return X[:, indices]


def run_single_experiment(model_name="logistic_regression", feature_config="all", imbalance_strategy="none"):
    """Run one experiment and return a single-row metrics DataFrame."""
    X_train, X_test, y_train, y_test, feature_names = load_official_data()

    feature_configurations = joblib.load(FEATURE_CONFIG_PATH)
    if feature_config not in feature_configurations:
        raise ValueError(
            f"Unknown feature config {feature_config!r}. "
            f"Available configs: {sorted(feature_configurations.keys())}"
        )

    selected_features = feature_configurations[feature_config]
    X_train_selected = select_feature_subset(X_train, feature_names, selected_features)
    X_test_selected = select_feature_subset(X_test, feature_names, selected_features)

    X_train_resampled, y_train_resampled, model_options = apply_imbalance_strategy(
        X_train_selected,
        y_train.to_numpy(),
        strategy=imbalance_strategy,
    )

    model = build_model(model_name, **model_options)
    model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test_selected)
    y_score = get_model_scores(model, X_test_selected)
    metrics = evaluate_predictions(y_test.to_numpy(), y_pred, y_score)

    record = {
        "model": model_name,
        "feature_configuration": feature_config,
        "imbalance_strategy": imbalance_strategy,
        "positive_class": "Revenue=True",
        "train_rows": int(X_train_selected.shape[0]),
        "test_rows": int(X_test_selected.shape[0]),
        "selected_feature_count": int(len(selected_features)),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
    }

    return pd.DataFrame([record])


def save_result_table(df, file_name="single_validation_result.csv"):
    """Persist the experiment result under results/nayantara/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / file_name
    df.to_csv(path, index=False)
    return path


def run_experiment_grid():
    """Run all 48 agreed combinations and record any failures explicitly.

    The output is saved to results/nayantara/ without touching Taruni or Dishita
    artifacts.
    """
    expected_combinations = list(product(MODEL_NAMES, FEATURE_CONFIGS, IMBALANCE_STRATEGIES))
    results = []
    failures = []

    for model_name, feature_config, imbalance_strategy in expected_combinations:
        try:
            result = run_single_experiment(
                model_name=model_name,
                feature_config=feature_config,
                imbalance_strategy=imbalance_strategy,
            )
            result_row = result.iloc[0].to_dict()
            results.append(result_row)
        except Exception as exc:  # pragma: no cover - failure logging is intentional
            failures.append(
                {
                    "model": model_name,
                    "feature_configuration": feature_config,
                    "imbalance_strategy": imbalance_strategy,
                    "error": str(exc),
                }
            )

    result_df = pd.DataFrame(results)
    failures_df = pd.DataFrame(failures)

    if len(expected_combinations) != 48:
        raise ValueError(f"Expected 48 combinations, got {len(expected_combinations)}.")

    if len(result_df) + len(failures_df) != 48:
        raise ValueError(
            "Experiment count mismatch: "
            f"successful={len(result_df)}, failed={len(failures_df)}, total={len(result_df) + len(failures_df)}"
        )

    if result_df.empty:
        raise ValueError("No experiment results were generated. Check the run logs or failures.")

    deduped = result_df[["model", "feature_configuration", "imbalance_strategy"]].drop_duplicates()
    if len(deduped) != 48:
        raise ValueError(
            "Duplicate or missing configuration values detected. "
            f"Expected 48 unique combos, found {len(deduped)}."
        )

    required_metrics = [
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
    missing_metrics = [metric for metric in required_metrics if metric not in result_df.columns]
    if missing_metrics:
        raise ValueError(f"Missing required metrics: {missing_metrics}")

    result_path = OUTPUT_DIR / "full_experiment_results.csv"
    failures_path = OUTPUT_DIR / "full_experiment_failures.csv"
    summary_path = OUTPUT_DIR / "full_experiment_summary.json"

    result_df.to_csv(result_path, index=False)
    if not failures_df.empty:
        failures_df.to_csv(failures_path, index=False)
    else:
        pd.DataFrame(columns=["model", "feature_configuration", "imbalance_strategy", "error"]).to_csv(
            failures_path,
            index=False,
        )

    best_recall = result_df.sort_values("recall", ascending=False).head(5)
    best_f1 = result_df.sort_values("f1", ascending=False).head(5)
    summary = {
        "expected_combinations": 48,
        "successful_runs": int(len(result_df)),
        "failed_runs": int(len(failures_df)),
        "missing_metric_columns": missing_metrics,
        "best_purchase_recall": best_recall[[
            "model",
            "feature_configuration",
            "imbalance_strategy",
            "recall",
            "f1",
            "roc_auc",
        ]].to_dict(orient="records"),
        "best_f1": best_f1[[
            "model",
            "feature_configuration",
            "imbalance_strategy",
            "f1",
            "recall",
            "roc_auc",
        ]].to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Expected combinations: {len(expected_combinations)}")
    print(f"Successful runs: {len(result_df)}")
    print(f"Failed runs: {len(failures_df)}")
    print(f"Results saved to: {result_path}")
    print(f"Failures saved to: {failures_path}")
    print(f"Summary saved to: {summary_path}")

    return result_df, failures_df, summary


if __name__ == "__main__":
    result = run_single_experiment(
        model_name="logistic_regression",
        feature_config="all",
        imbalance_strategy="none",
    )
    output_path = save_result_table(result, "single_validation_result.csv")
    print(result.to_string(index=False))
    print(f"\nSaved validation result to: {output_path}")
