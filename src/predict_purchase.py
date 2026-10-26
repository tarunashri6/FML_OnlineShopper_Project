"""Simple purchase-intention prediction prototype.

This script is intentionally small and beginner-friendly. It follows the same
training-only workflow used in the project experiments:

1. Load the official training data.
2. Reuse the saved preprocessing pipeline.
3. Select the best-F1 feature set from the experiment results.
4. Apply the same imbalance strategy as the winning configuration.
5. Train a fresh model on the official training set only.
6. Validate a raw customer session before prediction.

It does not accept the preprocessed 75-feature vector as input. The accepted
input is the original raw session data with the project fields from the
training CSV (excluding Revenue).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from src.imbalance import apply_imbalance_strategy
    from src.model_builders import build_model
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from imbalance import apply_imbalance_strategy
    from model_builders import build_model


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "nayantara"
FEATURE_CONFIG_PATH = ROOT / "results" / "feature_selection" / "feature_configurations.joblib"
PIPELINE_PATH = ROOT / "results" / "taruni" / "preprocessing_pipeline.joblib"
EXPERIMENT_RESULTS_PATH = RESULTS_DIR / "full_experiment_results.csv"
MODEL_ARTIFACT_PATH = RESULTS_DIR / "best_f1_purchase_prototype.joblib"
MODEL_METADATA_PATH = RESULTS_DIR / "best_f1_purchase_prototype_metadata.json"

RAW_INPUT_COLUMNS = [
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
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]


def load_processed_feature_names() -> list[str]:
    """Return the official processed feature names used in the experiment."""
    return pd.read_csv(DATA_DIR / "feature_names.csv")["Feature"].tolist()


def load_preprocessing_pipeline():
    """Load and validate the saved raw-to-processed preprocessing pipeline."""
    pipeline = joblib.load(PIPELINE_PATH)
    expected_columns = RAW_INPUT_COLUMNS
    actual_columns = list(getattr(pipeline, "feature_names_in_", []))

    if not actual_columns:
        raise ValueError(
            "The saved preprocessing pipeline is missing feature names. "
            "Expected the same raw feature order as the training data."
        )

    if actual_columns != expected_columns:
        raise ValueError(
            "The saved preprocessing pipeline column order does not match the "
            f"official raw feature order.\nExpected: {expected_columns}\nActual: {actual_columns}"
        )

    return pipeline


def get_best_f1_configuration() -> dict[str, Any]:
    """Read the actual experiment results and return the best-F1 row."""
    experiment_df = pd.read_csv(EXPERIMENT_RESULTS_PATH)
    if experiment_df.empty:
        raise ValueError(f"No experiment results found in {EXPERIMENT_RESULTS_PATH}.")

    best_row = experiment_df.sort_values("f1", ascending=False).iloc[0].to_dict()
    return best_row


def train_best_f1_prototype() -> dict[str, Any]:
    """Train the best-F1 prototype using the same project workflow."""
    pipeline = load_preprocessing_pipeline()
    feature_configurations = joblib.load(FEATURE_CONFIG_PATH)
    best_row = get_best_f1_configuration()
    selected_features = feature_configurations[best_row["feature_configuration"]]

    train_raw = pd.read_csv(DATA_DIR / "train_raw.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["Revenue"].astype(bool)
    processed_feature_names = load_processed_feature_names()

    missing = [name for name in selected_features if name not in processed_feature_names]
    if missing:
        raise ValueError(f"Selected feature names are missing from the processed feature list: {missing}")

    X_train_processed = pipeline.transform(train_raw[RAW_INPUT_COLUMNS])
    dense_train_processed = X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else np.asarray(X_train_processed)
    feature_frame = pd.DataFrame(dense_train_processed, columns=processed_feature_names)
    X_selected = feature_frame.loc[:, selected_features].to_numpy()

    X_resampled, y_resampled, model_options = apply_imbalance_strategy(
        X_selected,
        y_train.to_numpy(),
        strategy=best_row["imbalance_strategy"],
    )

    model = build_model(best_row["model"], **model_options)
    model.fit(X_resampled, y_resampled)

    bundle = {
        "model_name": best_row["model"],
        "feature_configuration": best_row["feature_configuration"],
        "imbalance_strategy": best_row["imbalance_strategy"],
        "positive_class": best_row["positive_class"],
        "raw_feature_columns": RAW_INPUT_COLUMNS,
        "selected_features": selected_features,
        "processed_feature_names": processed_feature_names,
        "preprocessing_pipeline": pipeline,
        "model": model,
        "best_f1_score": float(best_row["f1"]),
        "best_recall": float(best_row["recall"]),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_ARTIFACT_PATH)

    metadata = {
        "model_name": best_row["model"],
        "feature_configuration": best_row["feature_configuration"],
        "imbalance_strategy": best_row["imbalance_strategy"],
        "positive_class": best_row["positive_class"],
        "raw_feature_columns": RAW_INPUT_COLUMNS,
        "selected_features": selected_features,
        "best_f1_score": float(best_row["f1"]),
        "best_recall": float(best_row["recall"]),
        "roc_auc": float(best_row["roc_auc"]),
        "model_artifact": str(MODEL_ARTIFACT_PATH),
        "notes": [
            "This prototype uses the best-F1 configuration from the project experiment results.",
            "The accepted input is raw customer data, not the already-preprocessed 75-feature vector.",
            "PageValues is required for the most useful purchase-probability signal, and may be unavailable in some live sessions.",
        ],
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return bundle


def _to_bool(value: Any, field_name: str) -> bool:
    """Normalize boolean-like values from user input."""
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)

    if isinstance(value, str):
        text = value.strip().lower()
        lookup = {
            "true": True,
            "false": False,
            "yes": True,
            "no": False,
            "1": True,
            "0": False,
        }
        if text in lookup:
            return lookup[text]

    raise ValueError(f"Field '{field_name}' must be a boolean-like value (True/False, 1/0, yes/no).")


def _normalise_input_row(raw_session: dict[str, Any]) -> pd.DataFrame:
    """Validate and reorder one raw customer session for prediction."""
    if raw_session is None:
        raise ValueError("No session input was provided.")

    if isinstance(raw_session, pd.DataFrame):
        if raw_session.shape[0] != 1:
            raise ValueError("Please provide one session row only.")
        raw_session = raw_session.iloc[0].to_dict()

    if isinstance(raw_session, pd.Series):
        raw_session = raw_session.to_dict()

    if not isinstance(raw_session, dict):
        raise TypeError(
            "Session input must be a dictionary or a single-row DataFrame. "
            "Do not provide the already-preprocessed 75-feature vector."
        )

    if any(col.startswith("numerical__") or col.startswith("categorical__") for col in raw_session.keys()):
        raise ValueError(
            "This looks like a preprocessed feature vector, not raw customer input. "
            "Please provide the original session fields listed in the project documentation."
        )

    if "Revenue" in raw_session:
        raise ValueError(
            "Please do not include the target column 'Revenue' in the prediction input. "
            "This script expects raw customer attributes only."
        )

    missing = [name for name in RAW_INPUT_COLUMNS if name not in raw_session]
    if missing:
        raise ValueError(
            "Missing required raw input fields: " + ", ".join(missing) + ". "
            "These fields come from the original online shopper session data."
        )

    row = {name: raw_session[name] for name in RAW_INPUT_COLUMNS}

    for name in [
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
    ]:
        try:
            row[name] = float(row[name])
        except (TypeError, ValueError):
            raise ValueError(f"Field '{name}' should be numeric.") from None

    for name in ["Month", "VisitorType"]:
        value = str(row[name]).strip()
        if not value:
            raise ValueError(f"Field '{name}' is required.")
        row[name] = value

    row["Weekend"] = _to_bool(row["Weekend"], "Weekend")

    for name in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        try:
            row[name] = int(float(row[name]))
        except (TypeError, ValueError):
            raise ValueError(f"Field '{name}' should be an integer value.") from None

    return pd.DataFrame([row], columns=RAW_INPUT_COLUMNS)


def load_bundle() -> dict[str, Any]:
    """Load the saved prototype bundle, creating it if it is absent."""
    if not MODEL_ARTIFACT_PATH.exists():
        return train_best_f1_prototype()
    return joblib.load(MODEL_ARTIFACT_PATH)


def predict_purchase(raw_session: dict[str, Any], bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    """Predict purchase intent for one raw customer session."""
    bundle = bundle or load_bundle()
    pipeline = bundle["preprocessing_pipeline"]
    model = bundle["model"]
    selected_features = bundle["selected_features"]
    processed_names = bundle["processed_feature_names"]

    session_df = _normalise_input_row(raw_session)
    processed = pipeline.transform(session_df)
    dense_processed = processed.toarray() if hasattr(processed, "toarray") else np.asarray(processed)
    processed_df = pd.DataFrame(dense_processed, columns=processed_names)
    selected_df = processed_df.loc[:, selected_features]
    feature_array = selected_df.to_numpy()

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_array)[0]
        if len(probabilities) == 2:
            purchase_probability = float(probabilities[1])
        else:
            purchase_probability = None
    else:
        purchase_probability = None

    prediction = model.predict(feature_array)[0]
    purchase_class = bool(prediction)

    result = {
        "predicted_purchase": purchase_class,
        "purchase_probability": purchase_probability,
        "predicted_label": "purchase" if purchase_class else "no_purchase",
        "positive_class": "Revenue=True",
    }

    return result


def _build_demo_example() -> dict[str, Any]:
    """Return a real held-out raw session from the project test set for demonstration."""
    test_raw = pd.read_csv(DATA_DIR / "test_raw.csv")
    if test_raw.empty:
        raise ValueError("No test rows were found in data/processed/test_raw.csv")

    example = test_raw[test_raw["Revenue"] == True].head(1)
    if example.empty:
        example = test_raw.head(1)

    row = example.iloc[0].drop(labels=["Revenue"]).to_dict()
    return row


def _parse_session_json(raw_text: str) -> dict[str, Any]:
    """Parse a JSON string representing one session."""
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Input is not valid JSON. Please pass a dictionary-like JSON object.") from exc
    if not isinstance(payload, dict):
        raise ValueError("The JSON payload must be an object with raw customer fields.")
    return payload


def main() -> None:
    """Command-line interface for the prediction prototype."""
    parser = argparse.ArgumentParser(description="Predict purchase intent from one raw customer session.")
    parser.add_argument(
        "--session-json",
        type=str,
        default=None,
        help="JSON object with the raw customer values, such as Administrative, Month, PageValues, and VisitorType.",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Print a real held-out example from the test set and predict on it as a demonstration.",
    )
    args = parser.parse_args()

    if args.example:
        example = _build_demo_example()
        print("Demo input (held-out row, not independent validation):")
        print(json.dumps(example, indent=2, default=str))
        result = predict_purchase(example)
        print("\nPrediction:")
        print(json.dumps(result, indent=2))
        return

    if args.session_json is None:
        parser.error(
            "Provide raw customer data with --session-json or run with --example. "
            "The input must not be the already-preprocessed 75-feature vector."
        )

    session_input = _parse_session_json(args.session_json)
    result = predict_purchase(session_input)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
