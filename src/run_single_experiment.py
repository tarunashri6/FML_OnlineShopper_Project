"""Convenience script for a single validation run.

This script deliberately runs exactly one configuration so we can validate the
project interfaces before expanding to all 48 experiments.
"""

try:
    from src.experiment_runner import run_single_experiment, save_result_table
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from experiment_runner import run_single_experiment, save_result_table


if __name__ == "__main__":
    result = run_single_experiment(
        model_name="logistic_regression",
        feature_config="all",
        imbalance_strategy="none",
    )
    output_path = save_result_table(result, file_name="single_validation_result.csv")
    print(result.to_string(index=False))
    print(f"\nSaved validation result to: {output_path}")
