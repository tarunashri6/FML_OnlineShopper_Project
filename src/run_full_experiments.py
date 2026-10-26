"""Run the complete 48-model experiment grid for the shopper dataset.

This script uses the project’s existing modules and keeps the output under the
Nayantara results folder only.
"""

try:
    from src.experiment_runner import run_experiment_grid
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from experiment_runner import run_experiment_grid


if __name__ == "__main__":
    run_experiment_grid()
