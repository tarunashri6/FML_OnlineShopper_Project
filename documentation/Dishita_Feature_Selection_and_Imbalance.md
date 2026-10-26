# Dishita — Feature Selection and Class-Imbalance Handling

## 1. Objective

This component prepares reusable feature-selection configurations and class-imbalance strategies for the Online Shoppers Purchasing Intention classification project.

The implementation uses the team's existing processed training data and official train/test split. It does not create a new split or modify the held-out test set.

## 2. Feature selection

The processed dataset contains 75 features: 10 numerical features and 65 one-hot encoded categorical features.

Four configurations are generated:

| Configuration | Method | Selected features |
|---|---|---:|
| `all` | Use all processed features | 75 |
| `correlation` | Remove numerical features whose absolute pairwise correlation exceeds 0.95; retain one-hot categorical features | 75 |
| `rfe` | Recursive Feature Elimination using Logistic Regression | 40 |
| `tree_importance` | Rank features using an Extra Trees classifier and retain the top 40 | 40 |

### Correlation filtering

Absolute pairwise correlations are calculated using the processed training matrix. The current training data has no numerical feature pair exceeding the 0.95 threshold, so no features are removed.

This result does not imply that every feature is independent. It only means that the implemented numerical-feature filtering rule found no pair above its threshold. One-hot categorical columns are retained by the current method.

### Recursive Feature Elimination

RFE uses Logistic Regression with `max_iter=2000`, `random_state=42`, and `step=0.1`. The implementation selects 40 features from the 75 available features.

The selected feature count is a project configuration, not a claim that 40 is the optimal count.

### Tree-based importance

An Extra Trees classifier uses 200 estimators, `random_state=42`, and `n_jobs=-1`. Features are ranked by the fitted model's feature importance, and the top 40 are retained.

Feature importance is a ranking criterion, not proof of causal importance or improved test performance.

## 3. Class-imbalance strategies

The official training set contains 9,864 observations:

| Target class | Meaning | Count | Proportion |
|---|---|---:|---:|
| `False` | No purchase | 8,338 | 84.53% |
| `True` | Purchase | 1,526 | 15.47% |

Three configurations are provided:

- **`none`:** use the original training observations without resampling.
- **`smote`:** apply SMOTE to the training matrix with `random_state=42`. The validated output contains 8,338 observations in each class, for 16,676 training observations.
- **`class_weight`:** preserve the original training observations and return `class_weight='balanced'` as a model-fitting option.

Class weighting must be passed to a compatible estimator; it is not a resampling operation.

### SMOTE limitation

The shared processed matrix includes one-hot encoded categorical features. Ordinary SMOTE can produce fractional values in these columns, so synthetic rows may not represent valid one-hot categorical records. This limitation must be considered when interpreting the results. A categorical-aware alternative should be discussed with the team before the final comparison.

## 4. Data leakage safeguards

- Feature-selection methods are fitted using the official training matrix only.
- SMOTE is applied to training data only.
- The held-out test set is not resampled.
- The official test set remains at 2,466 observations and 75 features.
- Feature selection and imbalance handling should be applied consistently within the team's experiment workflow.

## 5. Generated outputs

The feature-selection script writes its outputs under `results/feature_selection/`:

- `all_features.csv`
- `correlation_features.csv`
- `rfe_features.csv`
- `tree_importance_features.csv`
- `correlation_removed_features.csv`
- `rfe_feature_ranking.csv`
- `tree_feature_importance.csv`
- `feature_selection_summary.csv`
- `feature_configurations.joblib`

The implementation is in `src/feature_selection.py`. Class-imbalance handling is in `src/imbalance.py`.

## 6. Validation status

The feature-selection script completed successfully and reported feature counts of 75, 75, 40, and 40 for the four configurations.

The imbalance validation confirmed that `none` and `class_weight` preserve the original training distribution, SMOTE balances the training classes, and the official test set remains unchanged.

These checks validate the implementation and data handling. They do not establish which configuration gives the best predictive performance; that conclusion must come from the team's model evaluation.
