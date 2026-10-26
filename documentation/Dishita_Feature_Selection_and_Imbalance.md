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
| `correlation` | Remove numerical features whose absolute pairwise correlation exceeds 0.85; retain one-hot categorical features | 73 |
| `rfe` | Recursive Feature Elimination using Logistic Regression | 40 |
| `tree_importance` | Rank features using an Extra Trees classifier and retain the top 40 | 40 |

### Correlation filtering

Absolute pairwise correlations are calculated using the processed training matrix. With the threshold set at 0.85, two collinear numerical features are removed:
1. `numerical__ProductRelated_Duration` (correlated with `numerical__ProductRelated` at $r \approx 0.8537$)
2. `numerical__ExitRates` (correlated with `numerical__BounceRates` at $r \approx 0.9133$)

This produces a genuinely reduced 73-feature set while retaining all one-hot categorical indicators.

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

### SMOTE limitation & SMOTENC resolution

Ordinary SMOTE applied to a one-hot encoded matrix can produce fractional values in categorical dummy columns (e.g. `Weekend_True = 0.38`), violating one-hot encoding semantics.

To resolve this:
1. **`SMOTENC` on Raw Data**: `src/imbalance.py` implements `resample_raw_with_smotenc`, which applies `SMOTENC` directly to the raw feature table prior to one-hot encoding. Continuous numerical features are interpolated, while nominal features are selected from majority nearest neighbours, guaranteeing valid discrete category values.
2. **Categorical Dummy Snapping**: For processed matrices, `snap_categorical_dummies` threshold-snaps one-hot indicators to discrete `{0.0, 1.0}`, removing fractional artifacts.
3. **In-Fold Cross-Validation Pipeline**: `src/cross_validation.py` packages `SMOTENC`, standardisation, feature selection, and the classifier into an `imblearn.pipeline.Pipeline`, guaranteeing that oversampling and preprocessing are executed strictly inside training folds with zero validation leakage.

## 4. Data leakage safeguards

- Feature-selection methods are fitted using the official training matrix only.
- In cross-validation (`src/cross_validation.py`), `SMOTENC` and standardisation are fitted strictly inside each training fold using `imblearn.pipeline.Pipeline`.
- Neither validation folds nor the held-out test set are ever resampled.
- The official held-out test set remains at 2,466 observations and 75 features.

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
