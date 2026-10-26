# PageValues Target-Leakage & Pure Browsing Behavior Ablation Study

## 1. Executive Summary

In Google Analytics, `PageValues` is an analytics-derived metric calculated from historical e-commerce transactions on pages visited prior to completing a transaction. Because of this formula, `PageValues` is at severe risk of **target leakage** if used in real-time scoring of active browsing sessions.

As committed in the project synopsis (*"PageValues is expected to dominate, consistent with prior literature, tested for robustness here"*), this ablation study systematically compares model performance and SHAP explanations **with** versus **without** `PageValues`.

---

## 2. Benchmark Comparison (With vs. Without `PageValues`)

Evaluated on the held-out test set (2,466 sessions; 382 purchases) and 5-fold Stratified Cross-Validation:

| Model Configuration | Condition | Recall (Test) | F1-Score (Test) | ROC-AUC (Test) | 5-Fold CV ROC-AUC |
|---|---|---:|---:|---:|---:|
| **Random Forest (Baseline)** | With PageValues | 0.4764 | 0.5938 | 0.9135 | 0.9228 ± 0.0033 |
| | **Without PageValues** | **0.1126** | **0.1866** | **0.7501** | **0.7697 ± 0.0078** |
| **Random Forest (SMOTE)** | With PageValues | 0.6387 | 0.6362 | 0.9161 | 0.9889 ± 0.0017 |
| | **Without PageValues** | **0.2356** | **0.2936** | **0.7529** | **0.9731 ± 0.0010** |
| **Logistic Regression (Baseline)** | With PageValues | 0.3586 | 0.4824 | 0.8828 | 0.8944 ± 0.0061 |
| | **Without PageValues** | **0.0340** | **0.0636** | **0.7340** | **0.7518 ± 0.0144** |
| **Logistic Regression (Class Weight)** | With PageValues | 0.7435 | 0.5917 | 0.8932 | 0.9044 ± 0.0065 |
| | **Without PageValues** | **0.7487** | **0.3852** | **0.7348** | **0.7526 ± 0.0154** |
| **SVM (Baseline)** | With PageValues | 0.4686 | 0.5629 | 0.8824 | 0.8815 ± 0.0087 |
| | **Without PageValues** | **0.0026** | **0.0052** | **0.6887** | **0.6844 ± 0.0236** |
| **SVM (Class Weight)** | With PageValues | 0.7513 | 0.6336 | 0.8979 | 0.9064 ± 0.0069 |
| | **Without PageValues** | **0.6885** | **0.3771** | **0.7360** | **0.7548 ± 0.0087** |

---

## 3. Key Findings

1. **Catastrophic Recall Collapse in Unweighted Baselines**:
   * Without `PageValues` and without imbalance mitigation, baseline models almost completely fail to detect purchasing sessions.
   * SVM baseline recall collapsed from **46.9% down to 0.26%**, and Logistic Regression collapsed from **35.9% to 3.4%**.
   * This proves that unweighted baseline models were heavily reliant on `PageValues` to separate the classes.

2. **Imbalance Strategies are Essential for Real-Time Prediction**:
   * When `PageValues` is unavailable, class-weighting and resampling become **mandatory**:
     * `Logistic Regression (Class Weight)` recovered purchase recall to **74.87%**.
     * `SVM (Class Weight)` recovered purchase recall to **68.85%**.

3. **Realistic Discrimination Ceiling of Pure Browsing Behavior**:
   * Removing `PageValues` lowers the maximum ROC-AUC from **~0.92 to ~0.75**.
   * An ROC-AUC of 0.75 demonstrates that raw clickstream dynamics (durations, bounce rates, visit counts, month) still provide statistically informative and actionable discrimination above random guessing (0.50).

---

## 4. SHAP Feature Attribution Shift

When `PageValues` is present, it dominates the explanations (mean |SHAP| = **0.1724**, over 6× higher than any other feature).

When `PageValues` is removed, the true behavioral signals emerge as the leading predictors:

| Rank | Top Features (WITH PageValues) | Mean \|SHAP\| | Top Features (WITHOUT PageValues / Pure Behaviour) | Mean \|SHAP\| |
|:---:|:---|:---:|:---|:---:|
| 1 | `PageValues` | 0.1724 | **`ExitRates`** | 0.0477 |
| 2 | `Month_Nov` | 0.0270 | **`ProductRelated_Duration`** | 0.0411 |
| 3 | `ExitRates` | 0.0200 | **`Administrative_Duration`** | 0.0383 |
| 4 | `Administrative_Duration` | 0.0176 | **`Month_Nov`** | 0.0270 |
| 5 | `Administrative` | 0.0145 | **`ProductRelated`** | 0.0198 |
| 6 | `Month_May` | 0.0126 | **`Administrative`** | 0.0177 |
| 7 | `TrafficType_2` | 0.0123 | **`TrafficType_2`** | 0.0152 |
| 8 | `ProductRelated_Duration` | 0.0122 | **`BounceRates`** | 0.0147 |
| 9 | `VisitorType_New_Visitor` | 0.0113 | **`OperatingSystems_2`** | 0.0130 |
| 10 | `BounceRates` | 0.0112 | **`VisitorType_Returning_Visitor`** | 0.0110 |

**Interpretation**: In live session monitoring where `PageValues` is absent, retailers should focus on **session engagement depth** (`ProductRelated_Duration` and `Administrative_Duration`), **visitor hesitation/exit risk** (`ExitRates` and `BounceRates`), and **seasonal intent** (`Month_Nov`).

---

## 5. Artifacts Generated

- Script: [`src/ablation_pagevalues.py`](file:///Users/nayantaranambiar/FML_OnlineShopper_Project/src/ablation_pagevalues.py)
- Metrics CSV: [`results/nayantara/pagevalues_ablation_results.csv`](file:///Users/nayantaranambiar/FML_OnlineShopper_Project/results/nayantara/pagevalues_ablation_results.csv)
- SHAP Summary JSON: [`results/nayantara/pagevalues_ablation_shap_summary.json`](file:///Users/nayantaranambiar/FML_OnlineShopper_Project/results/nayantara/pagevalues_ablation_shap_summary.json)
- Performance Plot: [`figures/nayantara/pagevalues_ablation_performance.png`](file:///Users/nayantaranambiar/FML_OnlineShopper_Project/figures/nayantara/pagevalues_ablation_performance.png)
- SHAP Comparison Plot: [`figures/nayantara/pagevalues_ablation_shap_comparison.png`](file:///Users/nayantaranambiar/FML_OnlineShopper_Project/figures/nayantara/pagevalues_ablation_shap_comparison.png)
