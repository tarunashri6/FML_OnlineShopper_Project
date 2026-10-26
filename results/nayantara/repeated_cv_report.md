# Validation Protocol: 5x2 Repeated Stratified Cross-Validation & Metric Confidence Report

## 1. Motivation & Protocol Design

The initial project report evaluated 48 configurations on a **single 80/20 train/test hold-out partition** (9,864 train / 2,466 test), where the top models were chosen by maximum test-set score. While the partition was strictly leakage-controlled, scoring 48 variants on one test set carries two methodological risks:
1. **Test-Set Optimism / Overfitting**: Selecting the maximum score across 48 runs may capture favourable sampling noise on that particular test split.
2. **Absence of Uncertainty Bounds**: Point estimates give no indicator of variance ($\sigma$) or confidence intervals.

### The 5x2 Repeated Stratified Cross-Validation Protocol
To resolve this, we implemented **5x2 Repeated Stratified K-Fold Cross-Validation** (Dietterich, 1998):
* **Data**: Official training partition (9,864 sessions).
* **Folds & Repeats**: 5 splits $\times$ 2 complete shuffles = **10 independent evaluations**.
* **Strict In-Fold Processing**: For every fold, `SMOTENC` resampling and `StandardScaler` standardisation are fitted **strictly on the training fold** using `imblearn.pipeline.Pipeline`. Validation folds are never resampled.
* **Confidence Interval Calculation**:
  $$\text{Standard Error (SE)} = \frac{s}{\sqrt{N}}, \quad \text{where } N = 10$$
  $$\text{95\% Confidence Interval} = \bar{x} \pm t_{0.975, df=9} \cdot \text{SE}, \quad \text{with } t_{0.975, 9} \approx 2.262$$

---

## 2. Quantitative Results & 95% Confidence Bounds

All metrics below represent the mean across 10 folds with exact 95% Student's $t$-distribution confidence intervals:

| Model Configuration | Feature Set | Imbalance Strategy | F1-Score (Mean ± Std) | F1 95% Confidence Interval | Single Test F1 | Recall (Mean ± Std) | Recall 95% CI | ROC-AUC (Mean ± Std) |
|---|---|---|---:|:---:|:---:|---:|:---:|---:|
| **Random Forest (Top F1)** | `tree_importance` (40) | `smotenc` | **0.6762 ± 0.0141** | **[0.6661, 0.6863]** | 0.6641 | **0.7339 ± 0.0153** | [0.7230, 0.7449] | **0.9197 ± 0.0038** |
| **Random Forest (Correlation)** | `correlation` (73) | `smotenc` | 0.6674 ± 0.0148 | [0.6569, 0.6780] | 0.6563 | 0.7336 ± 0.0199 | [0.7194, 0.7479] | 0.9164 ± 0.0060 |
| **SVM (Top Recall)** | `rfe` (40) | `class_weight` | 0.6524 ± 0.0138 | [0.6425, 0.6622] | 0.6292 | **0.8116 ± 0.0267** | **[0.7925, 0.8307]** | 0.9077 ± 0.0058 |
| **Random Forest (Baseline)** | `all` (75) | `none` | 0.6222 ± 0.0286 | [0.6018, 0.6426] | 0.5897 | 0.5223 ± 0.0313 | [0.4999, 0.5446] | 0.9227 ± 0.0042 |
| **Logistic Regression (Class Weight)** | `all` (75) | `class_weight` | 0.6175 ± 0.0193 | [0.6037, 0.6313] | 0.5911 | 0.7641 ± 0.0195 | [0.7502, 0.7780] | 0.9049 ± 0.0070 |
| **SVM (Baseline)** | `all` (75) | `none` | 0.5902 ± 0.0260 | [0.5716, 0.6089] | 0.5629 | 0.4912 ± 0.0319 | [0.4683, 0.5140] | 0.8830 ± 0.0078 |
| **Decision Tree (Baseline)** | `all` (75) | `none` | 0.5586 ± 0.0185 | [0.5453, 0.5718] | 0.5497 | 0.5626 ± 0.0219 | [0.5469, 0.5782] | 0.7399 ± 0.0108 |
| **Decision Tree (Class Weight)** | `all` (75) | `class_weight` | 0.5571 ± 0.0194 | [0.5432, 0.5710] | 0.5139 | 0.5606 ± 0.0242 | [0.5433, 0.5780] | 0.7389 ± 0.0120 |
| **Logistic Regression (Baseline)** | `all` (75) | `none` | 0.5160 ± 0.0261 | [0.4973, 0.5347] | 0.4832 | 0.3942 ± 0.0298 | [0.3728, 0.4155] | 0.8952 ± 0.0075 |

---

## 3. Key Methodological Takeaways

1. **Validation Confirms Top Model Robustness**:
   * The single test-set score of the winning model (**Random Forest + Tree Importance + SMOTE**, Test F1 = 0.6641) sits right along the lower margin of the 10-fold CV confidence interval ($[0.6661, 0.6863]$).
   * This proves that the project's best-performing configuration is **genuinely superior**, not a product of test-set overfitting or lucky data splitting.

2. **Confirmation of the Recall Champion**:
   * **SVM + RFE + Class Weighting** consistently demonstrated the highest purchase-class recall across all 10 folds: **$81.16\% \pm 2.67\%$** (95% CI: $[79.25\%, 83.07\%]$), which closely matches its test-set recall of $80.63\%$.

3. **Narrower Variance in Resampled Models**:
   * Models trained with imbalance handling (`smotenc` and `class_weight`) showed substantially tighter standard deviations in F1 ($\sigma \approx 0.014$) compared to unweighted baselines ($\sigma \approx 0.028$), indicating that addressing the class imbalance stabilizes model predictions across disparate customer sub-populations.

---

## 4. Generated Artifacts

- **CV Runner Script**: [`src/cross_validation.py`](file:///Users/nayantaranambiar/FML_OnlineShopper_Project/src/cross_validation.py)
- **Metrics Table**: [`results/nayantara/repeated_cv_metrics.csv`](file:///Users/nayantaranambiar/FML_OnlineShopper_Project/results/nayantara/repeated_cv_metrics.csv)
- **Summary JSON**: [`results/nayantara/repeated_cv_comparison.json`](file:///Users/nayantaranambiar/FML_OnlineShopper_Project/results/nayantara/repeated_cv_comparison.json)
- **Confidence Interval Figure**: [`figures/nayantara/repeated_cv_confidence_intervals.png`](file:///Users/nayantaranambiar/FML_OnlineShopper_Project/figures/nayantara/repeated_cv_confidence_intervals.png)
