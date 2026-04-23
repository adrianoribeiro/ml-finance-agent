# Model Card — Credit Risk Prediction

## Model Details

- **Name**: credit-risk-model
- **Version**: 1.0
- **Type**: Binary classification
- **Framework**: Scikit-learn (Logistic Regression), PyTorch (MLP)
- **Owner**: Adriano Ribeiro
- **Date**: April 2026

## Intended Use

Predict the probability of a client defaulting on financial obligations within 2 years. Designed to support credit analysts in decision-making — not to replace human judgment.

**Not intended for**: Automated credit denial without human review.

## Training Data

- **Source**: Give Me Some Credit (Kaggle)
- **Size**: 149,986 clients after cleaning (originally 150,000)
- **Target**: `SeriousDlqin2yrs` (1 = defaulted, 0 = paid)
- **Default rate**: 6.68% (imbalanced)
- **Split**: 80% train (119,988) / 20% test (29,998), stratified

## Features

| Feature | Type |
|---------|------|
| RevolvingUtilizationOfUnsecuredLines | float |
| age | int |
| DebtRatio | float |
| MonthlyIncome | float |
| NumberOfOpenCreditLinesAndLoans | int |
| NumberRealEstateLoansOrLines | int |
| NumberOfDependents | float |
| income_missing | binary (engineered) |
| total_late_payments | int (engineered) |
| has_late_payment | binary (engineered) |
| has_severe_late | binary (engineered) |

## Performance

| Model | AUC |
|-------|-----|
| MLP PyTorch (64-32, dropout) | 0.8675 |
| Logistic Regression (balanced, scaled) | 0.8620 |
| Random Forest (100 trees, balanced) | 0.8387 |

Production model: Logistic Regression (simpler, interpretable, similar AUC to MLP).

## Preprocessing

- Removed 14 records with invalid age (< 18 or > 100)
- Imputed MonthlyIncome and NumberOfDependents with median
- Created `income_missing` flag before imputation
- Capped RevolvingUtilization at 1.5 and DebtRatio at 99th percentile
- Combined 3 late payment columns into `total_late_payments`
- StandardScaler applied for Logistic Regression and MLP

## Limitations

- Dataset is from a Kaggle competition, may not reflect current market conditions
- No demographic features (gender, ethnicity) — avoids direct discrimination but may have proxy bias through income and age
- Model assumes feature distributions remain stable over time — drift detection is implemented to monitor this
- Binary classification only — does not predict severity of default or expected loss

## Ethical Considerations

- **Fairness**: `class_weight='balanced'` used to avoid ignoring the minority class (defaulters). No protected attributes in features.
- **Explainability**: Model coefficients are exposed via `explain_decision` tool so users can understand predictions.
- **Human oversight**: System is designed to assist analysts, not make autonomous decisions.

## Monitoring

- PSI-based drift detection compares production data against training distribution
- Thresholds: PSI > 0.1 = warning, PSI > 0.2 = retrain trigger
- Operational metrics tracked: latency, prediction count, risk distribution
