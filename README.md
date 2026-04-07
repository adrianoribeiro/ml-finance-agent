# Credit Risk AI Agent

AI-powered agent for credit risk analysis. Predicts whether a client will default on their financial obligations within the next 2 years, and explains the reasoning in natural language.

## How the Dataset Was Built

```
2024                                            2026
  │                                               │
  │  Collected data from 150k clients:            │  Checked:
  │  - income                                     │  - Client 1: paid everything ✓ → 0
  │  - age                                        │  - Client 2: defaulted ✗ → 1
  │  - debts                                      │  - Client 3: paid everything ✓ → 0
  │  - payment delays                             │  - Client 4: defaulted ✗ → 1
  │  - etc                                        │
```

Now a new client arrives. We have their data today but don't know the future. The model uses patterns learned from 150k historical clients to predict: will this new client default or not?

This is supervised learning with a binary target:
- **0** = client will pay (no default)
- **1** = client will default

## Dataset Features

| Feature | Description |
|---------|-------------|
| **SeriousDlqin2yrs** | TARGET — client defaulted within 2 years (0=no, 1=yes) |
| RevolvingUtilizationOfUnsecuredLines | Credit card + personal credit usage ratio (balance / limit) |
| age | Client age |
| NumberOfTime30-59DaysPastDueNotWorse | Number of times 30-59 days late |
| DebtRatio | Monthly debt payments / monthly income |
| MonthlyIncome | Monthly income |
| NumberOfOpenCreditLinesAndLoans | Number of open credit lines and loans |
| NumberOfTimes90DaysLate | Number of times 90+ days late |
| NumberRealEstateLoansOrLines | Number of real estate loans |
| NumberOfTime60-89DaysPastDueNotWorse | Number of times 60-89 days late |
| NumberOfDependents | Number of dependents |

Source: [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) — Kaggle competition dataset with 150,000 samples.

## Como rodar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
jupyter notebook notebooks/
```

Os notebooks devem ser executados em ordem (01, 02, 03).

Após rodar o notebook 03, visualize os experimentos com:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
