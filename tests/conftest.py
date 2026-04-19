import pandas as pd
import pytest


@pytest.fixture
def raw_df():
    """Small fake dataset that mimics the real CSV structure."""
    return pd.DataFrame({
        "SeriousDlqin2yrs": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        "RevolvingUtilizationOfUnsecuredLines": [0.3, 0.9, 2.5, 0.1, 0.5, 0.7, 0.2, 0.8, 0.4, 0.6],
        "age": [35, 0, 52, 105, 28, 45, 60, 22, 40, 55],
        "NumberOfTime30-59DaysPastDueNotWorse": [0, 2, 0, 1, 3, 0, 0, 1, 0, 0],
        "DebtRatio": [0.3, 0.8, 0.1, 50000.0, 0.5, 0.4, 0.2, 0.6, 0.3, 0.1],
        "MonthlyIncome": [5000.0, None, 8000.0, 3000.0, None, 7000.0, 6000.0, 4000.0, None, 9000.0],
        "NumberOfOpenCreditLinesAndLoans": [5, 3, 8, 2, 6, 4, 7, 3, 5, 9],
        "NumberOfTimes90DaysLate": [0, 1, 0, 0, 2, 0, 0, 0, 1, 0],
        "NumberRealEstateLoansOrLines": [1, 0, 2, 1, 0, 1, 3, 0, 1, 2],
        "NumberOfTime60-89DaysPastDueNotWorse": [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        "NumberOfDependents": [2.0, None, 0.0, 1.0, 3.0, 0.0, 2.0, None, 1.0, 0.0],
    })


@pytest.fixture
def clean_df():
    """Small clean dataset ready for training."""
    return pd.DataFrame({
        "SeriousDlqin2yrs": [0, 1, 0, 0, 1, 0, 0, 0],
        "RevolvingUtilizationOfUnsecuredLines": [0.3, 0.9, 0.5, 0.1, 0.5, 0.7, 0.2, 0.8],
        "age": [35, 42, 52, 28, 28, 45, 60, 22],
        "DebtRatio": [0.3, 0.8, 0.1, 0.5, 0.5, 0.4, 0.2, 0.6],
        "MonthlyIncome": [5000, 6000, 8000, 3000, 4000, 7000, 6000, 4000],
        "NumberOfOpenCreditLinesAndLoans": [5, 3, 8, 2, 6, 4, 7, 3],
        "NumberRealEstateLoansOrLines": [1, 0, 2, 1, 0, 1, 3, 0],
        "NumberOfDependents": [2, 0, 0, 1, 3, 0, 2, 1],
        "income_missing": [0, 0, 0, 0, 1, 0, 0, 0],
        "total_late_payments": [0, 4, 0, 1, 6, 0, 0, 1],
        "has_late_payment": [0, 1, 0, 1, 1, 0, 0, 1],
        "has_severe_late": [0, 1, 0, 0, 1, 0, 0, 0],
    })
