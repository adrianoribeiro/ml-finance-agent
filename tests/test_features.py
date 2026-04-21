from src.features.feature_engineering import (
    remove_invalid_ages,
    fill_missing,
    cap_outliers,
    create_late_payment_features,
)


def test_remove_invalid_ages(raw_df):
    result = remove_invalid_ages(raw_df)
    # age=0 and age=105 should be removed
    assert (result["age"] >= 18).all()
    assert (result["age"] <= 100).all()
    assert len(result) == 8


def test_fill_missing_no_nulls(raw_df):
    result = fill_missing(raw_df.copy())
    assert result["MonthlyIncome"].isnull().sum() == 0
    assert result["NumberOfDependents"].isnull().sum() == 0


def test_fill_missing_creates_flag(raw_df):
    result = fill_missing(raw_df.copy())
    assert "income_missing" in result.columns
    # 3 rows had null MonthlyIncome
    assert result["income_missing"].sum() == 3


def test_cap_outliers_revolving(raw_df):
    result = cap_outliers(raw_df.copy())
    assert result["RevolvingUtilizationOfUnsecuredLines"].max() <= 1.5


def test_cap_outliers_debt_ratio(raw_df):
    result = cap_outliers(raw_df.copy())
    cap = raw_df["DebtRatio"].quantile(0.99)
    assert result["DebtRatio"].max() <= cap


def test_create_late_payment_features(raw_df):
    result = create_late_payment_features(raw_df.copy())

    # Original columns should be gone
    assert "NumberOfTime30-59DaysPastDueNotWorse" not in result.columns
    assert "NumberOfTime60-89DaysPastDueNotWorse" not in result.columns
    assert "NumberOfTimes90DaysLate" not in result.columns

    # New columns should exist
    assert "total_late_payments" in result.columns
    assert "has_late_payment" in result.columns
    assert "has_severe_late" in result.columns


def test_late_payment_values(raw_df):
    result = create_late_payment_features(raw_df.copy())
    # Row 0: 0+0+0 = 0 late payments
    assert result.iloc[0]["total_late_payments"] == 0
    assert result.iloc[0]["has_late_payment"] == 0
    # Row 1: 2+1+1 = 4 late payments, has severe (90+ days)
    assert result.iloc[1]["total_late_payments"] == 4
    assert result.iloc[1]["has_late_payment"] == 1
    assert result.iloc[1]["has_severe_late"] == 1


def test_no_nulls_after_pipeline(raw_df):
    df = remove_invalid_ages(raw_df)
    df = fill_missing(df)
    df = cap_outliers(df)
    df = create_late_payment_features(df)
    assert df.isnull().sum().sum() == 0


def test_row_count_preserved_after_fill(raw_df):
    before = len(raw_df)
    result = fill_missing(raw_df.copy())
    assert len(result) == before
