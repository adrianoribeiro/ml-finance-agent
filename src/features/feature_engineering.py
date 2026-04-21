import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def remove_invalid_ages(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["age"] >= 18) & (df["age"] <= 100)].copy()


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    # Flag before filling — missing income may be predictive
    df["income_missing"] = df["MonthlyIncome"].isnull().astype(int)

    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
        df["NumberOfDependents"].median()
    )
    return df


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df["RevolvingUtilizationOfUnsecuredLines"] = df[
        "RevolvingUtilizationOfUnsecuredLines"
    ].clip(upper=1.5)

    debt_cap = df["DebtRatio"].quantile(0.99)
    df["DebtRatio"] = df["DebtRatio"].clip(upper=debt_cap)
    return df


def create_late_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    df["total_late_payments"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )
    df["has_late_payment"] = (df["total_late_payments"] > 0).astype(int)
    df["has_severe_late"] = (df["NumberOfTimes90DaysLate"] > 0).astype(int)

    df = df.drop(
        columns=[
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
        ]
    )
    return df


def compute_features(raw_path: str) -> pd.DataFrame:
    """Full pipeline: raw CSV -> clean DataFrame ready for training."""
    df = load_raw_data(raw_path)
    df = remove_invalid_ages(df)
    df = fill_missing(df)
    df = cap_outliers(df)
    df = create_late_payment_features(df)
    return df
