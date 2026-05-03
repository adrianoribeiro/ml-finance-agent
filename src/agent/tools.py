import json
import logging

import joblib
import numpy as np
import pandas as pd
from langchain.tools import tool

from src.agent.rag_pipeline import RAGRetriever
from src.config import Config

logger = logging.getLogger(__name__)

_retriever = RAGRetriever()

# Load model, scaler and feature names once
_model = joblib.load(Config.MODEL_PATH)
_scaler = joblib.load(Config.SCALER_PATH)
_feature_names = joblib.load(Config.FEATURE_NAMES_PATH)
_df = pd.read_csv(Config.DATA_PATH)


@tool
def predict_risk(features_json: str) -> str:
    """Predict default probability for a client. Input is a JSON string with client features.
    Required keys: RevolvingUtilizationOfUnsecuredLines, age, DebtRatio, MonthlyIncome,
    NumberOfOpenCreditLinesAndLoans, NumberRealEstateLoansOrLines, NumberOfDependents,
    income_missing, total_late_payments, has_late_payment, has_severe_late.
    Example: {"age": 35, "MonthlyIncome": 5000, "DebtRatio": 0.3, ...}
    """
    try:
        data = json.loads(features_json)
    except json.JSONDecodeError:
        return "Error: invalid JSON."

    # Fill missing fields with dataset median
    for feat in _feature_names:
        if feat not in data:
            data[feat] = float(_df[feat].median())

    row = pd.DataFrame([data])[_feature_names]
    row_scaled = _scaler.transform(row)
    prob = _model.predict_proba(row_scaled)[0][1]

    risk = "LOW" if prob < 0.3 else "MEDIUM" if prob < 0.6 else "HIGH"
    return f"Default probability: {prob:.1%}. Risk level: {risk}."


@tool
def query_data(question: str) -> str:
    """Query statistics from the credit risk dataset. Use this for questions like
    'average income of defaulters', 'how many clients defaulted', 'age distribution', etc.
    Input is a natural language question about the data.
    """
    df = _df.copy()
    defaulters = df[df["SeriousDlqin2yrs"] == 1]
    non_defaulters = df[df["SeriousDlqin2yrs"] == 0]

    stats = {
        "total_clients": len(df),
        "total_defaulters": len(defaulters),
        "default_rate": f"{defaulters.shape[0] / len(df):.2%}",
        "avg_income_defaulters": f"{defaulters['MonthlyIncome'].mean():.0f}",
        "avg_income_non_defaulters": f"{non_defaulters['MonthlyIncome'].mean():.0f}",
        "avg_age_defaulters": f"{defaulters['age'].mean():.1f}",
        "avg_age_non_defaulters": f"{non_defaulters['age'].mean():.1f}",
        "avg_debt_ratio_defaulters": f"{defaulters['DebtRatio'].mean():.2f}",
        "avg_debt_ratio_non_defaulters": f"{non_defaulters['DebtRatio'].mean():.2f}",
        "pct_late_payment_defaulters": f"{defaulters['has_late_payment'].mean():.1%}",
        "pct_late_payment_non_defaulters": f"{non_defaulters['has_late_payment'].mean():.1%}",
        "median_income": f"{df['MonthlyIncome'].median():.0f}",
        "median_age": f"{df['age'].median():.0f}",
    }

    return json.dumps(stats, indent=2)


@tool
def explain_decision(features_json: str) -> str:
    """Explain which features most contributed to a client's risk prediction.
    Input is the same JSON string with client features used in predict_risk.
    """
    try:
        data = json.loads(features_json)
    except json.JSONDecodeError:
        return "Error: invalid JSON."

    for feat in _feature_names:
        if feat not in data:
            data[feat] = float(_df[feat].median())

    row = pd.DataFrame([data])[_feature_names]
    row_scaled = _scaler.transform(row)

    # Use model coefficients to find top factors
    coefficients = _model.coef_[0]
    contributions = row_scaled[0] * coefficients

    # Sort by absolute contribution
    indices = np.argsort(np.abs(contributions))[::-1]

    explanations = []
    for i in indices[:5]:
        feat = _feature_names[i]
        value = data[feat]
        median = float(_df[feat].median())
        direction = "increases" if contributions[i] > 0 else "decreases"
        comparison = "above" if value > median else "below" if value < median else "at"
        explanations.append(
            f"- {feat}={value} ({comparison} median {median:.1f}): {direction} risk"
        )

    return "Top factors:\n" + "\n".join(explanations)


@tool
def search_docs(query: str) -> str:
    """Search the project documentation for information about features, data dictionary,
    credit risk concepts, or model details. Use this when the user asks about what a
    feature means, how the data was collected, or domain-specific questions.
    Input is a natural language query.
    """
    results = _retriever.search(query, k=3)
    if not results:
        return "No relevant documents found."
    return "\n\n---\n\n".join(results)
