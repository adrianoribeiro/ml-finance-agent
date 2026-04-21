import torch
from src.models.baseline import get_logistic_regression, get_random_forest, CreditMLP


def test_logistic_regression_fits(clean_df):
    model = get_logistic_regression()
    X = clean_df.drop("SeriousDlqin2yrs", axis=1)
    y = clean_df["SeriousDlqin2yrs"]
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)


def test_random_forest_fits(clean_df):
    model = get_random_forest()
    X = clean_df.drop("SeriousDlqin2yrs", axis=1)
    y = clean_df["SeriousDlqin2yrs"]
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)


def test_predictions_are_probabilities(clean_df):
    model = get_logistic_regression()
    X = clean_df.drop("SeriousDlqin2yrs", axis=1)
    y = clean_df["SeriousDlqin2yrs"]
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    # Probabilities should be between 0 and 1
    assert (probs >= 0).all()
    assert (probs <= 1).all()


def test_mlp_forward_pass():
    model = CreditMLP(n_features=11)
    X = torch.randn(5, 11)
    output = model(X)
    assert output.shape == (5,)


def test_mlp_output_range():
    model = CreditMLP(n_features=11)
    model.eval()
    X = torch.randn(10, 11)
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits)
    assert (probs >= 0).all()
    assert (probs <= 1).all()
