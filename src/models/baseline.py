import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_logistic_regression():
    return LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    )


def get_random_forest():
    return RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )


class CreditMLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
