import logging

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.baseline import CreditMLP

logger = logging.getLogger(__name__)


def split_data(
    df: pd.DataFrame, target_col: str = "SeriousDlqin2yrs", test_size: float = 0.2
):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


def train_sklearn(model, X_train, X_test, y_train, y_test, run_name: str, scale=False):
    """Train a sklearn model and log to MLflow."""
    scaler = None
    X_tr, X_te = X_train, X_test

    if scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

    with mlflow.start_run(run_name=run_name):
        model.fit(X_tr, y_train)

        y_prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_metric("auc", auc)
        if scale:
            mlflow.log_param("scaler", "StandardScaler")
        mlflow.sklearn.log_model(model, "model")

        logger.info(f"{run_name} AUC: {auc:.4f}")
        return auc


def train_mlp(
    X_train,
    X_test,
    y_train,
    y_test,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
):
    """Train MLP PyTorch model and log to MLflow."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    X_train_t = torch.FloatTensor(X_tr)
    y_train_t = torch.FloatTensor(y_train.values)
    X_test_t = torch.FloatTensor(X_te)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = CreditMLP(X_train_t.shape[1])

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos])

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_prob = torch.sigmoid(model(X_test_t)).numpy()

    auc = roc_auc_score(y_test, y_prob)

    with mlflow.start_run(run_name="mlp-pytorch"):
        mlflow.log_param("model_type", "MLP")
        mlflow.log_param("hidden_layers", "64-32")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_metric("auc", auc)

        torch.save(model.state_dict(), "/tmp/mlp_credit.pt")
        mlflow.log_artifact("/tmp/mlp_credit.pt")

        logger.info(f"mlp-pytorch AUC: {auc:.4f}")
        return auc
