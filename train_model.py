"""
Train the Adoption-Intent Linear Regression model and serialize every
artifact the app needs (model, means/stds for explainability, R²,
correlation matrix, descriptive stats) into a single pickle file.

Run this ONCE (or again whenever the dataset changes) before starting
the app:

    python train_model.py

This keeps training and serving cleanly separated — app.py never
re-fits the model, it only loads what this script produces.
"""

import joblib
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression

VAR_ORDER = ["TR", "PU", "PEOU", "PR", "FL", "TA"]
DATA_PATH = "MBA_AI_Investment_Dataset.csv"
OUTPUT_PATH = "model_artifacts.pkl"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[VAR_ORDER]
    y = df["AD"]

    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)
    n = len(df)
    p = X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    means = X.mean()
    stds = X.std()
    y_std = y.std()

    # Standardized coefficients: comparable "importance" across variables
    std_coefs = pd.Series(model.coef_, index=VAR_ORDER) * stds / y_std
    corr = X.corr()
    desc = df[VAR_ORDER + ["AD"]].describe().T.round(2)

    artifacts = {
        "model": model,
        "means": means,
        "stds": stds,
        "r2": r2,
        "adj_r2": adj_r2,
        "n": n,
        "std_coefs": std_coefs,
        "corr": corr,
        "y": y,
        "desc": desc,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    joblib.dump(artifacts, OUTPUT_PATH)
    print(f"Saved trained model + diagnostics to '{OUTPUT_PATH}'")
    print(f"R\u00b2 = {r2:.3f}  |  Adjusted R\u00b2 = {adj_r2:.3f}  |  N = {n}")


if __name__ == "__main__":
    main()
