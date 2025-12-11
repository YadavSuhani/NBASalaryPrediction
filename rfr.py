#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def main():

    df = pd.read_csv("final_data.csv")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    y = df["year3_salary"]

    drop_cols = ["year1_salary", "year2_salary", "year3_salary", "year1_cap_alloc", "year2_cap_alloc", "year3_cap_alloc", "year3_cap_space", "year3_option", "year3_team", "name", "college", "draft_team", "pos",]

    X = df.drop(columns=drop_cols, errors="ignore")

    X = X.select_dtypes(include=[np.number])

    X = X.fillna(X.median())
    y = y.fillna(y.median())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Samples: {len(df)}")
    print(f"Features used: {X.shape[1]}")
    print(f"MAE: {mae:,.2f}")
    print(f"R^2: {r2:.4f}")

    fi = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(fi["feature"].head(20)[::-1], fi["importance"].head(20)[::-1])
    plt.xlabel("Importance")
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig("rf_feature_importance.png", dpi=160)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    min_v = min(y_test.min(), y_pred.min())
    max_v = max(y_test.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v])
    plt.xlabel("True Year 3 Salary")
    plt.ylabel("Predicted Year 3 Salary")
    plt.title("Parity Plot")
    plt.tight_layout()
    plt.savefig("rf_parity_plot.png", dpi=160)

    residuals = y_pred - y_test.values
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residuals Histogram")
    plt.tight_layout()
    plt.savefig("rf_residuals_hist.png", dpi=160)



if __name__ == "__main__":
    main()
