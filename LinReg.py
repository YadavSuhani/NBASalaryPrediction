import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
draft_df = pd.read_csv("players_draft_contracts.csv")
finance_df = pd.read_csv("team_finance.csv")
draft_df["3rd Year Option"] = (
    draft_df["3rd Year Option"]
    .replace("[\$,]", "", regex=True)
    .astype(float)
)
draft_df = draft_df.rename(columns={"Name": "NAME", "Team Drafted": "TEAM"})
nba_df["PTS_TOTAL"] = nba_df[["PTS_S1", "PTS_S2"]].sum(axis=1)
nba_df["AST_TOTAL"] = nba_df[["AST_S1", "AST_S2"]].sum(axis=1)
nba_df["REB_TOTAL"] = nba_df[["REB_S1", "REB_S2"]].sum(axis=1)
nba_df["MIN_TOTAL"] = nba_df[["MIN_S1", "MIN_S2"]].sum(axis=1)
finance_df.columns = finance_df.columns.str.strip()
finance_df = finance_df.rename(columns={"Team": "TEAM"})
merged = pd.merge(
    nba_df,
    draft_df[["NAME", "TEAM", "Draft Pick", "3rd Year Option"]],
    on="NAME",
    how="inner"
)

merged = pd.merge(
    merged,
    finance_df[["TEAM", "2011-12 Cap Space", "2012-13 Cap Space"]],
    on="TEAM",
    how="left"
)
for col in ["2011-12 Cap Space", "2012-13 Cap Space"]:
    merged[col] = (
        merged[col]
        .astype(str)
        .str.replace(r"[\$,()]", "", regex=True)
        .replace("", "0")
        .astype(float)
    )
X = merged[[
    "PTS_TOTAL", "AST_TOTAL", "REB_TOTAL", "MIN_TOTAL",
    "Draft Pick", "2011-12 Cap Space", "2012-13 Cap Space"
]]
y = merged["3rd Year Option"].fillna(merged["3rd Year Option"].median())
X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = np.mean(np.abs(y_test - y_pred))

metrics_df = pd.DataFrame({
    "Metric": ["R² Score", "Root Mean Squared Error (RMSE)", "Mean Absolute Error (MAE)"],
    "Value": [round(r2, 3), f"${rmse:,.0f}", f"${mae:,.0f}"]
})
print(metrics_df.to_string(index=False))
residuals = y_test - y_pred

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_pred, y=residuals, color="steelblue", alpha=0.7)
plt.axhline(0, color="red", linestyle="--", lw=2, label="Zero Residual Line")
plt.xlabel("Predicted 3rd-Year Salary ($)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot – Linear Regression Model")
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'${x*1e-6:.1f}M'))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'${x*1e-6:.1f}M'))

plt.legend()
plt.tight_layout()
plt.show()
