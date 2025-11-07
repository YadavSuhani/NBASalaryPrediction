# XGboost_regression_with_plots.py
import pandas as pd, numpy as np, re, os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# =======================
# CONFIG
# =======================
CSV_PATH = "merged_player_finance_with_stats.csv"
TARGET_COL = "3rd Year Option"
FEATURE_COLS = [
    "Draft Pick","3rd Year Option",
    "MIN_S1","MIN_S2","PTS_S1","PTS_S2","AST_S1","AST_S2",
    "REB_S1","REB_S2","STL_S1","STL_S2","FG3M_S1","FG3M_S2",
    "FG3A_S1","FG3A_S2","FTM_S1","FTM_S2","FTA_S1","FTA_S2",
    "FGM_S1","FGM_S2","FGA_S1","FGA_S2",
    "cap_space_s1","cap_alloc_s1","cap_space_s2","cap_alloc_s2"
]
TOP_K_IMPORTANCE = 20
RAND = 42
TEST_SIZE = 0.25

# =======================
# LOAD & FILTER (your rule: drop any NA row)
# =======================
df = pd.read_csv(CSV_PATH)
df = df[FEATURE_COLS].dropna()
print(f"[INFO] After dropping NA rows: {df.shape[0]} players remain")

# -----------------------
# Target → numeric
# -----------------------
def to_float_money(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace("$","").replace(",","")
    s = s.replace("(","-").replace(")","")
    try: return float(s)
    except: return np.nan

y = df[TARGET_COL].apply(to_float_money) if df[TARGET_COL].dtype==object else df[TARGET_COL].astype(float)

valid = y.notna()
df = df.loc[valid]
y  = y.loc[valid]

# -----------------------
# Features → numeric only
# -----------------------
X = df.drop(columns=[TARGET_COL]).copy()

if X["Draft Pick"].dtype==object:
    X["Draft Pick"] = X["Draft Pick"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)

for col in X.columns:
    if X[col].dtype==object:
        s = (X[col].astype(str)
                .str.replace(r"[\$,]", "", regex=True)
                .str.replace(r"\((.*)\)", r"-\1", regex=True)
                .str.strip())
        X[col] = pd.to_numeric(s, errors="coerce")

mask = X.notna().all(axis=1) & y.notna()
X, y = X.loc[mask], y.loc[mask]
print(f"[INFO] After numeric coercions: X={X.shape}, y={y.shape}")

# =======================
# SPLIT
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RAND
)

# =======================
# MODEL
# =======================
model = XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=RAND,
    n_jobs=-1
)
model.fit(X_train, y_train)

# =======================
# METRICS
# =======================
pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred, squared=False)
r2   = r2_score(y_test, pred)

print(f"\n✅ MAE : {mae:,.2f}")
print(f"✅ RMSE: {rmse:,.2f}")
print(f"✅ R²  : {r2:,.4f}")

# =======================
# PLOTS
# =======================
os.makedirs("figs", exist_ok=True)

# 1) Parity plot (True vs Pred)
plt.figure(figsize=(7,7))
plt.scatter(y_test, pred, alpha=0.7)
mn, mx = float(min(y_test.min(), pred.min())), float(max(y_test.max(), pred.max()))
plt.plot([mn, mx], [mn, mx], linestyle="--")
plt.title("Parity Plot: True vs Predicted 3rd Year Option")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.tight_layout()
plt.savefig("figs/parity_plot.png", dpi=160)
plt.show()

# 2) Residuals histogram (Pred - True)
resid = pred - y_test
plt.figure(figsize=(10,5))
plt.hist(resid, bins=20)
plt.title("Residuals Histogram (Predicted - True)")
plt.xlabel("Residual")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("figs/residuals_hist.png", dpi=160)
plt.show()

# 3) Feature importance (top-k)
importances = model.feature_importances_
fi = (pd.DataFrame({"feature": X.columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(TOP_K_IMPORTANCE))

plt.figure(figsize=(max(8, 0.5*len(fi)), 5))
plt.bar(range(len(fi)), fi["importance"].values)
plt.xticks(range(len(fi)), fi["feature"].tolist(), rotation=60, ha="right")
plt.ylabel("Importance")
plt.title(f"Top {len(fi)} Feature Importances")
plt.tight_layout()
plt.savefig("figs/feature_importance.png", dpi=160)
plt.show()

# Optional: print the table too
print("\n🔎 Top features:")
print(fi.reset_index(drop=True))
