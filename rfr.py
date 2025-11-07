#!/usr/bin/env python3
# rfr_nba_salary.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

#methods
def money_to_float(series: pd.Series) -> pd.Series:
    """Convert strings like '$174,418,704.00' or '$(25,478,740.00)' into floats."""
    if series is None:
        return series
    s = series.astype(str)
    s = s.str.replace(r'[\$,]', '', regex=True)
    s = s.str.replace(r'^\((.*)\)$', r'-\1', regex=True)
    s = s.str.strip()
    return pd.to_numeric(s, errors="coerce")

def safe_rename(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Rename columns if they exist."""
    to_rename = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=to_rename)

def grab_first_matching(colnames: List[str], candidates: List[str]) -> str:
    """Return first name from candidates that exists in colnames, else ''."""
    cname = next((c for c in candidates if c in colnames), "")
    return cname

def combine_stat_pairs(df: pd.DataFrame, base_names: List[str]) -> pd.DataFrame:
    """
    For each base name B, if columns B_S1 and/or B_S2 exist, create:
      - B_SUM = sum of available seasons
      - B_MEAN = mean across available seasons
    """
    out = df.copy()
    for base in base_names:
        s1 = f"{base}_S1"
        s2 = f"{base}_S2"
        vals = []
        if s1 in out.columns: vals.append(out[s1])
        if s2 in out.columns: vals.append(out[s2])
        if len(vals) == 0:
            continue
        mat = pd.concat(vals, axis=1)
        out[f"{base}_SUM"] = mat.sum(axis=1, skipna=True)
        out[f"{base}_MEAN"] = mat.mean(axis=1, skipna=True)
    return out


stats_path = "nba_two_seasons.csv"
contracts_path = "players_draft_contracts.csv"
finance_path = "team_finance.csv"

if not all(os.path.exists(p) for p in [stats_path, contracts_path, finance_path]):
    raise FileNotFoundError(
        "Make sure nba_two_seasons.csv, players_draft_contracts.csv, and team_finance.csv "
        "are in the same directory as this script."
    )

stats = pd.read_csv(stats_path)
contracts = pd.read_csv(contracts_path)
finance = pd.read_csv(finance_path)

#normalize column names
stats = safe_rename(stats, {
    "PLAYER": "NAME",
    "Player": "NAME",
    "player": "NAME",
    "TEAM": "TEAM_ABBR",
    "Team": "TEAM_ABBR",
    "team": "TEAM_ABBR",
})

contracts = safe_rename(contracts, {
    "PLAYER": "Name",
    "Player": "Name",
    "player": "Name",
    "TEAM": "Team Drafted",
    "Team": "Team Drafted",
    "team": "Team Drafted",
})

finance = safe_rename(finance, {
    "TEAM": "Team",
    "team": "Team",
})

stats_name_col = grab_first_matching(stats.columns.tolist(), ["NAME", "Name", "Player"])
contracts_name_col = grab_first_matching(contracts.columns.tolist(), ["Name", "NAME", "Player"])

if not stats_name_col or not contracts_name_col:
    raise ValueError(
        f"Could not find a player name column in stats/contacts.\n"
        f"Stats columns: {list(stats.columns)[:10]}\n"
        f"Contracts columns: {list(contracts.columns)[:10]}"
    )


first_year_col = grab_first_matching(contracts.columns.tolist(), ["1st Year", "First Year", "Year 1", "Year1"])
if not first_year_col:
    raise ValueError(
        "Could not find a '1st Year' salary column in players_draft_contracts.csv. "
        "Expected one of: ['1st Year','First Year','Year 1','Year1']"
    )


contracts["Salary"] = money_to_float(contracts[first_year_col])

contracts = safe_rename(contracts, {"Team Drafted": "TEAM_ABBR"})
if "TEAM_ABBR" not in contracts.columns:
    contracts["TEAM_ABBR"] = np.nan

contracts_keep = [c for c in ["Name", "TEAM_ABBR", "Salary"] if c in contracts.columns]
contracts_model = contracts[contracts_keep].dropna(subset=["Salary"]).copy()

finance.columns = [c.strip() for c in finance.columns]

cap_cols_candidates = [
    "2011-12 Total Cap Allocation", "2011-12 Cap Space",
]
cap_cols = [c for c in finance.columns if any(c.startswith(x) for x in cap_cols_candidates)]

if not cap_cols:
    cap_cols = [c for c in finance.columns if "2011-12" in c]


for c in cap_cols:
    finance[c] = money_to_float(finance[c])

finance_keep = ["Team"] + cap_cols
finance_model = finance.loc[:, [c for c in finance_keep if c in finance.columns]].copy()
finance_model = finance_model.rename(columns={"Team": "TEAM_ABBR"})


base_stats = ["MIN", "PTS", "AST", "REB", "OREB", "DREB", "STL", "BLK", "TOV", "PF", "FGM", "FGA", "FTM", "FTA", "PLUS_MINUS"]
stats_aug = combine_stat_pairs(stats, base_stats)


id_like_cols = set([
    stats_name_col, "PLAYER_ID", "DOB", "POSITION", "NATIONALITY", "HEIGHT", "WEIGHT", "SEASON1_ID", "SEASON2_ID", "TEAM_ABBR"]).intersection(stats_aug.columns)


merged = stats_aug.merge(
    contracts_model,
    left_on=stats_name_col,
    right_on="Name",
    how="inner",
    validate="m:1"
)

if "TEAM_ABBR_x" in merged.columns or "TEAM_ABBR_y" in merged.columns:
    merged["TEAM_ABBR_ALL"] = merged.get("TEAM_ABBR_y", merged.get("TEAM_ABBR_x"))
    if "TEAM_ABBR_ALL" in merged.columns and "TEAM_ABBR_x" in merged.columns:
        merged["TEAM_ABBR_ALL"] = merged["TEAM_ABBR_ALL"].fillna(merged["TEAM_ABBR_x"])
else:
    if "TEAM_ABBR" in merged.columns:
        merged["TEAM_ABBR_ALL"] = merged["TEAM_ABBR"]
    else:
        merged["TEAM_ABBR_ALL"] = np.nan

merged = merged.merge(
    finance_model,
    left_on="TEAM_ABBR_ALL",
    right_on="TEAM_ABBR",
    how="left"
)

target = "Salary"

drop_from_features = set()
drop_from_features.update(id_like_cols)
drop_from_features.update(["Name", "TEAM_ABBR", "TEAM_ABBR_ALL"])
drop_from_features.update([stats_name_col])

numeric_merged = merged.select_dtypes(include=[np.number]).copy()

if target not in numeric_merged.columns:
    raise ValueError("Target 'Salary' not found as numeric after cleaning.")

X = numeric_merged.drop(columns=[target], errors="ignore")
y = numeric_merged[target].copy()

std = X.std(numeric_only=True)
X = X.loc[:, std > 0]

if X.shape[1] == 0:
    raise ValueError("No usable numeric features found after cleaning. Check your CSVs.")

#train/split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

#random forest model
rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Model Performance ===")
print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")
print(f"Features used: {X.shape[1]}")
print(f"MAE: ${mae:,.2f}")
print(f"R² : {r2:.4f}")


#csv
pred_out = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": y_pred
}, index=y_test.index)

pred_out_path = "rf_predictions.csv"
pred_out.to_csv(pred_out_path, index=False)
print(f"\nSaved predictions to: {pred_out_path}")

#plots
imp = pd.Series(rf.feature_importances_, index=X.columns)
imp = imp.sort_values(ascending=False)

top_k = 20
plt.figure(figsize=(11, 5))
imp.head(top_k).plot(kind="bar")
plt.title(f"Top {top_k} Feature Importances (Random Forest)")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("rf_feature_importances.png", dpi=160)
print("Saved feature importance plot -> rf_feature_importances.png")


plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, s=18, alpha=0.7)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, linestyle="--")
plt.xlabel("True Salary")
plt.ylabel("Predicted Salary")
plt.title("Parity Plot: True vs Predicted Salary")
plt.tight_layout()
plt.savefig("rf_parity_plot.png", dpi=160)
print("Saved parity plot -> rf_parity_plot.png")


residuals = y_pred - y_test.values
plt.figure(figsize=(8,4))
plt.hist(residuals, bins=30)
plt.title("Residuals Histogram (Predicted - True)")
plt.xlabel("Residual")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("rf_residuals_hist.png", dpi=160)
print("Saved residuals histogram -> rf_residuals_hist.png")

print("\nDone.")
