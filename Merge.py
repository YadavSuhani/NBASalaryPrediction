# Merge.py
import pandas as pd, numpy as np, re
from pathlib import Path

# --- paths (adjust to your paths if needed) ---
draft_path = "players_draft_contracts.csv"
two_path = "nba_two_seasons.csv"
fin_path = "team_finance.csv"

# --- read ---
draft = pd.read_csv(draft_path)
two   = pd.read_csv(two_path)
fin   = pd.read_csv(fin_path)

# --- helpers ---
def normalize_season(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    m = re.match(r"^\s*(\d{4})\s*-\s*(\d{2,4})\s*$", s) or re.match(r"^\s*(\d{4})\s*[/–]\s*(\d{2,4})\s*$", s)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if y2 < 100:
            y2 = (y1//100)*100 + y2
            if y2 < y1: y2 = y1 + 1
        return f"{y1:04d}-{y2:04d}"
    m2 = re.match(r"^\s*(\d{4})\s*$", s)
    if m2:
        y1 = int(m2.group(1)); return f"{y1:04d}-{(y1+1):04d}"
    return s

def to_num_money(x):
    if pd.isna(x): return np.nan
    s = str(x).replace("$","").replace(",","").replace("(","-").replace(")","").strip()
    try: return float(s)
    except: return np.nan

def norm_team(s):
    return None if pd.isna(s) else str(s).upper().strip()

# tidy names (just in case)
draft.columns = [c.strip() for c in draft.columns]
two.columns   = [c.strip() for c in two.columns]
fin.columns   = [c.strip() for c in fin.columns]

# ============================================================
# Finance: wide -> long
# ============================================================
alloc_pat = re.compile(r"^(\d{4}-\d{2,4})\s+Total Cap Allocation$")
space_pat = re.compile(r"^(\d{4}-\d{2,4})\s+Cap Space$")

alloc_cols = {}
space_cols = {}
for c in fin.columns:
    ma = alloc_pat.match(c)
    ms = space_pat.match(c)
    if ma: alloc_cols[ma.group(1)] = c
    if ms: space_cols[ms.group(1)] = c

records = []
for _, row in fin.iterrows():
    team = norm_team(row["Team"])
    for raw_season, alloc_col in alloc_cols.items():
        season_norm = normalize_season(raw_season)
        space_col = space_cols.get(raw_season)
        cap_alloc = to_num_money(row[alloc_col]) if alloc_col in fin.columns else np.nan
        cap_space = to_num_money(row[space_col]) if space_col in fin.columns else np.nan
        records.append({"Team": team, "season_norm": season_norm,
                        "cap_alloc": cap_alloc, "cap_space": cap_space})
finance_long = pd.DataFrame.from_records(records)

# ============================================================
# Draft contracts (drop 4th-year)
# ============================================================
draft_clean = draft.copy()
if "4th Year Option" in draft_clean.columns:
    draft_clean = draft_clean.drop(columns=["4th Year Option"])
draft_clean["_team_norm"]  = draft_clean["Team Drafted"].map(norm_team)
draft_clean["_player_key"] = draft_clean["Name"].astype(str).str.upper().str.strip()

# ============================================================
# Two-seasons: seasons + ALL stats (suffix _S1/_S2)
# ============================================================
two_clean = two.copy()
two_clean["_player_key"]  = two_clean["NAME"].astype(str).str.upper().str.strip()
two_clean["season1_norm"] = two_clean["SEASON1_ID"].map(normalize_season)
two_clean["season2_norm"] = two_clean["SEASON2_ID"].map(normalize_season)

# Collect all stat columns with _S1 or _S2 suffix
stat_cols = [c for c in two_clean.columns if re.search(r"_S[12]$", c)]
# (Optional) ensure numeric for stats that are numeric strings.
for c in stat_cols:
    if two_clean[c].dtype == object:
        two_clean[c] = pd.to_numeric(two_clean[c], errors="ignore")

# Keep minimal ID + all stats
two_keep = two_clean[["_player_key", "season1_norm", "season2_norm"] + stat_cols].drop_duplicates(subset=["_player_key"])

# ============================================================
# Merge draft + two_seasons (adds seasons + stats)
# ============================================================
merged = pd.merge(draft_clean, two_keep, on="_player_key", how="left")

# ============================================================
# Attach finance for both seasons using DRAFT team
# ============================================================
s1 = finance_long.rename(columns={"season_norm":"season1_norm","cap_space":"cap_space_s1","cap_alloc":"cap_alloc_s1"})
s2 = finance_long.rename(columns={"season_norm":"season2_norm","cap_space":"cap_space_s2","cap_alloc":"cap_alloc_s2"})

merged = pd.merge(
    merged, s1,
    left_on=["_team_norm","season1_norm"],
    right_on=["Team","season1_norm"],
    how="left"
).drop(columns=["Team"])

merged = pd.merge(
    merged, s2,
    left_on=["_team_norm","season2_norm"],
    right_on=["Team","season2_norm"],
    how="left"
).drop(columns=["Team"])

# ============================================================
# Final column order & drop helper columns
# ============================================================
financial_cols = ["cap_space_s1","cap_alloc_s1","cap_space_s2","cap_alloc_s2"]
draft_cols = [c for c in draft_clean.columns if c not in ["_team_norm","_player_key"]]

# Put draft first, then stats, then finances, then anything leftover; then drop helpers
ordered = draft_cols + stat_cols + financial_cols + [c for c in merged.columns if c not in draft_cols + stat_cols + financial_cols + ["_team_norm","_player_key","season1_norm","season2_norm"]]
merged = merged[ordered]
merged = merged.drop(columns=["_team_norm","_player_key","season1_norm","season2_norm", "PLUS_MINUS_S1", "PLUS_MINUS_S2"], errors="ignore")

# ============================================================
# Save
# ============================================================
out_path = "merged_player_finance_with_stats.csv"
merged.to_csv(out_path, index=False)
print(f"Saved → {out_path}")
print(merged.head(10))