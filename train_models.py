# =============================
# train_models.py
# =============================
# Trains three models using team recent-form features (last 5 matches) + Bet365 odds (if present):
#   1) Full-time Result (H/D/A)
#   2) Both Teams To Score (BTTS Yes)
#   3) Over 2.5 Goals (Yes)
# Saves: result_model.pkl, btts_model.pkl, over25_model.pkl
# ---------------------------------

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_FILES = sorted([str(p) for p in Path(".").glob("E0*.csv")])
print(f"Found data files: {DATA_FILES}")
DATE_COL = "Date"  # dd/mm/yy per notes

# Columns available per notes (use a safe subset)
STAT_COLS = [
    "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC",
    "HY", "AY", "HR", "AR"
]

# Odds columns that may or may not exist
ODDS_1X2 = ["B365H", "B365D", "B365A"]
ODDS_OU = ["B365>2.5", "B365<2.5"]


def load_data(files):
    dfs = []
    for f in files:
        p = Path(f)
        if not p.exists():
            raise FileNotFoundError(f"Missing data file: {f}")
        df = pd.read_csv(p)
        dfs.append(df)
        print(f"{f}: {len(df)} matches")

    df = pd.concat(dfs, ignore_index=True)
    # Parse dates
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=[DATE_COL, "HomeTeam", "AwayTeam", "FTR"])  # core fields
    return df


def add_targets(df):
    # Result target
    df["target_result"] = df["FTR"].map({"A": 0, "D": 1, "H": 2})
    # BTTS target (Yes=1 if both scored at least 1)
    df["target_btts"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
    # Over 2.5 target (Yes=1 if total goals >= 3)
    df["target_over25"] = ((df["FTHG"] + df["FTAG"]) >= 3).astype(int)
    return df


def compute_team_form_features(df, window=5):
    """
    For each team and each match date, compute rolling means over the *previous* `window` matches.
    """
    home = df[[DATE_COL, "HomeTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR"]].copy()
    home.rename(columns={
        "HomeTeam": "Team",
        "FTHG": "GF",
        "FTAG": "GA",
        "HS": "S",
        "AS": "S_opp",
        "HST": "ST",
        "AST": "ST_opp",
        "HC": "C",
        "AC": "C_opp",
        "HY": "Y",
        "AY": "Y_opp",
        "HR": "R",
        "AR": "R_opp",
    }, inplace=True)

    away = df[[DATE_COL, "AwayTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR"]].copy()
    away.rename(columns={
        "AwayTeam": "Team",
        "FTHG": "GA",
        "FTAG": "GF",
        "HS": "S_opp",
        "AS": "S",
        "HST": "ST_opp",
        "AST": "ST",
        "HC": "C_opp",
        "AC": "C",
        "HY": "Y_opp",
        "AY": "Y",
        "HR": "R_opp",
        "AR": "R",
    }, inplace=True)

    long_df = pd.concat([home, away], ignore_index=True)
    long_df.sort_values(["Team", DATE_COL], inplace=True)

    roll_cols = ["GF", "GA", "S", "S_opp", "ST", "ST_opp", "C", "C_opp", "Y", "Y_opp", "R", "R_opp"]
    for col in roll_cols:
        long_df[f"{col}_ma{window}"] = (
            long_df.groupby("Team")[col]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

    feat_cols = [DATE_COL, "Team"] + [f"{c}_ma{window}" for c in roll_cols]
    feats = long_df[feat_cols].copy()

    df = df.merge(
        feats.add_prefix("home_"),
        left_on=["HomeTeam", DATE_COL],
        right_on=["home_Team", "home_Date"],
        how="left",
    )
    df = df.merge(
        feats.add_prefix("away_"),
        left_on=["AwayTeam", DATE_COL],
        right_on=["away_Team", "away_Date"],
        how="left",
    )

    df.drop(columns=[c for c in df.columns if c.endswith("_Team") or c.endswith("_Date")], inplace=True)
    return df


def add_odds_features(df):
    for col in ODDS_1X2:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if all(c in df.columns for c in ODDS_1X2):
        df["B365H_prob"] = 1.0 / df["B365H"]
        df["B365D_prob"] = 1.0 / df["B365D"]
        df["B365A_prob"] = 1.0 / df["B365A"]
        s = df[["B365H_prob", "B365D_prob", "B365A_prob"]].sum(axis=1)
        df[["B365H_prob", "B365D_prob", "B365A_prob"]] = df[["B365H_prob", "B365D_prob", "B365A_prob"]].div(s, axis=0)

    for col in ODDS_OU:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if all(c in df.columns for c in ODDS_OU):
        df["B365_over25_prob"] = 1.0 / df["B365>2.5"]
        df["B365_under25_prob"] = 1.0 / df["B365<2.5"]
        s2 = df[["B365_over25_prob", "B365_under25_prob"]].sum(axis=1)
        df[["B365_over25_prob", "B365_under25_prob"]] = df[["B365_over25_prob", "B365_under25_prob"]].div(s2, axis=0)
    return df


def build_feature_matrix(df, window=5):
    df = add_targets(df.copy())
    df = compute_team_form_features(df, window=window)
    df = add_odds_features(df)

    form_cols = [
        f"home_{c}_ma{window}" for c in ["GF","GA","S","S_opp","ST","ST_opp","C","C_opp","Y","Y_opp","R","R_opp"]
    ] + [
        f"away_{c}_ma{window}" for c in ["GF","GA","S","S_opp","ST","ST_opp","C","C_opp","Y","Y_opp","R","R_opp"]
    ]

    odds_cols_result = ["B365H_prob", "B365D_prob", "B365A_prob"] if "B365H_prob" in df.columns else []
    odds_cols_ou = ["B365_over25_prob", "B365_under25_prob"] if "B365_over25_prob" in df.columns else []

    X_result = df[form_cols + odds_cols_result].copy()
    X_btts = df[form_cols + odds_cols_result + odds_cols_ou].copy()
    X_over25 = df[form_cols + odds_cols_ou + odds_cols_result].copy()

    mask_valid = X_result.notna().all(axis=1) & df[["target_result","target_btts","target_over25"]].notna().all(axis=1)
    X_result = X_result[mask_valid]
    X_btts = X_btts[mask_valid]
    X_over25 = X_over25[mask_valid]
    y_result = df.loc[mask_valid, "target_result"]
    y_btts = df.loc[mask_valid, "target_btts"]
    y_over25 = df.loc[mask_valid, "target_over25"]

    return X_result, y_result, X_btts, y_btts, X_over25, y_over25, form_cols, odds_cols_result, odds_cols_ou


def train_and_save_models():
    df = load_data(DATA_FILES)
    X_result, y_result, X_btts, y_btts, X_over25, y_over25, form_cols, odds1x2_cols, oddsou_cols = build_feature_matrix(df, window=5)

    # --- Diagnostics ---
    print(f"Total matches loaded: {len(df)}")
    print(f"Usable rows for training: {len(X_result)}")
    if len(X_result) == 0:
        raise ValueError(
            "No valid rows available for training. "
            "Possible causes: missing Bet365 columns or too few matches per team."
        )

    # --- Fill missing values rather than drop ---
    X_result = X_result.fillna(X_result.mean())
    X_btts = X_btts.fillna(X_btts.mean())
    X_over25 = X_over25.fillna(X_over25.mean())

    # --- Train/Test splits ---
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_result, y_result, test_size=0.2, random_state=42)
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(X_btts, y_btts, test_size=0.2, random_state=42)
    Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(X_over25, y_over25, test_size=0.2, random_state=42)

    # --- Apply recency weighting (newer matches get more influence) ---
    df["years_ago"] = (df["Date"].max() - df["Date"]).dt.days / 365
    df["weight"] = np.exp(-0.25 * df["years_ago"])

    # Align weights with train indices for each model
    w_result = df.loc[Xr_tr.index, "weight"]
    w_btts = df.loc[Xb_tr.index, "weight"]
    w_over25 = df.loc[Xo_tr.index, "weight"]

    clf_result = RandomForestClassifier(
        n_estimators=800,
        max_depth=15,  # prevents overfitting but allows complexity
        min_samples_leaf=2,  # stabilizes leaf splits
        random_state=42,
        n_jobs=-1
    )
    clf_btts = RandomForestClassifier(
        n_estimators=800,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    clf_over25 = RandomForestClassifier(
        n_estimators=800,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    clf_result.fit(Xr_tr, yr_tr, sample_weight=w_result)
    clf_btts.fit(Xb_tr, yb_tr, sample_weight=w_btts)
    clf_over25.fit(Xo_tr, yo_tr, sample_weight=w_over25)

    print("RESULT model accuracy:", accuracy_score(yr_te, clf_result.predict(Xr_te)))
    print("BTTS model accuracy:", accuracy_score(yb_te, clf_btts.predict(Xb_te)))
    print("Over2.5 model accuracy:", accuracy_score(yo_te, clf_over25.predict(Xo_te)))

    joblib.dump({
        "model": clf_result,
        "form_cols": form_cols,
        "odds1x2_cols": odds1x2_cols,
        "oddsou_cols": oddsou_cols,
    }, "result_model.pkl")

    joblib.dump({
        "model": clf_btts,
        "form_cols": form_cols,
        "odds1x2_cols": odds1x2_cols,
        "oddsou_cols": oddsou_cols,
    }, "btts_model.pkl")

    joblib.dump({
        "model": clf_over25,
        "form_cols": form_cols,
        "odds1x2_cols": odds1x2_cols,
        "oddsou_cols": oddsou_cols,
    }, "over25_model.pkl")

    print("✅ Saved: result_model.pkl, btts_model.pkl, over25_model.pkl")

if __name__ == "__main__":
    train_and_save_models()
