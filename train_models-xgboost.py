# =============================
# train_models.py (XGBoost version)
# =============================
# Uses team form (last 5 matches) + Bet365 odds (if available)
# and recency weighting to train:
#   1) Full-time Result (H/D/A)
#   2) Both Teams To Score (Yes)
#   3) Over 2.5 Goals (Yes)
# Saves: result_model.pkl, btts_model.pkl, over25_model.pkl
# ---------------------------------

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib
from xgboost.callback import EarlyStopping

DATA_FILES = sorted([str(p) for p in Path(".").glob("E0*.csv")])
DATE_COL = "Date"

ODDS_1X2 = ["B365H", "B365D", "B365A"]
ODDS_OU = ["B365>2.5", "B365<2.5"]


def load_data(files):
    dfs = []
    for f in files:
        p = Path(f)
        df = pd.read_csv(p)
        print(f"{p.name}: {len(df)} matches")
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"]).sort_values(DATE_COL)
    print(f"Total matches loaded: {len(df)}")
    return df


def add_targets(df):
    df["target_result"] = df["FTR"].map({"A": 0, "D": 1, "H": 2})
    df["target_btts"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
    df["target_over25"] = ((df["FTHG"] + df["FTAG"]) >= 3).astype(int)
    return df


def compute_team_form_features(df, window=7):
    """
    Compute rolling form stats for each team (shots, corners, cards, goals)
    plus goal-scoring trends over recent matches.
    """
    home = df[[DATE_COL, "HomeTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST",
               "HC", "AC", "HY", "AY", "HR", "AR"]].copy()
    home.rename(columns={
        "HomeTeam": "Team", "FTHG": "GF", "FTAG": "GA",
        "HS": "S", "AS": "S_opp", "HST": "ST", "AST": "ST_opp",
        "HC": "C", "AC": "C_opp", "HY": "Y", "AY": "Y_opp",
        "HR": "R", "AR": "R_opp"
    }, inplace=True)

    away = df[[DATE_COL, "AwayTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST",
               "HC", "AC", "HY", "AY", "HR", "AR"]].copy()
    away.rename(columns={
        "AwayTeam": "Team", "FTHG": "GA", "FTAG": "GF",
        "HS": "S_opp", "AS": "S", "HST": "ST_opp", "AST": "ST",
        "HC": "C_opp", "AC": "C", "HY": "Y_opp", "AY": "Y",
        "HR": "R_opp", "AR": "R"
    }, inplace=True)

    long_df = pd.concat([home, away], ignore_index=True)
    long_df.sort_values(["Team", DATE_COL], inplace=True)

    roll_cols = ["GF", "GA", "S", "S_opp", "ST", "ST_opp",
                 "C", "C_opp", "Y", "Y_opp", "R", "R_opp"]

    # Rolling averages (form window)
    for col in roll_cols:
        long_df[f"{col}_ma{window}"] = (
            long_df.groupby("Team")[col]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

    # --- Goal scoring trend: goals scored, conceded, goal diff
    long_df["GF_trend"] = long_df.groupby("Team")["GF"].transform(
        lambda s: s.shift(1).ewm(span=window, adjust=False).mean()
    )
    long_df["GA_trend"] = long_df.groupby("Team")["GA"].transform(
        lambda s: s.shift(1).ewm(span=window, adjust=False).mean()
    )
    long_df["GD_trend"] = long_df["GF_trend"] - long_df["GA_trend"]

    # Merge back into main DataFrame
    feats = long_df[[DATE_COL, "Team"] +
                    [f"{c}_ma{window}" for c in roll_cols] +
                    ["GF_trend", "GA_trend", "GD_trend"]]

    df = df.merge(feats.add_prefix("home_"), left_on=["HomeTeam", DATE_COL],
                  right_on=["home_Team", "home_Date"], how="left")
    df = df.merge(feats.add_prefix("away_"), left_on=["AwayTeam", DATE_COL],
                  right_on=["away_Team", "away_Date"], how="left")

    df.drop(columns=[c for c in df.columns if c.endswith("_Team") or c.endswith("_Date")], inplace=True)

    # --- Relative form difference features (include goal trends)
    cols_to_diff = [f"{stat}_ma{window}" for stat in ["GF", "GA", "S", "ST", "C", "Y", "R"]] + \
                   ["GF_trend", "GA_trend", "GD_trend"]
    for col in cols_to_diff:
        home_col = f"home_{col}"
        away_col = f"away_{col}"
        if home_col in df.columns and away_col in df.columns:
            df[f"diff_{col}"] = df[home_col] - df[away_col]

    return df


def add_odds_features(df):
    for col in ODDS_1X2 + ODDS_OU:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if all(c in df.columns for c in ODDS_1X2):
        df["B365H_prob"] = 1 / df["B365H"]
        df["B365D_prob"] = 1 / df["B365D"]
        df["B365A_prob"] = 1 / df["B365A"]
        s = df[["B365H_prob", "B365D_prob", "B365A_prob"]].sum(axis=1)
        df[["B365H_prob", "B365D_prob", "B365A_prob"]] = df[["B365H_prob", "B365D_prob", "B365A_prob"]].div(s, axis=0)

    if all(c in df.columns for c in ODDS_OU):
        df["B365_over25_prob"] = 1 / df["B365>2.5"]
        df["B365_under25_prob"] = 1 / df["B365<2.5"]
        s2 = df[["B365_over25_prob", "B365_under25_prob"]].sum(axis=1)
        df[["B365_over25_prob", "B365_under25_prob"]] = df[["B365_over25_prob", "B365_under25_prob"]].div(s2, axis=0)

    # Normalize odds to mean 0, std 1 (z-score)
    for col in ["B365H_prob", "B365D_prob", "B365A_prob", "B365_over25_prob", "B365_under25_prob"]:
        if col in df.columns:
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)

    # Standardize all numeric feature columns globally (exclude targets)
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if not c.startswith("target_")
    ]
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / (df[numeric_cols].std() + 1e-6)

    return df


def build_feature_matrix(df, window=7):
    """
    Construct feature matrices for Result, BTTS, and Over2.5 models.
    Includes rolling team form, attack/defense form, team strength, head-to-head, and odds.
    Optimized to avoid DataFrame fragmentation warnings.
    """
    df = add_targets(df.copy())
    df = compute_team_form_features(df, window)
    df = compute_attack_defense_form(df)

    print("DEBUG: Columns after attack/defense form ->",
          [c for c in df.columns if "attack_form" in c or "defense_form" in c])

    df = compute_team_strength(df)
    df = add_head_to_head(df)
    df = add_odds_features(df)

    # --- Feature groups
    form_cols = [
        f"home_{c}_ma{window}" for c in ["GF", "GA", "S", "S_opp", "ST", "ST_opp", "C", "C_opp", "Y", "Y_opp", "R", "R_opp"]
    ] + [
        f"away_{c}_ma{window}" for c in ["GF", "GA", "S", "S_opp", "ST", "ST_opp", "C", "C_opp", "Y", "Y_opp", "R", "R_opp"]
    ]

    odds_cols_result = ["B365H_prob", "B365D_prob", "B365A_prob"] if "B365H_prob" in df.columns else []
    odds_cols_ou = ["B365_over25_prob", "B365_under25_prob"] if "B365_over25_prob" in df.columns else []

    # --- Add contextual columns if missing (optimized batch creation)
    extra_cols = [
        "HomeTeam_strength", "AwayTeam_strength",
        "strength_diff", "strength_sum",
        "h2h_avg_FTHG", "h2h_avg_FTAG"
    ]
    missing_cols = [c for c in extra_cols if c not in df.columns]
    if missing_cols:
        df = pd.concat(
            [df, pd.DataFrame(np.nan, index=df.index, columns=missing_cols)],
            axis=1
        )

    # --- Create feature sets
    X_result = df[form_cols + odds_cols_result].copy()
    X_btts = df[form_cols + odds_cols_result + odds_cols_ou].copy()
    X_over25 = df[form_cols + odds_cols_ou + odds_cols_result].copy()

    # --- Drop incomplete rows
    mask = X_result.notna().all(axis=1)
    X_result, X_btts, X_over25 = X_result[mask], X_btts[mask], X_over25[mask]
    y_result, y_btts, y_over25 = (
        df.loc[mask, "target_result"],
        df.loc[mask, "target_btts"],
        df.loc[mask, "target_over25"],
    )

    # --- Clip extreme values for stability
    for X in [X_result, X_btts, X_over25]:
        X.clip(-5, 5, inplace=True)

    # --- Defragment (optional small speed-up)
    df = df.copy()

    return X_result, y_result, X_btts, y_btts, X_over25, y_over25, form_cols, odds_cols_result, odds_cols_ou

def compute_team_strength(df):
    """
    Compute a time-decayed team strength score: exponential weighted average of
    goal difference over the last 10 matches, with more weight on recent results.
    """
    df["GoalDiff"] = df["FTHG"] - df["FTAG"]
    ratings = []

    for team_col in ["HomeTeam", "AwayTeam"]:
        tmp = df[[DATE_COL, team_col, "GoalDiff"]].copy()
        tmp = tmp.rename(columns={team_col: "Team"})
        tmp.sort_values(["Team", DATE_COL], inplace=True)

        # Exponentially weighted moving average (like ELO-style decay)
        tmp["strength"] = (
            tmp.groupby("Team")["GoalDiff"]
            .transform(lambda s: s.shift(1).ewm(span=10, adjust=False).mean())
        )

        # Rename back for merge
        tmp.rename(columns={"Team": team_col, "strength": f"{team_col}_strength"}, inplace=True)
        ratings.append(tmp[[DATE_COL, team_col, f"{team_col}_strength"]])

    for r in ratings:
        df = df.merge(r, on=[DATE_COL, r.columns[1]], how="left")

    return df

def add_head_to_head(df):
    df = df.sort_values(DATE_COL)
    df["fixture_key"] = df["HomeTeam"] + "_" + df["AwayTeam"]

    prev = (
        df.groupby("fixture_key")[["FTHG", "FTAG"]]
        .shift(1)
        .rolling(3, min_periods=1)
        .mean()
        .rename(columns={"FTHG": "h2h_avg_FTHG", "FTAG": "h2h_avg_FTAG"})
    )
    return pd.concat([df, prev], axis=1)

def compute_attack_defense_form(df, span=7):
    """
    Compute attacking and defensive form ratings per team
    using exponentially weighted averages of goals and shots on target.
    Produces: HomeTeam_attack_form, AwayTeam_attack_form,
              HomeTeam_defense_form, AwayTeam_defense_form,
              attack_diff, defense_diff
    """
    df = df.copy()
    df["GoalDiff"] = df["FTHG"] - df["FTAG"]

    ratings = []
    for team_col, gf_col, ga_col, st_col, st_opp_col in [
        ("HomeTeam", "FTHG", "FTAG", "HST", "AST"),
        ("AwayTeam", "FTAG", "FTHG", "AST", "HST")
    ]:
        tmp = df[[DATE_COL, team_col, gf_col, ga_col, st_col, st_opp_col]].rename(
            columns={
                team_col: "Team",
                gf_col: "GF",
                ga_col: "GA",
                st_col: "ST",
                st_opp_col: "ST_opp",
            }
        )

        tmp.sort_values(["Team", DATE_COL], inplace=True)

        # Calculate exponentially weighted attacking & defensive form
        tmp["attack_form"] = (
            (tmp["GF"] + 0.5 * tmp["ST"]).shift(1)
            .groupby(tmp["Team"])
            .transform(lambda s: s.ewm(span=span, adjust=False).mean())
        )
        tmp["defense_form"] = (
            (tmp["GA"] + 0.5 * tmp["ST_opp"]).shift(1)
            .groupby(tmp["Team"])
            .transform(lambda s: s.ewm(span=span, adjust=False).mean())
        )

        tmp.rename(
            columns={
                "Team": team_col,
                "attack_form": f"{team_col}_attack_form",
                "defense_form": f"{team_col}_defense_form",
            },
            inplace=True,
        )

        ratings.append(tmp[[DATE_COL, team_col, f"{team_col}_attack_form", f"{team_col}_defense_form"]])

    # Merge results back efficiently
    for r in ratings:
        df = df.merge(r, on=[DATE_COL, r.columns[1]], how="left", copy=False)

    # Derived relative strength
    df["attack_diff"] = df["HomeTeam_attack_form"] - df["AwayTeam_attack_form"]
    df["defense_diff"] = df["HomeTeam_defense_form"] - df["AwayTeam_defense_form"]

    # Fill NaNs early games with neutral values (optional)
    df[["HomeTeam_attack_form", "AwayTeam_attack_form",
        "HomeTeam_defense_form", "AwayTeam_defense_form",
        "attack_diff", "defense_diff"]] = df[[
            "HomeTeam_attack_form", "AwayTeam_attack_form",
            "HomeTeam_defense_form", "AwayTeam_defense_form",
            "attack_diff", "defense_diff"
        ]].fillna(0)

    return df

def train_and_save_models():
    print("\n🏗️  Starting training pipeline...\n")

    # --- Load and prepare data
    df = load_data(DATA_FILES)
    X_result, y_result, X_btts, y_btts, X_over25, y_over25, form_cols, odds1x2_cols, oddsou_cols = build_feature_matrix(df)

    print(f"📊  Usable matches for training: {len(X_result):,}")
    print("⚽  Target distribution (Result):")
    print(y_result.value_counts(normalize=True).rename_axis("Outcome").map(lambda x: f"{x:.2%}"))
    print()

    # --- Recency weighting
    df["years_ago"] = (df["Date"].max() - df["Date"]).dt.days / 365
    df["weight"] = np.exp(-0.25 * df["years_ago"])

    # --- Train/test splits
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_result, y_result, test_size=0.2, random_state=42)
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(X_btts, y_btts, test_size=0.2, random_state=42)
    Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(X_over25, y_over25, test_size=0.2, random_state=42)

    # --- Sample weights
    from sklearn.utils.class_weight import compute_sample_weight
    w_result_final = df.loc[Xr_tr.index, "weight"] * compute_sample_weight(class_weight="balanced", y=yr_tr)
    w_btts_final   = df.loc[Xb_tr.index, "weight"] * compute_sample_weight(class_weight="balanced", y=yb_tr)
    w_over25_final = df.loc[Xo_tr.index, "weight"] * compute_sample_weight(class_weight="balanced", y=yo_tr)

    # --- Define models
    print("🚀  Training models...\n")
    clf_result = XGBClassifier(
        n_estimators=1200, max_depth=9, learning_rate=0.02,
        subsample=0.9, colsample_bytree=0.85, gamma=0.4,
        reg_lambda=1.8, reg_alpha=0.1, eval_metric="mlogloss",
        random_state=42, n_jobs=-1
    )
    clf_btts = XGBClassifier(
        n_estimators=900, max_depth=8, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9, gamma=0.4,
        reg_lambda=1.5, reg_alpha=0.2, min_child_weight=1,
        eval_metric="logloss", random_state=42, n_jobs=-1
    )
    clf_over25 = XGBClassifier(
        n_estimators=900, max_depth=8, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9, gamma=0.4,
        reg_lambda=1.5, reg_alpha=0.2, min_child_weight=1,
        eval_metric="logloss", random_state=42, n_jobs=-1
    )

    # --- Train
    clf_result.fit(Xr_tr, yr_tr, sample_weight=w_result_final)
    clf_btts.fit(Xb_tr, yb_tr, sample_weight=w_btts_final)
    clf_over25.fit(Xo_tr, yo_tr, sample_weight=w_over25_final)

    # --- Evaluate
    acc_result = accuracy_score(yr_te, clf_result.predict(Xr_te))
    acc_btts   = accuracy_score(yb_te, clf_btts.predict(Xb_te))
    acc_over25 = accuracy_score(yo_te, clf_over25.predict(Xo_te))

    print("✅  Training complete!\n")
    print(f"🎯  Model Accuracies:")
    print(f"   • Result (Win/Draw/Loss): {acc_result:.3f}")
    print(f"   • BTTS (Both Teams To Score): {acc_btts:.3f}")
    print(f"   • Over 2.5 Goals: {acc_over25:.3f}\n")

    # --- Save
    joblib.dump({
        "model": clf_result, "form_cols": form_cols,
        "odds1x2_cols": odds1x2_cols, "oddsou_cols": oddsou_cols,
    }, "result_model.pkl")
    joblib.dump({
        "model": clf_btts, "form_cols": form_cols,
        "odds1x2_cols": odds1x2_cols, "oddsou_cols": oddsou_cols,
    }, "btts_model.pkl")
    joblib.dump({
        "model": clf_over25, "form_cols": form_cols,
        "odds1x2_cols": odds1x2_cols, "oddsou_cols": oddsou_cols,
    }, "over25_model.pkl")

    print("💾  Saved models:")
    print("   • result_model.pkl")
    print("   • btts_model.pkl")
    print("   • over25_model.pkl\n")

    # --- Feature importance summaries
    def print_top_features(model, label):
        importances = model.get_booster().get_score(importance_type='weight')
        top_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:8]
        print(f"🏅  Top features ({label}):")
        for feat, score in top_feats:
            print(f"      {feat:25s} {score:.0f}")
        print()

    print_top_features(clf_result, "Result")
    print_top_features(clf_btts, "BTTS")
    print_top_features(clf_over25, "Over 2.5")

    print("✅  All tasks complete.\n")


if __name__ == "__main__":
    train_and_save_models()
