import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="EPL Fixture Predictor", page_icon="⚽", layout="wide")

DATA_FILES = sorted([str(p) for p in Path(".").glob("E0*.csv")])
print(f"Found data files: {DATA_FILES}")
DATE_COL = "Date"


@st.cache_data
def load_matches():
    dfs = []
    for f in DATA_FILES:
        p = Path(f)
        if not p.exists():
            st.error(f"Missing data file: {f}")
            st.stop()
        df = pd.read_csv(p)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["HomeTeam", "AwayTeam"]).sort_values(DATE_COL)

    return df


@st.cache_resource
def load_bundle(path):
    return joblib.load(path)


matches = load_matches()

teams = sorted(pd.unique(pd.concat([matches["HomeTeam"], matches["AwayTeam"]]).dropna()))

# Sidebar inputs
st.sidebar.header("Fixture Input")
home_team = st.sidebar.selectbox("Home Team", teams, index=0)
away_team = st.sidebar.selectbox("Away Team", teams, index=1 if len(teams) > 1 else 0)

st.sidebar.markdown("### Bet365 Odds (enter fractions, e.g. 5/2, 6/4, evens)")

def parse_fractional_odds(fraction):
    fraction = fraction.strip().lower()
    if fraction in ["evs", "evens"]:
        return 2.0
    try:
        num, den = fraction.split("/")
        return float(num) / float(den) + 1.0
    except Exception:
        return None

home_input = st.sidebar.text_input("Home Win (e.g. 6/4)", value="11/10")
draw_input = st.sidebar.text_input("Draw (e.g. 5/2)", value="12/5")
away_input = st.sidebar.text_input("Away Win (e.g. 3/1)", value="9/4")

over_input = st.sidebar.text_input("Over 2.5 (e.g. 10/11)", value="10/11")
under_input = st.sidebar.text_input("Under 2.5 (e.g. 4/5)", value="4/5")

# Convert to decimals
h_odds = parse_fractional_odds(home_input) or 2.2
d_odds = parse_fractional_odds(draw_input) or 3.4
a_odds = parse_fractional_odds(away_input) or 3.1

o_over = parse_fractional_odds(over_input) or 2.0
o_under = parse_fractional_odds(under_input) or 1.8

st.title("⚽ EPL Fixture Predictor")
st.caption("Uses each team's last 5 matches + optional Bet365 odds to predict Result, BTTS, and Over 2.5.")


def team_last5_features(df, team, window=5):
    mask_home = df["HomeTeam"] == team
    mask_away = df["AwayTeam"] == team
    sub = df.loc[mask_home | mask_away].copy()
    sub.sort_values(DATE_COL, inplace=True)

    # Home perspective
    sub_home = sub.loc[mask_home].copy()
    sub_home["GF"], sub_home["GA"] = sub_home["FTHG"], sub_home["FTAG"]
    sub_home["S"], sub_home["S_opp"] = sub_home["HS"], sub_home["AS"]
    sub_home["ST"], sub_home["ST_opp"] = sub_home["HST"], sub_home["AST"]
    sub_home["C"], sub_home["C_opp"] = sub_home["HC"], sub_home["AC"]
    sub_home["Y"], sub_home["Y_opp"] = sub_home["HY"], sub_home["AY"]
    sub_home["R"], sub_home["R_opp"] = sub_home["HR"], sub_home["AR"]

    # Away perspective
    sub_away = sub.loc[mask_away].copy()
    sub_away["GF"], sub_away["GA"] = sub_away["FTAG"], sub_away["FTHG"]
    sub_away["S"], sub_away["S_opp"] = sub_away["AS"], sub_away["HS"]
    sub_away["ST"], sub_away["ST_opp"] = sub_away["AST"], sub_away["HST"]
    sub_away["C"], sub_away["C_opp"] = sub_away["AC"], sub_away["HC"]
    sub_away["Y"], sub_away["Y_opp"] = sub_away["AY"], sub_away["HY"]
    sub_away["R"], sub_away["R_opp"] = sub_away["AR"], sub_away["HR"]

    uni = pd.concat([sub_home, sub_away], ignore_index=True)
    uni.sort_values(DATE_COL, inplace=True)

    last5 = uni.tail(window)
    means = last5[["GF", "GA", "S", "S_opp", "ST", "ST_opp", "C", "C_opp", "Y", "Y_opp", "R", "R_opp"]].mean()
    return means


result_bundle = load_bundle("result_model.pkl")
btts_bundle = load_bundle("btts_model.pkl")
over25_bundle = load_bundle("over25_model.pkl")

res_model = result_bundle["model"]
btts_model = btts_bundle["model"]
over25_model = over25_bundle["model"]

form_cols = result_bundle["form_cols"]
odds1x2_cols = result_bundle["odds1x2_cols"]
oddsou_cols = result_bundle["oddsou_cols"]

home_feats = team_last5_features(matches, home_team)
away_feats = team_last5_features(matches, away_team)

row = {}
for c in ["GF", "GA", "S", "S_opp", "ST", "ST_opp", "C", "C_opp", "Y", "Y_opp", "R", "R_opp"]:
    row[f"home_{c}_ma5"] = home_feats.get(c, np.nan)
    row[f"away_{c}_ma5"] = away_feats.get(c, np.nan)

H_prob, D_prob, A_prob = 1/h_odds, 1/d_odds, 1/a_odds
s = H_prob + D_prob + A_prob
H_prob, D_prob, A_prob = H_prob/s, D_prob/s, A_prob/s

OU_over, OU_under = 1/o_over, 1/o_under
s2 = OU_over + OU_under
OU_over, OU_under = OU_over/s2, OU_under/s2

if odds1x2_cols:
    row["B365H_prob"], row["B365D_prob"], row["B365A_prob"] = H_prob, D_prob, A_prob
if oddsou_cols:
    row["B365_over25_prob"], row["B365_under25_prob"] = OU_over, OU_under

# --- NEW: Add placeholders for extended features from training
# Attack/Defense form (approximation using team scoring trends)
row["HomeTeam_attack_form"] = home_feats["GF"] * 0.7 + home_feats["ST"] * 0.3
row["AwayTeam_attack_form"] = away_feats["GF"] * 0.7 + away_feats["ST"] * 0.3
row["HomeTeam_defense_form"] = home_feats["GA"] * 0.7 + home_feats["ST_opp"] * 0.3
row["AwayTeam_defense_form"] = away_feats["GA"] * 0.7 + away_feats["ST_opp"] * 0.3

row["attack_diff"] = row["HomeTeam_attack_form"] - row["AwayTeam_attack_form"]
row["defense_diff"] = row["HomeTeam_defense_form"] - row["AwayTeam_defense_form"]

# Team strength and head-to-head placeholders
row["HomeTeam_strength"] = 0.0
row["AwayTeam_strength"] = 0.0
row["strength_diff"] = 0.0
row["strength_sum"] = 0.0
row["h2h_avg_FTHG"] = 0.0
row["h2h_avg_FTAG"] = 0.0

X_result = pd.DataFrame([{col: row.get(col, np.nan) for col in form_cols + odds1x2_cols}])
X_btts   = pd.DataFrame([{col: row.get(col, np.nan) for col in form_cols + odds1x2_cols + oddsou_cols}])
X_over25 = pd.DataFrame([{col: row.get(col, np.nan) for col in form_cols + oddsou_cols + odds1x2_cols}])

# Fill any missing values (in case of few matches)
X_result = X_result.fillna(X_result.mean())
X_btts = X_btts.fillna(X_btts.mean())
X_over25 = X_over25.fillna(X_over25.mean())

# Predict probabilities
res_proba = res_model.predict_proba(X_result)[0]
res_labels = ["Away Win", "Draw", "Home Win"]  # matches encoding 0=A,1=D,2=H
res_pick = res_labels[int(np.argmax(res_proba))]

btts_proba_yes = btts_model.predict_proba(X_btts)[0][1]
over25_proba_yes = over25_model.predict_proba(X_over25)[0][1]

# --- Compare model vs bookmaker implied probabilities
bookmaker_probs = {
    "Home Win": H_prob,
    "Draw": D_prob,
    "Away Win": A_prob
}
model_probs = {
    "Home Win": float(res_proba[2]),
    "Draw": float(res_proba[1]),
    "Away Win": float(res_proba[0])
}

# Compute % differences between model and bookmaker
comparison = {}
for k in model_probs:
    comparison[k] = {
        "model": model_probs[k] * 100,
        "book": bookmaker_probs[k] * 100,
        "delta": (model_probs[k] - bookmaker_probs[k]) * 100
    }

# --- Generate qualitative insight
insights = []
for outcome, vals in comparison.items():
    diff = vals["delta"]
    if abs(diff) < 3:
        continue  # skip near-identical probabilities
    if diff > 0:
        insights.append(f"📈 Model is **more confident** in {outcome} than the bookmaker (+{diff:.1f}%).")
    else:
        insights.append(f"📉 Model is **less confident** in {outcome} than the bookmaker ({diff:.1f}%).")

# Summary headline
if res_pick in comparison:
    confidence = comparison[res_pick]["model"]
    market = comparison[res_pick]["book"]
    if confidence > market:
        summary = f"💰 **Value Signal:** Model favours **{res_pick} ({confidence:.1f}%)** vs market ({market:.1f}%)."
    else:
        summary = f"⚠️ **Caution:** Market favours **{res_pick} ({market:.1f}%)**, but model is less confident ({confidence:.1f}%)."
else:
    summary = "No strong difference between model and market probabilities."


# Display Results
left, right = st.columns([3, 2])
with left:
    st.subheader("Fixture")
    st.write(f"**{home_team}** vs **{away_team}**")

    st.subheader("Predictions")
    st.write(f"**BTTS:** {btts_proba_yes * 100:.1f}% chance of Yes")
    st.write(f"**Over 2.5 Goals:** {over25_proba_yes * 100:.1f}% chance of Yes")

    st.write("**Full-Time Result Probabilities:**")
    st.progress(float(res_proba[2]))
    st.write(f"Home Win: {res_proba[2] * 100:.1f}%")
    st.progress(float(res_proba[1]))
    st.write(f"Draw: {res_proba[1] * 100:.1f}%")
    st.progress(float(res_proba[0]))
    st.write(f"Away Win: {res_proba[0] * 100:.1f}%")

    st.success(f"🏆 Game Prediction: **{res_pick}**")

    st.markdown("---")
    st.subheader("📊 Model vs Market Analysis")

    col1, col2, col3 = st.columns(3)
    for label, vals in comparison.items():
        with eval(f"col{['Away Win','Draw','Home Win'].index(label)+1}"):
            delta = vals["delta"]
            color = "🟩" if delta > 0 else ("🟥" if delta < 0 else "⬜️")
            st.metric(label, f"{vals['model']:.1f}%", f"{delta:+.1f}% vs book", delta_color="off")
            st.caption(f"Bookmaker: {vals['book']:.1f}% {color}")

    st.markdown("---")
    st.markdown(summary)
    if insights:
        for i in insights:
            st.write(i)
    else:
        st.info("Model and bookmaker are broadly aligned on this fixture.")


with right:
    st.subheader("📊 Recent Form (Last 5 Matches)")
    st.caption("Average stats from each team's last five games (league only).")

    # --- Define metric groups with readable labels
    metric_groups = {
        "⚽ Offensive": {
            "GF": "Goals Scored",
            "ST": "Shots on Target",
            "S": "Total Shots",
            "C": "Corners Won"
        },
        "🛡️ Defensive": {
            "GA": "Goals Conceded",
            "ST_opp": "Shots on Target Faced",
            "S_opp": "Shots Faced",
            "C_opp": "Corners Conceded"
        },
        "🚨 Discipline": {
            "Y": "Yellow Cards",
            "R": "Red Cards"
        }
    }

    # --- Helper to build a labeled dataframe for each group
    def build_form_table(group_dict):
        rows = []
        for key, label in group_dict.items():
            h_val = home_feats.get(key, np.nan)
            a_val = away_feats.get(key, np.nan)
            rows.append({
                "Metric": label,
                f"{home_team} (Home Avg)": round(h_val, 2),
                f"{away_team} (Away Avg)": round(a_val, 2),
            })
        return pd.DataFrame(rows)

    # --- Render each section
    for section, metrics in metric_groups.items():
        st.markdown(f"### {section}")
        df_section = build_form_table(metrics)
        styled_df = df_section.style.highlight_max(
            axis=1, subset=df_section.columns[1:], color="#C3E6CB"
        )

        st.dataframe(styled_df, hide_index=True, use_container_width=True)

st.caption("Note: Predictions use the last-5 match form + optional Bet365 implied probabilities.")