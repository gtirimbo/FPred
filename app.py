# app.py
import streamlit as st
import pickle
from FootballModel import FootballModel

st.set_page_config(page_title="Football Prediction App", layout="wide")

st.title("Football Prediction App")
st.write("Select a league, home team, and away team to see expected goals and probabilities.")

# --- Define available leagues and pickles ---

league_files = {
    "EPL": "model_pl.pkl",
    "La Liga": "model_li.pkl",
    "Bundesliga": "model_bl.pkl",
    "Serie A": "model_sa.pkl",
    "Ligue 1": "model_l1.pkl"
}

# --- League selection ---
league = st.selectbox("Select League", list(league_files.keys()))

# --- Load pickled model safely ---
with open(league_files[league], "rb") as f:
    model = pickle.load(f)

# --- Team selection ---
team_h = st.selectbox("Home Team", model.params.index, index=0)
team_a = st.selectbox("Away Team", model.params.index, index=1)

# --- Predict button ---
if st.button("Predict"):
    # Expected goals
    xG_home, xG_away = model.predict(team_h, team_a)
    st.subheader("Expected Goals")
    st.write(f"{team_h}: {xG_home:.2f}, {team_a}: {xG_away:.2f}")

    # Outcome probabilities
    st.subheader("Match Outcome Probabilities")
    probs = model.outcome_probabilities(team_h, team_a, max_goals=10)
    st.json(probs)

    # Scoreline probability table
    st.subheader("Scoreline Probability Table")
    table = model.probability_table(team_h, team_a, max_goals=5)
    st.dataframe(table.style.format("{:.3f}"))

    # 1X2 bar chart
    st.subheader("1X2 Probabilities")
