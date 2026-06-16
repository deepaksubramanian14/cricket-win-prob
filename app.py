"""
Streamlit web app for cricket win probability models.

Supports both Test cricket (3-class: team1_win / team2_win / draw) and
ODI cricket (binary: team1_win / team2_win). User picks format in the
sidebar, then chooses replay / manual / live mode.

Run locally:    streamlit run app.py
Deploy:         push repo to GitHub, connect at share.streamlit.io
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st
import xgboost as xgb
import altair as alt

DATA_DIR = Path("data")

# ----------------------------- Format configs -----------------------------

# ODI constants
ODI_BALLS_PER_INNINGS = 300
ODI_POWERPLAY_END = 10
ODI_MIDDLE_END = 40

ODI_FEATURE_COLUMNS = [
    "innings_num", "over", "ball_in_innings", "ball_in_match",
    "phase", "is_chasing",
    "innings_runs", "innings_wickets", "wickets_in_hand",
    "balls_remaining_in_innings", "current_run_rate",
    "target", "runs_needed", "required_run_rate", "rr_gap",
    "partnership_runs", "partnership_balls",
    "recent_dot_pct", "recent_run_rate", "recent_boundaries",
    "balls_since_wicket",
    "career_bat_avg_odi", "career_bat_sr_odi",
    "career_bowl_avg_odi", "career_bowl_sr_odi", "career_bowl_econ_odi",
    "is_team1_batting",
]

ODI_DEFAULT_CAREER = {
    "career_bat_avg_odi": 25.0,
    "career_bat_sr_odi": 80.0,
    "career_bowl_avg_odi": 35.0,
    "career_bowl_sr_odi": 40.0,
    "career_bowl_econ_odi": 5.5,
}

# Test constants
TEST_OVERS_PER_DAY = 90
TEST_BALLS_PER_DAY = TEST_OVERS_PER_DAY * 6      # 540
TEST_MAX_BALLS_IN_MATCH = TEST_BALLS_PER_DAY * 5  # 2700
TEST_FINAL_SESSION_BALLS = 180

TEST_FEATURE_COLUMNS = [
    "innings_num", "day_of_match", "over", "ball_in_innings", "ball_in_match",
    "balls_remaining_in_match", "balls_remaining_today", "days_remaining",
    "is_final_session",
    "innings_runs", "innings_wickets", "innings_run_rate", "current_lead",
    "partnership_runs", "partnership_balls",
    "recent_dot_pct", "recent_run_rate", "recent_boundaries",
    "balls_since_wicket",
    "career_bat_avg", "career_bat_sr",
    "career_bowl_avg", "career_bowl_sr", "career_bowl_econ",
    "is_team1_batting",
]

TEST_DEFAULT_CAREER = {
    "career_bat_avg": 30.0,
    "career_bat_sr": 50.0,
    "career_bowl_avg": 35.0,
    "career_bowl_sr": 60.0,
    "career_bowl_econ": 3.0,
}


# ----------------------------- Loading (cached) -----------------------------

@st.cache_resource
def load_test_model() -> xgb.Booster | None:
    p = DATA_DIR / "win_prob_model.json"
    if not p.exists():
        return None
    m = xgb.Booster()
    m.load_model(str(p))
    return m


@st.cache_resource
def load_odi_model() -> xgb.Booster | None:
    p = DATA_DIR / "win_prob_model_odi.json"
    if not p.exists():
        return None
    m = xgb.Booster()
    m.load_model(str(p))
    return m


@st.cache_data
def load_test_career() -> pd.DataFrame:
    p = DATA_DIR / "career_stats_test_lookup.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["player"] + list(TEST_DEFAULT_CAREER.keys()))
    return pd.read_parquet(p)


@st.cache_data
def load_odi_career() -> pd.DataFrame:
    p = DATA_DIR / "career_stats_odi_lookup.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["player"] + list(ODI_DEFAULT_CAREER.keys()))
    return pd.read_parquet(p)


def _resolve_teams(df: pd.DataFrame) -> tuple[str, str]:
    """Get team names from a famous-match CSV. Handles older CSVs that lack
    explicit team1/team2 columns by falling back to batting/bowling team."""
    cols = df.columns
    if "team1" in cols and "team2" in cols:
        return str(df.iloc[0]["team1"]), str(df.iloc[0]["team2"])
    if "team_1" in cols and "team_2" in cols:
        return str(df.iloc[0]["team_1"]), str(df.iloc[0]["team_2"])
    if "batting_team" in cols and "bowling_team" in cols:
        # First row's batting team batted first, so it's team1 by convention.
        return str(df.iloc[0]["batting_team"]), str(df.iloc[0]["bowling_team"])
    return "Team 1", "Team 2"


def _resolve_date(df: pd.DataFrame) -> str:
    for col in ("match_date", "date"):
        if col in df.columns:
            return str(df.iloc[0][col])[:10]
    return ""


def _resolve_outcome(df: pd.DataFrame) -> str:
    return str(df.iloc[0]["outcome"]) if "outcome" in df.columns else "unknown"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt older-schema famous-match CSVs to the current expected schema.

    Old schema (Test): p_<TeamName>_win columns embedded team names directly
    (e.g. p_England_win, p_Australia_win, p_draw) with no team1/team2/outcome.
    We extract team names from the columns, rename to standard prob_*, add
    team1/team2 columns, and derive outcome from the final-ball prediction.
    """
    cols = list(df.columns)

    # Find p_*_win columns that aren't already in standard form
    p_win_cols = [
        c for c in cols
        if c.startswith("p_") and c.endswith("_win")
        and c not in ("p_team1_win", "p_team2_win")
    ]

    if len(p_win_cols) >= 2:
        # Old per-team schema. Strip "p_" prefix and "_win" suffix.
        team1_name = p_win_cols[0][2:-4]
        team2_name = p_win_cols[1][2:-4]

        df = df.rename(columns={
            p_win_cols[0]: "prob_team1_win",
            p_win_cols[1]: "prob_team2_win",
        })
        if "p_draw" in cols:
            df = df.rename(columns={"p_draw": "prob_draw"})

        df["team1"] = team1_name
        df["team2"] = team2_name

        # Derive outcome from final-ball model prediction (model is ~97% accurate
        # at final ball, so this is usually right). Honest caveat: this is the
        # model's call, not the actual result, but it's good enough for display.
        last = df.iloc[-1]
        t1_p = last.get("prob_team1_win", 0)
        t2_p = last.get("prob_team2_win", 0)
        d_p = last.get("prob_draw", 0)
        if d_p >= t1_p and d_p >= t2_p:
            df["outcome"] = "draw"
        elif t1_p >= t2_p:
            df["outcome"] = "team1_win"
        else:
            df["outcome"] = "team2_win"

        if "day" in df.columns and "day_of_match" not in df.columns:
            df = df.rename(columns={"day": "day_of_match"})

    return df


@st.cache_data
def load_famous_matches(directory: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    folder = DATA_DIR / directory
    if not folder.exists():
        return out
    for csv in sorted(folder.glob("*_curve.csv")):
        df = pd.read_csv(csv)
        if len(df) == 0:
            continue
        df = _normalize_columns(df)
        # Skip CSVs that still lack the required columns after normalization
        if "prob_team1_win" not in df.columns or "prob_team2_win" not in df.columns:
            continue
        if "ball_in_match" not in df.columns:
            continue
        t1, t2 = _resolve_teams(df)
        date = _resolve_date(df)
        outcome = _resolve_outcome(df)
        if outcome == "team1_win":
            result = f"{t1} won (predicted)"
        elif outcome == "team2_win":
            result = f"{t2} won (predicted)"
        elif outcome == "draw":
            result = "draw (predicted)"
        else:
            result = outcome
        label = f"{t1} vs {t2} — {result}" if not date else f"{t1} vs {t2} ({date}) — {result}"
        out[label] = df
    return out


# ----------------------------- Shared utilities -----------------------------

def parse_over(over_str: str) -> int:
    s = str(over_str)
    if "." in s:
        ov, balls = s.split(".")
        return int(ov) * 6 + int(balls)
    return int(s) * 6


def lookup_career(name: str | None, career: pd.DataFrame, defaults: dict) -> dict:
    if not name or career.empty:
        return defaults.copy()
    hit = career[career["player"] == name]
    if len(hit) == 0:
        hit = career[career["player"].str.contains(name, case=False, na=False, regex=False)]
    if len(hit) == 0:
        return defaults.copy()
    row = hit.iloc[0]
    out = defaults.copy()
    for k in defaults:
        if pd.notna(row.get(k)):
            out[k] = float(row[k])
    return out


# ----------------------------- ODI feature computation -----------------------------

def compute_odi_features(state: dict, career: pd.DataFrame) -> dict:
    innings_num = int(state["innings_num"])
    is_chasing = 1 if innings_num == 2 else 0
    ball_in_innings = parse_over(state["current_over"])
    over_num = ball_in_innings // 6

    innings_runs = int(state["current_score"])
    innings_wickets = int(state["current_wickets"])
    wickets_in_hand = 10 - innings_wickets
    balls_remaining = max(0, ODI_BALLS_PER_INNINGS - ball_in_innings)
    current_rr = (innings_runs / (ball_in_innings / 6.0)) if ball_in_innings else 0.0

    target = int(state.get("target", 0))
    if is_chasing and target > 0:
        runs_needed = max(0, target - innings_runs)
        required_rr = (runs_needed / (balls_remaining / 6.0)) if (balls_remaining > 0 and runs_needed > 0) else 0.0
        rr_gap = required_rr - current_rr
    else:
        runs_needed, required_rr, rr_gap = 0, 0.0, 0.0

    phase = 0 if over_num < ODI_POWERPLAY_END else (1 if over_num < ODI_MIDDLE_END else 2)
    is_team1_batting = 1 if state["batting_team"] == state["team1"] else 0

    balls_in_inns_1 = int(state.get("balls_in_innings_1", ODI_BALLS_PER_INNINGS)) if is_chasing else 0
    ball_in_match = balls_in_inns_1 + ball_in_innings

    partnership_runs = int(state.get("partnership_runs", 0))
    partnership_balls = int(state.get("partnership_balls", 0))
    balls_since_wicket = int(state.get("balls_since_wicket", partnership_balls))

    last_6 = state.get("last_6_overs_runs")
    recent_rr = float(last_6) / 6.0 if last_6 is not None else current_rr
    recent_dot_pct = float(state.get("recent_dot_pct", 0.5))
    recent_boundaries = int(state.get("recent_boundaries", 0))

    bat = lookup_career(state.get("batter"), career, ODI_DEFAULT_CAREER)
    bowl = lookup_career(state.get("bowler"), career, ODI_DEFAULT_CAREER)

    return {
        "innings_num": innings_num,
        "over": over_num,
        "ball_in_innings": ball_in_innings,
        "ball_in_match": ball_in_match,
        "phase": phase,
        "is_chasing": is_chasing,
        "innings_runs": innings_runs,
        "innings_wickets": innings_wickets,
        "wickets_in_hand": wickets_in_hand,
        "balls_remaining_in_innings": balls_remaining,
        "current_run_rate": round(current_rr, 3),
        "target": target,
        "runs_needed": runs_needed,
        "required_run_rate": round(required_rr, 3),
        "rr_gap": round(rr_gap, 3),
        "partnership_runs": partnership_runs,
        "partnership_balls": partnership_balls,
        "recent_dot_pct": recent_dot_pct,
        "recent_run_rate": round(recent_rr, 3),
        "recent_boundaries": recent_boundaries,
        "balls_since_wicket": balls_since_wicket,
        **bat,
        **bowl,
        "is_team1_batting": is_team1_batting,
    }


def predict_odi(features: dict, model: xgb.Booster) -> tuple[float, float]:
    row = [features[c] for c in ODI_FEATURE_COLUMNS]
    d = xgb.DMatrix([row], feature_names=ODI_FEATURE_COLUMNS)
    p_t2 = float(model.predict(d)[0])
    return 1.0 - p_t2, p_t2


# ----------------------------- Test feature computation -----------------------------

def compute_test_features(state: dict, career: pd.DataFrame) -> dict:
    innings_num = int(state["innings_num"])
    ball_in_innings = parse_over(state["current_over"])
    over_num = ball_in_innings // 6

    # ball_in_match accumulates across innings. User supplies total match balls
    # bowled so far (across all innings up to current ball).
    ball_in_match = int(state["ball_in_match"])
    day_of_match = min(5, max(1, (ball_in_match - 1) // TEST_BALLS_PER_DAY + 1))

    balls_today_so_far = (ball_in_match - 1) % TEST_BALLS_PER_DAY + 1
    balls_remaining_in_match = max(0, TEST_MAX_BALLS_IN_MATCH - ball_in_match)
    balls_remaining_today = max(0, TEST_BALLS_PER_DAY - balls_today_so_far)
    days_remaining = max(0, 5 - day_of_match)
    is_final_session = int(balls_remaining_in_match <= TEST_FINAL_SESSION_BALLS)

    innings_runs = int(state["current_score"])
    innings_wickets = int(state["current_wickets"])
    innings_run_rate = (innings_runs / (ball_in_innings / 6.0)) if ball_in_innings else 0.0

    # current_lead is the SIGNED lead of team1 over team2 (negative if team2 ahead).
    # User supplies team1_total_runs and team2_total_runs across the match so far.
    team1_total = int(state["team1_total_runs"])
    team2_total = int(state["team2_total_runs"])
    current_lead = team1_total - team2_total

    is_team1_batting = 1 if state["batting_team"] == state["team1"] else 0

    partnership_runs = int(state.get("partnership_runs", 0))
    partnership_balls = int(state.get("partnership_balls", 0))
    balls_since_wicket = int(state.get("balls_since_wicket", partnership_balls))

    last_10 = state.get("last_10_overs_runs")
    recent_rr = float(last_10) / 10.0 if last_10 is not None else innings_run_rate
    recent_dot_pct = float(state.get("recent_dot_pct", 0.6))
    recent_boundaries = int(state.get("recent_boundaries", 0))

    bat = lookup_career(state.get("batter"), career, TEST_DEFAULT_CAREER)
    bowl = lookup_career(state.get("bowler"), career, TEST_DEFAULT_CAREER)

    return {
        "innings_num": innings_num,
        "day_of_match": day_of_match,
        "over": over_num,
        "ball_in_innings": ball_in_innings,
        "ball_in_match": ball_in_match,
        "balls_remaining_in_match": balls_remaining_in_match,
        "balls_remaining_today": balls_remaining_today,
        "days_remaining": days_remaining,
        "is_final_session": is_final_session,
        "innings_runs": innings_runs,
        "innings_wickets": innings_wickets,
        "innings_run_rate": round(innings_run_rate, 3),
        "current_lead": current_lead,
        "partnership_runs": partnership_runs,
        "partnership_balls": partnership_balls,
        "recent_dot_pct": recent_dot_pct,
        "recent_run_rate": round(recent_rr, 3),
        "recent_boundaries": recent_boundaries,
        "balls_since_wicket": balls_since_wicket,
        **bat,
        **bowl,
        "is_team1_batting": is_team1_batting,
    }


def predict_test(features: dict, model: xgb.Booster) -> tuple[float, float, float]:
    """Returns (P(team1_win), P(team2_win), P(draw))."""
    row = [features[c] for c in TEST_FEATURE_COLUMNS]
    d = xgb.DMatrix([row], feature_names=TEST_FEATURE_COLUMNS)
    probs = model.predict(d)[0]
    # multi:softprob output order matches the label order in training
    return float(probs[0]), float(probs[1]), float(probs[2])


# ----------------------------- UI -----------------------------

st.set_page_config(page_title="Cricket Win Probability", page_icon="🏏", layout="wide")
st.title("🏏 Cricket Win Probability")
st.caption("Ball-by-ball win prediction for Test and ODI cricket, trained on Cricsheet")

# Sidebar
format_choice = st.sidebar.radio("Format", ["Test", "ODI"])
mode = st.sidebar.radio("Mode", ["Replay famous match", "Manual entry", "Live (API)"])

# Format-specific loading
if format_choice == "Test":
    model = load_test_model()
    career = load_test_career()
    famous = load_famous_matches("famous_matches")
    is_test = True
else:
    model = load_odi_model()
    career = load_odi_career()
    famous = load_famous_matches("famous_matches_odi")
    is_test = False

if model is None:
    st.error(
        f"No trained {format_choice} model found in `data/`. "
        f"Run the {format_choice} training pipeline first."
    )
    st.stop()

# Sidebar stats
st.sidebar.markdown("---")
if is_test:
    st.sidebar.markdown(
        "**Test model**  \n"
        "3-class: team1_win / team2_win / draw  \n"
        "Calibration error: 0.027  \n"
        "Final-ball accuracy: 87.4%"
    )
else:
    st.sidebar.markdown(
        "**ODI model**  \n"
        "Binary: team1_win / team2_win  \n"
        "Calibration error: 0.014  \n"
        "Final-ball accuracy: 97.6%"
    )

# ----------------------------- Replay mode -----------------------------

if mode == "Replay famous match":
    if not famous:
        # Diagnostic: are there CSVs in the folder that just lack the columns we need?
        folder = DATA_DIR / ("famous_matches" if is_test else "famous_matches_odi")
        existing = sorted(folder.glob("*_curve.csv")) if folder.exists() else []
        if existing:
            # Show what columns the first one has so we can debug
            sample_df = pd.read_csv(existing[0], nrows=1)
            st.error(
                f"Found {len(existing)} CSV(s) in `{folder}/` but none have the columns "
                "the chart needs (`prob_team1_win`, `prob_team2_win`, `ball_in_match`)."
            )
            st.markdown(f"**Columns found in `{existing[0].name}`:**")
            st.code(", ".join(sample_df.columns.tolist()))
            st.markdown(
                "Tell Claude what columns are listed above so the loader can be adapted, "
                "or regenerate using the current test script."
            )
        else:
            if is_test:
                st.error(
                    "No CSVs found in `data/famous_matches/`.  \n"
                    "Run: `python3 06_test_model.py`"
                )
            else:
                st.error(
                    "No CSVs found in `data/famous_matches_odi/`.  \n"
                    "Run: `python3 06_test_model_odi.py`"
                )
    else:
        choice = st.selectbox("Pick a match", list(famous.keys()))
        df = famous[choice]
        t1, t2 = _resolve_teams(df)
        outcome = _resolve_outcome(df)
        if outcome == "team1_win":
            result_str = f"{t1} won"
        elif outcome == "team2_win":
            result_str = f"{t2} won"
        else:
            result_str = "draw"
        st.markdown(f"### {t1} vs {t2} — final outcome: **{result_str}**")

        # Chart: 3 lines for Test (with draw), 2 for ODI
        value_cols = ["prob_team1_win", "prob_team2_win"]
        team_map = {"prob_team1_win": t1, "prob_team2_win": t2}
        if is_test and "prob_draw" in df.columns:
            value_cols.append("prob_draw")
            team_map["prob_draw"] = "Draw"

        plot_df = df[["ball_in_match"] + value_cols].copy()
        plot_df = plot_df.melt(
            id_vars=["ball_in_match"],
            value_vars=value_cols,
            var_name="team",
            value_name="prob",
        )
        plot_df["team"] = plot_df["team"].map(team_map)

        # Smooth ball-by-ball noise with a 30-ball rolling mean. Underlying data
        # is preserved in `prob`; we plot the smoothed version only.
        plot_df["prob_smoothed"] = (
            plot_df.groupby("team")["prob"]
            .transform(lambda s: s.rolling(window=30, min_periods=1).mean())
        )

        chart = alt.Chart(plot_df).mark_line(strokeWidth=2).encode(
            x=alt.X("ball_in_match:Q", title="Ball in match"),
            y=alt.Y("prob_smoothed:Q", title="Win probability (30-ball smoothed)",
                    scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("team:N", title="Outcome"),
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)

        # Snapshots
        st.markdown("### Key moments")
        snapshots = [
            ("Match start", 0),
            ("Quarter way", len(df) // 4),
            ("Half way", len(df) // 2),
            ("Three-quarter", 3 * len(df) // 4),
            ("Final ball", -1),
        ]
        cols = st.columns(len(snapshots))
        for col, (label, idx) in zip(cols, snapshots):
            row = df.iloc[idx]
            with col:
                if is_test and "prob_draw" in df.columns:
                    st.metric(
                        label,
                        f"{t1[:8]}: {row['prob_team1_win']*100:.0f}%",
                        f"{t2[:8]}: {row['prob_team2_win']*100:.0f}% | D: {row['prob_draw']*100:.0f}%",
                        delta_color="off",
                    )
                else:
                    st.metric(
                        label,
                        f"{t1[:10]}: {row['prob_team1_win']*100:.0f}%",
                        f"{t2[:10]}: {row['prob_team2_win']*100:.0f}%",
                        delta_color="off",
                    )

# ----------------------------- Manual mode -----------------------------

elif mode == "Manual entry":
    st.markdown("Enter the current match state. Probabilities update on submit.")

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.text_input("Team 1 (batted first)", "England" if is_test else "India")
        team2 = st.text_input("Team 2", "Australia" if is_test else "Afghanistan")
        if is_test:
            innings = st.selectbox("Innings", [1, 2, 3, 4])
        else:
            innings = st.radio("Innings", [1, 2], horizontal=True)
        batting = st.selectbox("Batting team", [team1, team2])
    with col2:
        score = st.number_input("Current innings score", min_value=0, max_value=700, value=120, step=1)
        wickets = st.number_input("Current innings wickets", min_value=0, max_value=10, value=3, step=1)
        over = st.text_input("Current over in this innings (e.g. 24.3)", "24.3")

    if is_test:
        st.markdown("**Test-specific state:**")
        col3, col4 = st.columns(2)
        with col3:
            ball_in_match = st.number_input(
                "Total balls bowled in match so far (across all innings)",
                min_value=1, max_value=2700,
                value=parse_over(over),
                step=1,
                help="Roughly 540 per day. For day N use ~540 × (N-1) + balls bowled today."
            )
            team1_total = st.number_input(
                f"{team1} total runs so far (across all innings)",
                min_value=0, max_value=2000, value=int(score) if batting == team1 else 250, step=1
            )
        with col4:
            team2_total = st.number_input(
                f"{team2} total runs so far (across all innings)",
                min_value=0, max_value=2000, value=int(score) if batting == team2 else 250, step=1
            )
    else:
        target = st.number_input("Target (only if innings 2)", min_value=0, max_value=600,
                                  value=0, step=1)

    with st.expander("Optional details (better predictions)"):
        batter = st.text_input("Striker name", "")
        bowler = st.text_input("Bowler name", "")
        partnership = st.number_input("Partnership runs", min_value=0, value=0, step=1)
        if is_test:
            last_window = st.number_input("Runs in last 10 overs", min_value=0, value=0, step=1)
        else:
            last_window = st.number_input("Runs in last 6 overs", min_value=0, value=0, step=1)

    if st.button("Predict", type="primary"):
        state = {
            "team1": team1, "team2": team2,
            "innings_num": int(innings),
            "batting_team": batting,
            "current_score": int(score),
            "current_wickets": int(wickets),
            "current_over": over,
        }
        if batter: state["batter"] = batter
        if bowler: state["bowler"] = bowler
        if partnership: state["partnership_runs"] = int(partnership)

        try:
            if is_test:
                state["ball_in_match"] = int(ball_in_match)
                state["team1_total_runs"] = int(team1_total)
                state["team2_total_runs"] = int(team2_total)
                if last_window:
                    state["last_10_overs_runs"] = int(last_window)
                features = compute_test_features(state, career)
                p_t1, p_t2, p_draw = predict_test(features, model)

                st.markdown("### Prediction")
                col1, col2, col3 = st.columns(3)
                col1.metric(team1, f"{p_t1*100:.1f}%")
                col2.metric(team2, f"{p_t2*100:.1f}%")
                col3.metric("Draw", f"{p_draw*100:.1f}%")
                st.progress(p_t1, text=f"{team1}: {p_t1*100:.1f}%")
                st.progress(p_t2, text=f"{team2}: {p_t2*100:.1f}%")
                st.progress(p_draw, text=f"Draw: {p_draw*100:.1f}%")
            else:
                state["target"] = int(target) if innings == 2 else 0
                if last_window:
                    state["last_6_overs_runs"] = int(last_window)
                features = compute_odi_features(state, career)
                p_t1, p_t2 = predict_odi(features, model)

                st.markdown("### Prediction")
                col1, col2 = st.columns(2)
                col1.metric(team1, f"{p_t1*100:.1f}%")
                col2.metric(team2, f"{p_t2*100:.1f}%")
                st.progress(p_t1, text=f"{team1}: {p_t1*100:.1f}%")
                st.progress(p_t2, text=f"{team2}: {p_t2*100:.1f}%")

            with st.expander("Feature values fed to model"):
                st.json(features)
        except (ValueError, KeyError) as e:
            st.error(f"Input problem: {e}")

# ----------------------------- Live mode (placeholder) -----------------------------

elif mode == "Live (API)":
    st.info(
        f"Live mode for {format_choice} cricket. Connects to a cricket data API. "
        "Get a free key at cricketdata.org, paste in sidebar."
    )
    api_key = st.sidebar.text_input("CricketData.org API key", type="password")
    if not api_key:
        st.warning("Enter an API key in the sidebar to enable live mode.")
    else:
        st.info("Live mode wiring is the next step. For now use Manual or Replay mode.")
        # TODO: fetch /currentMatches filtered by format
        # TODO: fetch /match_info for selected
        # TODO: map JSON to state dict (Test or ODI shape per format)
        # TODO: predict + display with auto-refresh
