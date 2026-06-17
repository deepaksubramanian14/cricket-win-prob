"""CricketData.org → state dict for win probability models.

Two endpoints:
  GET /currentMatches           → list of live/recent matches
  GET /match_info?id=<id>       → full state for one match

match_info `score` array has one entry per innings:
  {"r": runs, "w": wickets, "o": overs_decimal, "inning": "<Team> Inning <N>"}
Overs are cricket notation: "17.4" means 17 overs 4 balls = 106 balls.

The state dicts returned here match the shape that app.py's
compute_test_features / compute_odi_features consume, so app.py can
reuse the existing feature pipeline unchanged.
"""

from typing import Optional
import requests

CRICKETDATA_BASE = "https://api.cricapi.com/v1"


def _parse_innings_team(inning_str: str, teams: list) -> Optional[str]:
    """'Australia Inning 1' -> 'Australia'."""
    for t in teams:
        if inning_str.startswith(t):
            return t
    return None


def overs_decimal_to_balls(overs: float) -> int:
    """Cricket overs notation -> total balls. 17.4 -> 17*6 + 4 = 106."""
    full = int(overs)
    part = round((overs - full) * 10)
    return full * 6 + part


# ---------- API calls ----------

def fetch_current_matches(api_key: str, offset: int = 0) -> list:
    r = requests.get(
        f"{CRICKETDATA_BASE}/currentMatches",
        params={"apikey": api_key, "offset": offset},
        timeout=10,
    )
    r.raise_for_status()
    return r.json().get("data", [])


def fetch_match_info(api_key: str, match_id: str) -> dict:
    r = requests.get(
        f"{CRICKETDATA_BASE}/match_info",
        params={"apikey": api_key, "id": match_id},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


# ---------- API response -> Manual-mode state dict ----------

def api_to_odi_state(api_response: dict) -> Optional[dict]:
    """match_info -> state dict for compute_odi_features.

    Returns None if match has no score data yet (e.g. fixture).
    """
    data = api_response.get("data", {})
    if data.get("matchType") != "odi":
        raise ValueError(f"Not an ODI: matchType={data.get('matchType')}")

    teams = data.get("teams", [])
    if len(teams) != 2:
        return None
    team1, team2 = teams

    scores = data.get("score", [])
    if not scores:
        return None

    current = scores[-1]
    batting_team = _parse_innings_team(current["inning"], teams)
    if batting_team is None:
        # Defensive: if team name doesn't match, fall back to team1
        batting_team = team1

    innings_num = len(scores)
    current_runs = current.get("r", 0)
    current_wickets = current.get("w", 0)
    current_overs_decimal = current.get("o", 0)

    # Convert "17.4" decimal back to text form "17.4" for parse_over in app.py
    current_over_str = _overs_to_text(current_overs_decimal)

    # Target: if chasing, first innings total + 1
    target = 0
    balls_in_innings_1 = 300  # default full 50 overs
    if innings_num == 2:
        first_innings = scores[0]
        target = first_innings.get("r", 0) + 1
        balls_in_innings_1 = overs_decimal_to_balls(first_innings.get("o", 50))

    return {
        "team1": team1,
        "team2": team2,
        "innings_num": innings_num,
        "batting_team": batting_team,
        "current_score": current_runs,
        "current_wickets": current_wickets,
        "current_over": current_over_str,
        "target": target,
        "balls_in_innings_1": balls_in_innings_1,
        # Display-only extras
        "_status": data.get("status", ""),
        "_match_name": data.get("name", ""),
        "_venue": data.get("venue", ""),
    }


def api_to_test_state(api_response: dict) -> Optional[dict]:
    """match_info -> state dict for compute_test_features.

    Returns None if match has no score data yet.
    """
    data = api_response.get("data", {})
    if data.get("matchType") != "test":
        raise ValueError(f"Not a Test: matchType={data.get('matchType')}")

    teams = data.get("teams", [])
    if len(teams) != 2:
        return None
    team1, team2 = teams

    scores = data.get("score", [])
    if not scores:
        return None

    current = scores[-1]
    batting_team = _parse_innings_team(current["inning"], teams)
    if batting_team is None:
        batting_team = team1

    innings_num = len(scores)
    current_runs = current.get("r", 0)
    current_wickets = current.get("w", 0)
    current_overs_decimal = current.get("o", 0)
    current_over_str = _overs_to_text(current_overs_decimal)

    # ball_in_match accumulates across all innings
    ball_in_match = sum(overs_decimal_to_balls(s.get("o", 0)) for s in scores)
    # At least 1 (compute_test_features divides into it)
    ball_in_match = max(1, ball_in_match)

    team1_total = sum(s.get("r", 0) for s in scores
                      if _parse_innings_team(s["inning"], teams) == team1)
    team2_total = sum(s.get("r", 0) for s in scores
                      if _parse_innings_team(s["inning"], teams) == team2)

    return {
        "team1": team1,
        "team2": team2,
        "innings_num": innings_num,
        "batting_team": batting_team,
        "current_score": current_runs,
        "current_wickets": current_wickets,
        "current_over": current_over_str,
        "ball_in_match": ball_in_match,
        "team1_total_runs": team1_total,
        "team2_total_runs": team2_total,
        # Display-only extras
        "_status": data.get("status", ""),
        "_match_name": data.get("name", ""),
        "_venue": data.get("venue", ""),
    }


def _overs_to_text(overs_decimal: float) -> str:
    """7.4 -> '7.4'. Handles edge cases like 7.0 -> '7' (parse_over handles both)."""
    if overs_decimal == int(overs_decimal):
        return str(int(overs_decimal))
    return str(overs_decimal)
