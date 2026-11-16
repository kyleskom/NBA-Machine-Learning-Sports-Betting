"""Team stats from DunkAPI, a drop-in for stats.nba.com's leaguedashteamstats (Base, PerGame).

stats.nba.com stopped answering leaguedashteamstats requests (issue #516). DunkAPI's get_team_stats returns the same
per-game numbers for every team, as they stood on a given date, so the stored tables and the trained models keep working.

Set DUNKAPI_KEY to use it. A free key (500 credits a month, no card) is at https://dunkapi.com/signup; each call costs
5 credits, so a daily update fits the free tier. Backfilling a whole season is about 240 calls.
"""
import os
from datetime import date, timedelta

import pandas as pd
import requests

API_URL = "https://mcp.dunkapi.com/v1/get_team_stats"

# NBA team ids for the TEAM_ID column, by current team name.
NBA_TEAM_IDS = {
    "Atlanta Hawks": 1610612737, "Boston Celtics": 1610612738, "Brooklyn Nets": 1610612751,
    "Charlotte Hornets": 1610612766, "Chicago Bulls": 1610612741, "Cleveland Cavaliers": 1610612739,
    "Dallas Mavericks": 1610612742, "Denver Nuggets": 1610612743, "Detroit Pistons": 1610612765,
    "Golden State Warriors": 1610612744, "Houston Rockets": 1610612745, "Indiana Pacers": 1610612754,
    "LA Clippers": 1610612746, "Los Angeles Lakers": 1610612747, "Memphis Grizzlies": 1610612763,
    "Miami Heat": 1610612748, "Milwaukee Bucks": 1610612749, "Minnesota Timberwolves": 1610612750,
    "New Orleans Pelicans": 1610612740, "New York Knicks": 1610612752, "Oklahoma City Thunder": 1610612760,
    "Orlando Magic": 1610612753, "Philadelphia 76ers": 1610612755, "Phoenix Suns": 1610612756,
    "Portland Trail Blazers": 1610612757, "Sacramento Kings": 1610612758, "San Antonio Spurs": 1610612759,
    "Toronto Raptors": 1610612761, "Utah Jazz": 1610612762, "Washington Wizards": 1610612764,
}

STATS = ["GP", "W", "L", "W_PCT", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
         "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "BLKA", "PF", "PFD", "PTS", "PLUS_MINUS"]
# Ranked like stats.nba.com: 1 is best, ties share the lower rank; for these fewer is better.
RANK_ASCENDING = {"L", "TOV", "BLKA", "PF"}


def api_key():
    return os.environ.get("DUNKAPI_KEY")


def season_end_year(season_key):
    """'2025-26' -> 2026 (DunkAPI names a season by the year it ends)."""
    return int(season_key[:4]) + 1


def get_team_stats(through, season_key, key=None):
    """Per-game team stats for games played through `through` (inclusive, like stats.nba.com's DateTo),
    as a DataFrame with the leaguedashteamstats Base columns. Empty when there is nothing to return."""
    key = key or api_key()
    if not key:
        raise RuntimeError("Set DUNKAPI_KEY (free key at https://dunkapi.com/signup).")
    body = {"season": season_end_year(season_key), "season_type": "regular",
            "before": (through + timedelta(days=1)).isoformat()}
    res = requests.post(API_URL, json=body, headers={"x-api-key": key}, timeout=30)
    if res.status_code == 402:
        raise RuntimeError("DunkAPI: out of credits. " + res.json().get("message", ""))
    res.raise_for_status()
    teams = res.json()["data"]["teams"]
    return to_frame(teams)


def to_frame(teams):
    if not teams:
        return pd.DataFrame(data={})
    rows = []
    for t in teams:
        gp = t["games"]
        rows.append({
            "TEAM_ID": NBA_TEAM_IDS.get(t["display_name"]),
            "TEAM_NAME": t["display_name"],
            "GP": gp, "W": t["wins"], "L": t["losses"],
            "W_PCT": round(t["wins"] / gp, 3) if gp else 0.0,
            "MIN": round(48 + 5 * (t.get("overtime_periods") or 0), 1),
            "FGM": t["fgm"], "FGA": t["fga"], "FG_PCT": t["fg_pct"],
            "FG3M": t["tpm"], "FG3A": t["tpa"], "FG3_PCT": t["tp_pct"],
            "FTM": t["ftm"], "FTA": t["fta"], "FT_PCT": t["ft_pct"],
            "OREB": t["oreb"], "DREB": t["dreb"], "REB": t["reb"], "AST": t["ast"], "TOV": t["tov"],
            "STL": t["stl"], "BLK": t["blk"], "BLKA": t["blka"], "PF": t["pf"], "PFD": t["pfd"],
            "PTS": t["pts"], "PLUS_MINUS": t["plus_minus"],
        })
    df = pd.DataFrame(rows).sort_values("TEAM_NAME").reset_index(drop=True)
    for col in STATS:
        df[f"{col}_RANK"] = df[col].rank(method="min", ascending=col in RANK_ASCENDING).astype(int)
    return df


if __name__ == "__main__":
    print(get_team_stats(date.today() - timedelta(days=1), "2025-26").head())
