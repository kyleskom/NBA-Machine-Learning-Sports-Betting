"""Import attributed consensus odds for seasons without an SBR archive."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sqlite3
import sys

import pandas as pd
import requests
import toml

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from src.Players.identity import team_id
from src.Storage.normalized import connect as connect_canonical, save_odds_rows

CONFIG_PATH = BASE_DIR / "config.toml"
PLAYER_DB_PATH = BASE_DIR / "Data" / "PlayerData.sqlite"
CANONICAL_DB_PATH = BASE_DIR / "Data" / "NBAData.sqlite"
DEFAULT_CSV_PATH = BASE_DIR / "Data" / "nba-odds-2026-27-consensus.csv"
SOURCE_URL = (
    "https://huggingface.co/datasets/minmaxxxer/nba-odds-2026-27/"
    "resolve/main/nba_2026_27_odds.csv"
)
SOURCE_ID = "huggingface:minmaxxxer/nba-odds-2026-27"
SPORTSBOOK = "market-consensus"
LICENSE = "CC-BY-4.0"

REQUIRED_COLUMNS = {
    "date_et", "season", "home_team", "away_team", "home_score",
    "away_score", "close_spread_home", "close_total",
}


def download_source(path=DEFAULT_CSV_PATH):
    response = requests.get(SOURCE_URL, timeout=45)
    response.raise_for_status()
    if not response.content.startswith(b"game_id,date_et,start_time_utc"):
        raise ValueError("Consensus odds download did not match the expected CSV format")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return path


def validate_scores(frame, player_db_path=PLAYER_DB_PATH):
    with sqlite3.connect(f"file:{Path(player_db_path).resolve()}?mode=ro", uri=True) as con:
        logs = pd.read_sql_query(
            """SELECT GAME_DATE, TEAM_ID, SUM(PTS) AS points
               FROM player_game_logs
               WHERE SEASON='2026-27' AND SEASON_TYPE='Regular Season'
               GROUP BY GAME_DATE, TEAM_ID""",
            con,
        )
    observed = {
        (str(row.GAME_DATE), int(row.TEAM_ID)): int(row.points)
        for row in logs.itertuples(index=False)
    }
    for row in frame.itertuples(index=False):
        day = pd.Timestamp(row.date_et).date().isoformat()
        for team, score in ((row.home_team, row.home_score), (row.away_team, row.away_score)):
            key = (day, team_id(team))
            if key not in observed or observed[key] != int(score):
                raise ValueError(
                    f"Score check failed for {day} {team}: source={score}, "
                    f"player-log total={observed.get(key)}"
                )


def build_rows(frame):
    last_played = {}
    rows = []
    games = frame.copy()
    games["game_day"] = pd.to_datetime(games["date_et"], errors="raise").dt.date
    if games.duplicated(["game_day", "home_team", "away_team"]).any():
        raise ValueError("Consensus CSV contains duplicate games")
    games = games.sort_values(["game_day", "start_time_utc"], kind="stable")
    for row in games.itertuples(index=False):
        day = row.game_day.isoformat()
        home_id, away_id = team_id(row.home_team), team_id(row.away_team)
        home_rest = (row.game_day - last_played[home_id]).days if home_id in last_played else 7
        away_rest = (row.game_day - last_played[away_id]).days if away_id in last_played else 7
        last_played[home_id] = row.game_day
        last_played[away_id] = row.game_day
        rows.append({
            "Date": day,
            "Home": row.home_team,
            "Away": row.away_team,
            "OU": float(row.close_total),
            # Canonical spread storage uses the away team's handicap.
            "Spread": -float(row.close_spread_home),
            # The public sample publishes no-vig probabilities, not executable
            # American prices. Leave payout prices blank instead of fabricating them.
            "ML_Home": None,
            "ML_Away": None,
            "Points": int(row.home_score) + int(row.away_score),
            "Win_Margin": int(row.home_score) - int(row.away_score),
            "Days_Rest_Home": home_rest,
            "Days_Rest_Away": away_rest,
        })
    return rows


def import_consensus(frame, csv_path=DEFAULT_CSV_PATH):
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Consensus CSV is missing required columns: {sorted(missing)}")
    if set(frame["season"].astype(str)) != {"2026-27"}:
        raise ValueError("This importer only accepts the 2026-27 regular season")
    if set(frame.get("season_type", pd.Series(dtype=str)).astype(str)) != {"regular"}:
        raise ValueError("Consensus input contains non-regular-season rows")
    validate_scores(frame)
    rows = build_rows(frame)
    with connect_canonical(CANONICAL_DB_PATH) as con:
        saved = save_odds_rows(
            con, rows, "2026-27", SPORTSBOOK, source=SOURCE_ID
        )
        # The source exposes fair probabilities, not wager payout prices.
        con.execute(
            "DELETE FROM odds_quotes WHERE sportsbook=? AND market='moneyline' AND source=?",
            (SPORTSBOOK, SOURCE_ID),
        )

    metadata = {
        "dataset": SOURCE_ID,
        "source_url": SOURCE_URL,
        "license": LICENSE,
        "accessed_at": datetime.now(timezone.utc).isoformat(),
        "rows_imported": saved,
        "fields_used": ["date_et", "home_team", "away_team", "home_score", "away_score", "close_spread_home", "close_total"],
        "odds_note": "Consensus closing total/spread only; no individual sportsbook payout prices were supplied.",
        "attribution": "minmaxxxer, NBA 2026-27 Odds and Results, CC BY 4.0",
    }
    Path(csv_path).with_suffix(".source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Imported {saved} 2026-27 consensus games from {Path(csv_path).name}.")
    return 0


def main(season="2026-27", csv_path=DEFAULT_CSV_PATH):
    config = toml.load(CONFIG_PATH)
    settings = config.get("create-games", {}).get(season, {})
    if settings.get("odds_sportsbook") != SPORTSBOOK:
        return 0
    path = download_source(csv_path)
    frame = pd.read_csv(io.BytesIO(path.read_bytes()))
    return import_consensus(frame, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    args = parser.parse_args()
    raise SystemExit(main(args.season, args.csv))
