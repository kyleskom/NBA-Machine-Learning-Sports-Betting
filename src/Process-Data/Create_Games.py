import argparse
import sqlite3
from datetime import datetime
import os
import sys
sys.path.insert(0, os.fspath(__import__("pathlib").Path(__file__).resolve().parents[2]))
from pathlib import Path

import numpy as np
import pandas as pd
import toml

from src.Players.identity import team_id
from src.Storage.normalized import (canonical_path_for, connect as connect_canonical,
                                    load_games as load_normalized_games,
                                    load_latest_team_snapshot)
from src.Utils.Dictionaries import (
    team_index_07,
    team_index_08,
    team_index_12,
    team_index_13,
    team_index_14,
    team_index_current,
)

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config.toml"
ODDS_DB_PATH = BASE_DIR / "Data" / "OddsData.sqlite"
TEAMS_DB_PATH = BASE_DIR / "Data" / "TeamData.sqlite"
OUTPUT_DB_PATH = BASE_DIR / "Data" / "dataset.sqlite"
OUTPUT_TABLE = "dataset_2012-26_team"

TEAM_INDEX_BY_SEASON = {
    "2007-08": team_index_07,
    "2008-09": team_index_08,
    "2009-10": team_index_08,
    "2010-11": team_index_08,
    "2011-12": team_index_08,
    "2012-13": team_index_12,
    "2013-14": team_index_13,
    "2014-15": team_index_14,
    "2015-16": team_index_14,
    "2016-17": team_index_14,
    "2017-18": team_index_14,
    "2018-19": team_index_14,
    "2019-20": team_index_14,
    "2020-21": team_index_14,
    "2021-22": team_index_14,
    "2022-23": team_index_current,
    "2023-24": team_index_current,
    "2024-25": team_index_current,
    "2025-26": team_index_current,
    "2026-27": team_index_current,
}

def table_exists(con, table_name):
    cursor = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def normalize_date(value):
    if isinstance(value, datetime):
        return value.date().isoformat()
    if hasattr(value, "date"):
        try:
            return value.date().isoformat()
        except Exception:
            pass
    return str(value)


def get_team_index_map(season_key):
    if season_key in TEAM_INDEX_BY_SEASON:
        return TEAM_INDEX_BY_SEASON[season_key]
    try:
        start_year = int(season_key.split("-")[0])
    except (ValueError, IndexError):
        return team_index_current
    return team_index_current if start_year >= 2022 else team_index_14


def fetch_team_table(teams_con, date_str, canonical_con=None):
    # DateTo snapshots contain that day's results. Only an earlier snapshot is safe.
    if canonical_con is not None:
        snapshot = load_latest_team_snapshot(canonical_con, date_str)
        if snapshot is not None and not snapshot.empty:
            return snapshot
    candidates = [r[0] for r in teams_con.execute("SELECT name FROM sqlite_master WHERE type='table'")
                  if len(r[0]) == 10 and r[0] < date_str and r[0][:4].isdigit()]
    if not candidates:
        return None
    previous = max(candidates)
    if (datetime.fromisoformat(date_str)-datetime.fromisoformat(previous)).days > 7:
        return None
    return pd.read_sql_query(f'SELECT * FROM "{previous}"', teams_con)


def build_game_features(team_df, home_team, away_team, index_map=None):
    home_id, away_id = team_id(home_team), team_id(away_team)
    home = team_df[team_df.TEAM_ID.astype(int) == home_id]
    away = team_df[team_df.TEAM_ID.astype(int) == away_id]
    if len(home) != 1 or len(away) != 1:
        return None
    return pd.concat([home.iloc[0], away.iloc[0].rename(index={c:f"{c}.1" for c in team_df.columns})])


def select_odds_table(odds_con, season_key):
    candidates = [
        f"odds_{season_key}_new",
        f"odds_{season_key}",
        f"{season_key}_new",
        f"{season_key}",
    ]
    for table_name in candidates:
        if table_exists(odds_con, table_name):
            return table_name
    return None


def main(output_table=None):
    if output_table is None:
        output_table = OUTPUT_TABLE
    config = toml.load(CONFIG_PATH)
    scores = []
    win_margin = []
    ou_values = []
    ou_cover = []
    games = []
    days_rest_away = []
    days_rest_home = []

    canonical_path = canonical_path_for(ODDS_DB_PATH)
    with sqlite3.connect(ODDS_DB_PATH) as odds_con, sqlite3.connect(TEAMS_DB_PATH) as teams_con, connect_canonical(canonical_path) as canonical_con:
        team_cache = {}
        for season_key, season_settings in config["create-games"].items():
            print(season_key)
            odds_sportsbook = season_settings.get("odds_sportsbook", "fanduel")
            odds_df = load_normalized_games(
                canonical_con, season_key, sportsbook=odds_sportsbook
            )
            if odds_df.empty:
                odds_table = select_odds_table(odds_con, season_key)
                if not odds_table:
                    print(f"Missing odds data for {season_key}.")
                    continue
                odds_df = pd.read_sql_query(f'SELECT * FROM "{odds_table}"', odds_con)
            if odds_df.empty:
                print(f"No odds data for {season_key}.")
                continue

            index_map = get_team_index_map(season_key)

            for row in odds_df.itertuples(index=False):
                date_str = normalize_date(row.Date)
                if date_str not in team_cache:
                    team_cache[date_str] = fetch_team_table(teams_con, date_str, canonical_con)
                team_df = team_cache[date_str]
                if team_df is None:
                    continue

                game = build_game_features(team_df, row.Home, row.Away, index_map)
                if game is None:
                    continue

                game["Date"] = date_str
                game["Date.1"] = date_str
                scores.append(row.Points)
                ou_values.append(row.OU)
                days_rest_home.append(row.Days_Rest_Home)
                days_rest_away.append(row.Days_Rest_Away)
                win_margin.append(1 if row.Win_Margin > 0 else 0)

                if row.Points < row.OU:
                    ou_cover.append(0)
                elif row.Points > row.OU:
                    ou_cover.append(1)
                else:
                    ou_cover.append(2)

                games.append(game)


    if not games:
        print("No game rows produced. Check odds and team tables.")
        return

    season = pd.concat(games, ignore_index=True, axis=1).T
    frame = season.drop(columns=["TEAM_ID", "TEAM_ID.1"], errors="ignore")
    frame["Score"] = np.asarray(scores)
    frame["Home-Team-Win"] = np.asarray(win_margin)
    frame["OU"] = np.asarray(ou_values)
    frame["OU-Cover"] = np.asarray(ou_cover)
    frame["Days-Rest-Home"] = np.asarray(days_rest_home)
    frame["Days-Rest-Away"] = np.asarray(days_rest_away)
    # Concatenating mixed team/name Series leaves numeric columns as object.
    # Infer their numeric dtype so SQLite stores numbers instead of numpy blobs.
    frame = frame.infer_objects()

    with sqlite3.connect(OUTPUT_DB_PATH) as con:
        frame.to_sql(output_table, con, if_exists="replace", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build training game-level dataset.")
    parser.add_argument(
        "--output-table",
        default=OUTPUT_TABLE,
        help=f"Destination table name in {OUTPUT_DB_PATH.name}.",
    )
    args = parser.parse_args()
    main(output_table=args.output_table)
