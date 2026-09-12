import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import toml

from src.Utils import PlayerContext
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
PLAYER_DB_PATH = BASE_DIR / "Data" / "PlayerData.sqlite"
OUTPUT_DB_PATH = BASE_DIR / "Data" / "dataset.sqlite"
PLAYER_FEATURES_TABLE = PlayerContext.PLAYER_FEATURES_TABLE
OUTPUT_TABLE = "dataset_2012-26_player_v1"

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
}

DEFAULT_PLAYER_WINDOWS = [7, 14, 30]
PLAYER_ROLL_METRICS = ("MIN", "PTS", "REB", "AST", "USG_WEIGHTED")
PLAYER_BASE_METRICS = [
    "PLR_MIN_SUM",
    "PLR_PTS_SUM",
    "PLR_REB_SUM",
    "PLR_AST_SUM",
    "PLR_USG_WEIGHTED_SUM",
]
PLAYER_INJURY_METRICS = [
    "INJ_OUT_MIN_WT",
    "INJ_Q_MIN_WT",
    "INJ_DOUBTFUL_MIN_WT",
    "INJ_AVAIL_MIN_WT",
    "INJ_COVERAGE",
]


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


def fetch_team_table(teams_con, date_str):
    if not table_exists(teams_con, date_str):
        return None
    return pd.read_sql_query(f'SELECT * FROM "{date_str}"', teams_con)


def build_game_features(team_df, home_team, away_team, index_map):
    home_index = index_map.get(home_team)
    away_index = index_map.get(away_team)
    if home_index is None or away_index is None:
        return None
    if len(team_df.index) != 30:
        return None

    home_team_series = team_df.iloc[home_index]
    away_team_series = team_df.iloc[away_index]
    return pd.concat([
        home_team_series,
        away_team_series.rename(index={col: f"{col}.1" for col in team_df.columns.values}),
    ])


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


def get_player_feature_columns(config):
    return PlayerContext.get_player_feature_columns(config)


def get_player_defaults(config, feature_cols):
    return PlayerContext.get_player_defaults(config, feature_cols)


def load_player_feature_map(player_con, table_name, feature_cols):
    return PlayerContext.load_player_feature_map(player_con, table_name, feature_cols)


def build_player_feature_row(feature_map, date_str, home_team, away_team, feature_cols, default_values):
    if "exact" not in feature_map:
        feature_map = {"exact": feature_map, "history": {}}
    return PlayerContext.build_player_feature_row(
        feature_map,
        date_str,
        home_team,
        away_team,
        feature_cols,
        default_values,
        allow_asof=False,
    )


def main(output_table=None, include_player_features=True):
    if output_table is None:
        output_table = OUTPUT_TABLE
    config = toml.load(CONFIG_PATH)
    player_feature_cols = get_player_feature_columns(config)
    player_defaults = get_player_defaults(config, player_feature_cols)

    scores = []
    win_margin = []
    ou_values = []
    ou_cover = []
    games = []
    days_rest_away = []
    days_rest_home = []
    player_rows = []

    with sqlite3.connect(ODDS_DB_PATH) as odds_con, sqlite3.connect(TEAMS_DB_PATH) as teams_con:
        player_feature_map = {"exact": {}, "history": {}}
        player_con = None
        if include_player_features and PLAYER_DB_PATH.exists():
            player_con = sqlite3.connect(PLAYER_DB_PATH)
            player_feature_map = PlayerContext.load_player_feature_lookup(
                player_con,
                PLAYER_FEATURES_TABLE,
                player_feature_cols,
            )

        for season_key in config["create-games"].keys():
            print(season_key)
            odds_table = select_odds_table(odds_con, season_key)
            if not odds_table:
                print(f"Missing odds tables for {season_key}.")
                continue

            odds_df = pd.read_sql_query(f'SELECT * FROM "{odds_table}"', odds_con)
            if odds_df.empty:
                print(f"No odds data for {season_key}.")
                continue

            index_map = get_team_index_map(season_key)

            for row in odds_df.itertuples(index=False):
                date_str = normalize_date(row.Date)
                team_df = fetch_team_table(teams_con, date_str)
                if team_df is None:
                    continue

                game = build_game_features(team_df, row.Home, row.Away, index_map)
                if game is None:
                    continue

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

                if include_player_features:
                    player_rows.append(
                        build_player_feature_row(
                            player_feature_map,
                            date_str,
                            row.Home,
                            row.Away,
                            player_feature_cols,
                            player_defaults,
                        )
                    )
                games.append(game)

        if player_con is not None:
            player_con.close()

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

    if include_player_features:
        if not player_rows:
            player_rows = [dict(player_defaults) for _ in range(len(frame.index))]
        player_frame = pd.DataFrame(player_rows)
        frame = pd.concat([frame.reset_index(drop=True), player_frame.reset_index(drop=True)], axis=1)

    for field in frame.columns.values:
        if "TEAM_" in field or "Date" in field:
            continue
        frame[field] = frame[field].astype(float)

    with sqlite3.connect(OUTPUT_DB_PATH) as con:
        frame.to_sql(output_table, con, if_exists="replace", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build training game-level dataset.")
    parser.add_argument(
        "--output-table",
        default=OUTPUT_TABLE,
        help=f"Destination table name in {OUTPUT_DB_PATH.name}.",
    )
    parser.add_argument(
        "--no-player-features",
        action="store_true",
        help="Disable player/injury feature join and build legacy-style dataset columns only.",
    )
    args = parser.parse_args()
    main(output_table=args.output_table, include_player_features=not args.no_player_features)
