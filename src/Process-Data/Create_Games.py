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

from src.Utils import PlayerContext
from src.Players.features import PlayerFeatureEngine
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
PLAYER_DB_PATH = BASE_DIR / "Data" / "PlayerData.sqlite"
OUTPUT_DB_PATH = BASE_DIR / "Data" / "dataset.sqlite"
PLAYER_FEATURES_TABLE = PlayerContext.PLAYER_FEATURES_TABLE
OUTPUT_TABLE = "dataset_2012-26_player_v2"

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


def get_player_feature_columns(config):
    return PlayerContext.get_player_feature_columns(config)


def get_player_defaults(config, feature_cols):
    return PlayerContext.get_player_defaults(config, feature_cols)


def build_player_feature_row(feature_map, date_str, home_team, away_team, feature_cols, default_values):
    return PlayerContext.build_player_feature_row(feature_map, date_str, home_team, away_team,
                                                  feature_cols, default_values)


def player_columns(frame):
    prefixes = ('H_PLR_', 'A_PLR_', 'H_INJ_', 'A_INJ_', 'H_LINEUP_', 'A_LINEUP_')
    return [column for column in frame.columns if column.startswith(prefixes)]


def player_v2_table_name(table_name):
    if table_name == OUTPUT_TABLE:
        return f'{table_name}_legacy_copy'
    return table_name if table_name.endswith('_player_v2') else f'{table_name}_player_v2'


def augment_existing_dataset(input_table, output_table=None, reference_table=OUTPUT_TABLE):
    """Create a player-aware copy of an existing game dataset without changing labels."""
    output_table = output_table or player_v2_table_name(input_table)
    key_columns = ['Date', 'TEAM_NAME', 'TEAM_NAME.1']
    with sqlite3.connect(OUTPUT_DB_PATH) as con:
        legacy = pd.read_sql_query(f'SELECT * FROM "{input_table}"', con)
        reference = pd.read_sql_query(f'SELECT * FROM "{reference_table}"', con)
        missing = set(key_columns) - set(legacy.columns)
        if missing:
            raise ValueError(f'{input_table} cannot be upgraded; missing game keys {sorted(missing)}')
        features = player_columns(reference)
        if not features:
            raise ValueError(f'{reference_table} does not contain player features')
        lookup = reference[key_columns + features].copy()
        if lookup.duplicated(key_columns).any():
            raise ValueError(f'{reference_table} has duplicate game identity keys')
        legacy = legacy.drop(columns=player_columns(legacy), errors='ignore')
        upgraded = legacy.merge(lookup, on=key_columns, how='left', validate='many_to_one', sort=False)
        unmatched = int(upgraded[features[0]].isna().sum())
        if unmatched:
            config = toml.load(CONFIG_PATH)
            engine = PlayerFeatureEngine.from_database(PLAYER_DB_PATH, config)
            if engine.logs.empty:
                raise ValueError(f'{input_table}: {unmatched} games need player history, but the player database is empty')
            missing_rows = upgraded[features[0]].isna()
            for index, row in upgraded.loc[missing_rows].iterrows():
                values = engine.matchup(row['TEAM_NAME'], row['TEAM_NAME.1'], row['Date'])
                upgraded.loc[index, features] = [values[column] for column in features]
            print(f'{input_table}: generated point-in-time player features for {unmatched} early rows absent from {reference_table}')
        upgraded.to_sql(output_table, con, if_exists='replace', index=False)
    return output_table, len(upgraded), len(features)


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

    canonical_path = canonical_path_for(ODDS_DB_PATH)
    with sqlite3.connect(ODDS_DB_PATH) as odds_con, sqlite3.connect(TEAMS_DB_PATH) as teams_con, connect_canonical(canonical_path) as canonical_con:
        player_feature_map = PlayerFeatureEngine.from_database(PLAYER_DB_PATH, config)
        if include_player_features and player_feature_map.logs.empty:
            raise ValueError("No player game logs. Run Get_Player_Data --backfill before building player-v2 data.")
        team_cache = {}
        for season_key in config["create-games"].keys():
            print(season_key)
            odds_df = load_normalized_games(canonical_con, season_key)
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

    if include_player_features:
        print("Player history coverage:", frame[["H_PLR_GAME_COVERAGE", "A_PLR_GAME_COVERAGE"]].mean().to_dict())
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
    parser.add_argument(
        "--augment-table",
        action="append",
        help="Create a <table>_player_v2 copy using player features from the current v2 dataset; repeat for multiple tables.",
    )
    args = parser.parse_args()
    if args.augment_table:
        for table in args.augment_table:
            output, rows, features = augment_existing_dataset(table)
            print(f'{table} -> {output}: {rows} rows, {features} player columns')
    else:
        main(output_table=args.output_table, include_player_features=not args.no_player_features)
