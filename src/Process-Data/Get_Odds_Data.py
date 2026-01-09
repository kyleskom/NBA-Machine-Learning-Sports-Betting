import argparse
import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import toml
from sbrscrape import Scoreboard

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.fspath(BASE_DIR))

CONFIG_PATH = BASE_DIR / "config.toml"
DB_PATH = BASE_DIR / "Data" / "OddsData.sqlite"
MIN_DELAY_SECONDS = 1
MAX_DELAY_SECONDS = 3


def load_config():
    return toml.load(CONFIG_PATH)


def iter_dates(start_date, end_date):
    date_pointer = start_date
    while date_pointer <= end_date:
        yield date_pointer
        date_pointer += timedelta(days=1)


def fetch_scoreboard(date_pointer):
    try:
        return Scoreboard(date=date_pointer)
    except Exception as exc:
        print(f"Failed to fetch odds data for {date_pointer}: {exc}")
        return None


def parse_date_value(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


def table_exists(con, table_name):
    cursor = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def select_current_season(config, today):
    for season_key, value in config["get-odds-data"].items():
        start_date = datetime.strptime(value["start_date"], "%Y-%m-%d").date()
        end_date = datetime.strptime(value["end_date"], "%Y-%m-%d").date()
        if start_date <= today <= end_date:
            return season_key, value, start_date, end_date
    return None, None, None, None


def get_existing_dates(con, season_key):
    if not table_exists(con, season_key):
        return set()
    cursor = con.execute(f'SELECT DISTINCT Date FROM "{season_key}"')
    dates = set()
    for (value,) in cursor.fetchall():
        date_value = parse_date_value(value)
        if date_value:
            dates.add(date_value)
    return dates


def get_teams_last_played(con, season_key, before_date):
    if not table_exists(con, season_key):
        return {}
    df = pd.read_sql_query(
        f'SELECT Date, Home, Away FROM "{season_key}" WHERE Date < ?',
        con,
        params=[before_date],
    )
    teams_last_played = {}
    for row in df.itertuples(index=False):
        date_value = parse_date_value(row.Date)
        if not date_value:
            continue
        teams_last_played[row.Home] = max(date_value, teams_last_played.get(row.Home, date_value))
        teams_last_played[row.Away] = max(date_value, teams_last_played.get(row.Away, date_value))
    return teams_last_played


def get_existing_games_by_date(con, season_key, start_date, end_date):
    if not table_exists(con, season_key):
        return {}
    df = pd.read_sql_query(
        f'SELECT Date, Home, Away FROM "{season_key}" WHERE Date BETWEEN ? AND ?',
        con,
        params=[start_date, end_date],
    )
    games_by_date = {}
    for row in df.itertuples(index=False):
        date_value = parse_date_value(row.Date)
        if not date_value:
            continue
        games_by_date.setdefault(date_value, []).append((row.Home, row.Away))
    return games_by_date


def append_game_rows(game_rows, date_pointer, game, sportsbook, teams_last_played):
    def days_rest(team):
        last_played = teams_last_played.get(team)
        if last_played is None:
            return 7
        return (date_pointer - last_played).days

    home_team = game["home_team"]
    away_team = game["away_team"]
    home_days = days_rest(home_team)
    away_days = days_rest(away_team)

    teams_last_played[home_team] = date_pointer
    teams_last_played[away_team] = date_pointer

    game_rows.append({
        "Date": date_pointer,
        "Home": home_team,
        "Away": away_team,
        "OU": game["total"][sportsbook],
        "Spread": game["away_spread"][sportsbook],
        "ML_Home": game["home_ml"][sportsbook],
        "ML_Away": game["away_ml"][sportsbook],
        "Points": game["away_score"] + game["home_score"],
        "Win_Margin": game["home_score"] - game["away_score"],
        "Days_Rest_Home": home_days,
        "Days_Rest_Away": away_days,
    })


def collect_odds_for_dates(dates, sportsbook, teams_last_played):
    df_data = []
    for date_pointer in dates:
        print("Getting odds data:", date_pointer)
        sb = fetch_scoreboard(date_pointer)
        if not sb or not hasattr(sb, "games") or not sb.games:
            time.sleep(random.randint(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))
            continue

        for game in sb.games:
            try:
                append_game_rows(df_data, date_pointer, game, sportsbook, teams_last_played)
            except KeyError:
                print(f"No {sportsbook} odds data found for game: {game}")

        time.sleep(random.randint(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))

    return df_data


def backfill_season(con, season_key, value, sportsbook, today):
    start_date = datetime.strptime(value["start_date"], "%Y-%m-%d").date()
    end_date = datetime.strptime(value["end_date"], "%Y-%m-%d").date()
    fetch_end = min(today - timedelta(days=1), end_date)

    existing_by_date = get_existing_games_by_date(con, season_key, start_date, fetch_end)
    teams_last_played = {}
    new_rows = []

    for date_pointer in iter_dates(start_date, fetch_end):
        if date_pointer in existing_by_date:
            for home_team, away_team in existing_by_date[date_pointer]:
                teams_last_played[home_team] = date_pointer
                teams_last_played[away_team] = date_pointer
            continue

        sb = fetch_scoreboard(date_pointer)
        if not sb or not hasattr(sb, "games") or not sb.games:
            time.sleep(random.randint(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))
            continue

        print("Backfilling odds data:", date_pointer)
        for game in sb.games:
            try:
                append_game_rows(new_rows, date_pointer, game, sportsbook, teams_last_played)
            except KeyError:
                print(f"No {sportsbook} odds data found for game: {game}")

        time.sleep(random.randint(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))

    if new_rows:
        df = pd.DataFrame(new_rows)
        df.to_sql(season_key, con, if_exists="append", index=False)


def main(sportsbook="fanduel", backfill=False, season=None, today=None, db_path=DB_PATH):
    config = load_config()
    if today is None:
        today = datetime.today().date()

    with sqlite3.connect(db_path) as con:
        if backfill:
            season_items = config["get-odds-data"].items()
            if season:
                season_items = [
                    (key, value) for key, value in season_items if key == season
                ]
                if not season_items:
                    print("Season not found in config:", season)
                    return
            for season_key, value in season_items:
                backfill_season(con, season_key, value, sportsbook, today)
            return

        season_key, value, start_date, end_date = select_current_season(config, today)
        if not season_key:
            print("No current season found for today:", today)
            return

        fetch_end = min(today, end_date)
        existing_dates = get_existing_dates(con, season_key)
        latest_date = max(existing_dates) if existing_dates else None
        fetch_start = start_date if latest_date is None else latest_date + timedelta(days=1)
        if fetch_start > fetch_end:
            print("No new odds dates to fetch. Latest date:", latest_date)
            return

        teams_last_played = get_teams_last_played(con, season_key, fetch_start)
        df_data = collect_odds_for_dates(iter_dates(fetch_start, fetch_end), sportsbook, teams_last_played)
        if df_data:
            df = pd.DataFrame(df_data)
            df.to_sql(season_key, con, if_exists="append", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NBA odds data from SBR.")
    parser.add_argument(
        "--sportsbook",
        default="fanduel",
        help="Sportsbook key to pull odds from (default: fanduel).",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Fetch missing odds dates for seasons in config.toml.",
    )
    parser.add_argument(
        "--season",
        help="Limit backfill to a single season key (e.g. 2025-26).",
    )
    args = parser.parse_args()
    main(sportsbook=args.sportsbook, backfill=args.backfill, season=args.season)
