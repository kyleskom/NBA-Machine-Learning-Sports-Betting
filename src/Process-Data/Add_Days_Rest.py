import argparse
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "Data" / "OddsData.sqlite"
DATE_FORMAT = "%Y-%m-%d"
DEFAULT_DATASETS = [
    "odds_2007-08",
    "odds_2008-09",
    "odds_2009-10",
    "odds_2010-11",
    "odds_2011-12",
    "odds_2012-13",
    "odds_2013-14",
    "odds_2014-15",
    "odds_2015-16",
    "odds_2016-17",
    "odds_2017-18",
    "odds_2018-19",
    "odds_2019-20",
    "odds_2020-21",
    "odds_2021-22",
    "odds_2022-23",
]


def get_date(date_string):
    match = re.search(r"(\d+)-\d+-(\d\d)(\d\d)", str(date_string))
    if not match:
        return None
    year1, month, day = match.groups()
    year = year1 if int(month) > 8 else int(year1) + 1
    return datetime.strptime(f"{year}-{month}-{day}", DATE_FORMAT)


def clamp_days_rest(days):
    if days <= 0:
        return 9
    return min(days, 9)


def add_days_rest_to_dataset(con, dataset):
    data = pd.read_sql_query(f'SELECT * FROM "{dataset}"', con)
    if data.empty:
        print(f"No rows found in {dataset}.")
        return
    if "Home" not in data.columns or "Away" not in data.columns or "Date" not in data.columns:
        print(f"Missing required columns in {dataset}.")
        return

    teams_last_played = {}
    days_rest_home = []
    days_rest_away = []

    for _, row in data.iterrows():
        current_date = get_date(row["Date"])
        if current_date is None:
            days_rest_home.append(9)
            days_rest_away.append(9)
            continue

        home_team = row["Home"]
        away_team = row["Away"]

        if home_team not in teams_last_played:
            home_games_rested = 10
        else:
            home_games_rested = clamp_days_rest((current_date - teams_last_played[home_team]).days)
        teams_last_played[home_team] = current_date

        if away_team not in teams_last_played:
            away_games_rested = 10
        else:
            away_games_rested = clamp_days_rest((current_date - teams_last_played[away_team]).days)
        teams_last_played[away_team] = current_date

        days_rest_home.append(home_games_rested)
        days_rest_away.append(away_games_rested)

    data["Days_Rest_Home"] = days_rest_home
    data["Days_Rest_Away"] = days_rest_away
    data.to_sql(dataset, con, if_exists="replace", index=False)


def main(datasets):
    with sqlite3.connect(DB_PATH) as con:
        for dataset in tqdm(datasets):
            add_days_rest_to_dataset(con, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add days-rest columns to odds tables.")
    parser.add_argument(
        "--datasets",
        help="Comma-separated list of odds tables (default: all configured).",
    )
    args = parser.parse_args()
    datasets = DEFAULT_DATASETS
    if args.datasets:
        datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    main(datasets)
