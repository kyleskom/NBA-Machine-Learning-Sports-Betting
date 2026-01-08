import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import toml

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config.toml"
DB_PATH = BASE_DIR / "Data" / "OddsData.sqlite"
DATE_FORMAT = "%Y-%m-%d"


def parse_legacy_date(raw_date, base_year, seen_january):
    date_parts = str(raw_date).split('-')
    if len(date_parts) < 3:
        return None, seen_january

    month = date_parts[2][:2]
    day = date_parts[2][2:]
    if len(month) != 2 or len(day) != 2:
        return None, seen_january

    if month == "01" and not seen_january:
        seen_january = True

    year = int(base_year) + (1 if seen_january else 0)
    date_str = f"{year}-{month}-{day}"
    return datetime.strptime(date_str, DATE_FORMAT).date(), seen_january


def main():
    config = toml.load(CONFIG_PATH)

    with sqlite3.connect(DB_PATH) as odds_con:
        for season_key in config["get-data"].keys():
            print(season_key)
            odds_df = pd.read_sql_query(
                f'SELECT * FROM "odds_{season_key}"', odds_con, index_col="index"
            )
            if odds_df.empty:
                print(f"No rows found for odds_{season_key}.")
                continue
            if "Date" not in odds_df.columns:
                print(f"Missing Date column in odds_{season_key}.")
                continue

            new_dates = []
            seen_january = False
            base_year = season_key.split("-")[0]

            for raw_date in odds_df["Date"].tolist():
                parsed_date, seen_january = parse_legacy_date(raw_date, base_year, seen_january)
                if parsed_date is None:
                    new_dates.append(None)
                    print(f"Skipping invalid date: {raw_date}")
                    continue
                new_dates.append(str(parsed_date))
                print(f"Old date = {raw_date} : New date = {parsed_date}")

            odds_df["Date"] = new_dates
            odds_df.drop(odds_df.filter(regex="Unname"), axis=1, inplace=True)
            odds_df.to_sql(f"odds_{season_key}_new", odds_con, if_exists="replace", index=False)


if __name__ == "__main__":
    main()
