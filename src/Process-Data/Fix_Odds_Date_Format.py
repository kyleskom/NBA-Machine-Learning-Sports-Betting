import sqlite3
from datetime import datetime

import pandas as pd
import toml

config = toml.load("config.toml")

odds_con = sqlite3.connect("Data/OddsData.sqlite")

date_format = "%Y-%m-%d"

for key, value in config['get-data'].items():
    print(key)
    odds_df = pd.read_sql_query(f"select * from \"odds_{key}\"", odds_con, index_col="index")
    team_table_str = key
    year_count = 0
    arr = []

    for row in odds_df.itertuples():
        date = row[2]
        date_array = date.split('-')
        if not date_array or len(date_array) < 2:
            continue
        year = date_array[0]
        month = date_array[2][:2]
        day = date_array[2][2:]

        if month == '01':
            year_count += 1

        if year_count > 0:
            year = str(int(year) + 1)

        date_str = f'{year}-{month}-{day}'
        new_date = datetime.strptime(date_str, date_format).date()

        arr.append(str(new_date))
        print(f'Old date = {date} : New date = {new_date}')

    odds_df['Date'] = arr
    odds_df.drop(odds_df.filter(regex="Unname"), axis=1, inplace=True)
    odds_df.to_sql(f'odds_{key}_new', odds_con, if_exists="replace")
