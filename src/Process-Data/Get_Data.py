import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta

import toml

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.Utils.tools import get_json_data, to_data_frame

config = toml.load("../../config.toml")

url = config['data_url']

con = sqlite3.connect("../../Data/TeamData.sqlite")
cursor = con.cursor()

for key, value in config['get-data'].items():
    date_pointer = datetime.strptime(value['start_date'], "%Y-%m-%d").date()
    end_date = datetime.strptime(value['end_date'], "%Y-%m-%d").date()

    while date_pointer <= end_date:
        table_name = date_pointer.strftime("%Y-%m-%d")

        # Check if the table already exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if cursor.fetchone():
            print(f"Skipping {table_name}, already exists.")
        else:
            print(f"Getting data: {table_name}")

            raw_data = get_json_data(
                url.format(date_pointer.month, date_pointer.day, value['start_year'], date_pointer.year, key))
            df = to_data_frame(raw_data)

            df['Date'] = str(date_pointer)
            df.to_sql(table_name, con, if_exists="replace")

            time.sleep(random.randint(1, 3))

        date_pointer += timedelta(days=1)

con.close()
