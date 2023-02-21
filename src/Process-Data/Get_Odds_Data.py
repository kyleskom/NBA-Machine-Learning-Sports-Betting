import random
import time
import pandas as pd
import sqlite3

from datetime import datetime
from tqdm import tqdm
from sbrscrape import Scoreboard

year = [2022, 2023]
season = ["2022-23"]

month = [10, 11, 12, 1, 2, 3, 4, 5, 6]
days = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

begin_year_pointer = year[0]
end_year_pointer = year[0]
count = 0
year_count = 0

sportsbook='fanduel'
df_data = []

con = sqlite3.connect("../../Data/odds.sqlite")

for season1 in tqdm(season):
    for month1 in tqdm(month):
        if month1 == 1:
            count += 1
            end_year_pointer = year[year_count]
        for day1 in tqdm(days):
            if month1 == 10 and day1 < 19:
                continue
            if month1 in [4,6,9,11] and day1 > 30:
                continue
            if month1 == 2 and day1 > 28:
                continue
            if end_year_pointer == datetime.now().year:
                if month1 == datetime.now().month and day1 >= datetime.now().day:
                    continue
                if month1 > datetime.now().month:
                    continue
            sb = Scoreboard(date=f"{end_year_pointer}-{month1:02}-{day1:02}")
            if not hasattr(sb, "games"):
                continue
            for game in sb.games:
                try:
                    df_data.append({
                        'Unnamed: 0': 0,
                        'Date': f"{season1}-{month1:02}{day1:02}",
                        'Home': game['home_team'],
                        'Away': game['away_team'],
                        'OU': game['total'][sportsbook],
                        'Spread': game['away_spread'][sportsbook],
                        'ML_Home': game['home_ml'][sportsbook],
                        'ML_Away': game['away_ml'][sportsbook],
                        'Points': game['away_score'] + game['home_score'],
                        'Win_Margin': game['home_score'] - game['away_score'],
                    })
                except KeyError:
                    print(f"No {sportsbook} odds data found for game: {game}")
            time.sleep(random.randint(1, 3))
    year_count += 1
    begin_year_pointer = year[count]

    df = pd.DataFrame(df_data,)
    df.to_sql(f"odds_{season1}", con, if_exists="replace")
con.close()