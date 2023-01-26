import os
import random
import time
from datetime import date, datetime, timedelta
import pandas as pd

from tqdm import tqdm
from sbrscrape import Scoreboard
from src.Utils.tools import get_json_data, to_data_frame

# year = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
year = [2022, 2023]
season = ["2022-23"]
# season = ["2007-08", "2008-09", "2009-10", "2010-11", "2011-12", "2012-13", "2013-14", "2014-15", "2015-16", "2016-17",
#           "2017-18", "2018-19", "2019-20", "2020-2021", "2021-2022"]

month = [10, 11, 12, 1 ] #, 2, 3, 4, 5, 6]
days = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

begin_year_pointer = year[0]
end_year_pointer = year[0]
count = 0
year_count = 0

sportsbook='fanduel'
df_data = []

for season1 in tqdm(season):
    for month1 in tqdm(month):
        if month1 == 1:
            count += 1
            end_year_pointer = year[count]
        for day1 in tqdm(days):
            if month1 == 10 and day1 < 19:
                continue
            if month1 in [4,6,9,11] and day1 > 30:
                continue
            if month1 == 2 and day1 > 28:
                continue
            if end_year_pointer == datetime.now().year:
                if month1 == datetime.now().month and day1 > datetime.now().day:
                    continue
                if month1 > datetime.now().month:
                    continue
            sb = Scoreboard(date=f"{end_year_pointer}-{month1}-{day1}")
            if not hasattr(sb, "games"):
                continue
            for game in sb.games:
                df_data.append({
                    'Home': game['home_team'],
                    'Away': game['away_team'],
                    'OU': game['total'][sportsbook],
                    'Spread': game['away_spread'][sportsbook],
                    'ML_Home': game['home_ml'][sportsbook],
                    'ML_Away': game['away_ml'][sportsbook],
                    'Points': game['away_score'] + game['home_score'],
                    'Win_Margin': game['home_score'] - game['away_score'],
                })
            time.sleep(random.randint(1, 3))
    year_count += 1
    begin_year_pointer = year[count]

    df = pd.DataFrame(df_data,)

    directory2 = os.fsdecode('../../Odds-Data/Odds-Data-Clean/')
    name = directory2 + '/' + '{}'.format(season1) + '.xlsx'
    df.to_excel(name, index=False)