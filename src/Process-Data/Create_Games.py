import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import toml

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.Utils.Dictionaries import team_index_07, team_index_08, team_index_12, team_index_13, team_index_14, \
    team_index_current

config = toml.load("../../config.toml")

df = pd.DataFrame
scores = []
win_margin = []
OU = []
OU_Cover = []
games = []
days_rest_away = []
days_rest_home = []
teams_con = sqlite3.connect("../../Data/TeamData.sqlite")
odds_con = sqlite3.connect("../../Data/OddsData.sqlite")

for key, value in config['create-games'].items():
    print(key)
    odds_df = pd.read_sql_query(f"select * from \"odds_{key}_new\"", odds_con, index_col="index")
    team_table_str = key
    year_count = 0
    season = key

    for row in odds_df.itertuples():
        home_team = row[2]
        away_team = row[3]

        date = row[1]

        team_df = pd.read_sql_query(f"select * from \"{date}\"", teams_con, index_col="index")
        if len(team_df.index) == 30:
            scores.append(row[8])
            OU.append(row[4])
            days_rest_home.append(row[10])
            days_rest_away.append(row[11])
            if row[9] > 0:
                win_margin.append(1)
            else:
                win_margin.append(0)

            if row[8] < row[4]:
                OU_Cover.append(0)
            elif row[8] > row[4]:
                OU_Cover.append(1)
            elif row[8] == row[4]:
                OU_Cover.append(2)

            if season == '2007-08':
                home_team_series = team_df.iloc[team_index_07.get(home_team)]
                away_team_series = team_df.iloc[team_index_07.get(away_team)]
            elif season == '2008-09' or season == "2009-10" or season == "2010-11" or season == "2011-12":
                home_team_series = team_df.iloc[team_index_08.get(home_team)]
                away_team_series = team_df.iloc[team_index_08.get(away_team)]
            elif season == "2012-13":
                home_team_series = team_df.iloc[team_index_12.get(home_team)]
                away_team_series = team_df.iloc[team_index_12.get(away_team)]
            elif season == '2013-14':
                home_team_series = team_df.iloc[team_index_13.get(home_team)]
                away_team_series = team_df.iloc[team_index_13.get(away_team)]
            elif season == '2022-23' or season == '2023-24':
                home_team_series = team_df.iloc[team_index_current.get(home_team)]
                away_team_series = team_df.iloc[team_index_current.get(away_team)]
            else:
                try:
                    home_team_series = team_df.iloc[team_index_14.get(home_team)]
                    away_team_series = team_df.iloc[team_index_14.get(away_team)]
                except Exception as e:
                    print(home_team)
                    raise e
            game = pd.concat([home_team_series, away_team_series.rename(
                index={col: f"{col}.1" for col in team_df.columns.values}
            )])
            games.append(game)
odds_con.close()
teams_con.close()
season = pd.concat(games, ignore_index=True, axis=1)
season = season.T
frame = season.drop(columns=['TEAM_ID', 'TEAM_ID.1'])
frame['Score'] = np.asarray(scores)
frame['Home-Team-Win'] = np.asarray(win_margin)
frame['OU'] = np.asarray(OU)
frame['OU-Cover'] = np.asarray(OU_Cover)
frame['Days-Rest-Home'] = np.asarray(days_rest_home)
frame['Days-Rest-Away'] = np.asarray(days_rest_away)
# fix types
for field in frame.columns.values:
    if 'TEAM_' in field or 'Date' in field or field not in frame:
        continue
    frame[field] = frame[field].astype(float)
con = sqlite3.connect("../../Data/dataset.sqlite")
frame.to_sql("dataset_2012-24_new", con, if_exists="replace")
con.close()
