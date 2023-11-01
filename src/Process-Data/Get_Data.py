import os
import sqlite3
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src.Utils.Dictionaries import (team_index_07, team_index_08,
                                    team_index_12, team_index_13,
                                    team_index_14, team_index_current)

season_array = ["2012-13", "2013-14", "2014-15", "2015-16", "2016-17", "2017-18", 
                "2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]

df = pd.DataFrame
scores = []
win_margin = []
OU = []
OU_Cover = []
games = []
days_rest_away = []
days_rest_home = []
teams_con = sqlite3.connect("../../Data/teams.sqlite")
odds_con = sqlite3.connect("../../Data/odds.sqlite")

for season in tqdm(season_array):
    # ... [rest of the code remains unchanged] ...

frame = season.drop(columns=['TEAM_ID', 'CFID', 'CFPARAMS', 'Unnamed: 0', 'Unnamed: 0.1', 
                             'CFPARAMS.1', 'TEAM_ID.1', 'CFID.1'])
frame['Score'] = np.asarray(scores)
frame['Home-Team-Win'] = np.asarray(win_margin)
frame['OU'] = np.asarray(OU)
frame['OU-Cover'] = np.asarray(OU_Cover)
frame['Days-Rest-Home'] = np.asarray(days_rest_home)
frame['Days-Rest-Away'] = np.asarray(days_rest_away)

# Calculate the Simple Moving Average (SMA) for Points_Scored
window_size = 10  # or any desired window size
frame['SMA_Points_Scored_Home'] = frame.groupby('TEAM_NAME')['PTS'].rolling(window=window_size).mean().reset_index(0, drop=True)
frame['SMA_Points_Scored_Away'] = frame.groupby('TEAM_NAME.1')['PTS.1'].rolling(window=window_size).mean().reset_index(0, drop=True)

# fix types
for field in frame.columns.values:
    if 'TEAM_' in field  or 'Date' in field or field not in frame:
        continue
    frame[field] = frame[field].astype(float)

con = sqlite3.connect("../../Data/dataset.sqlite")
frame.to_sql("dataset_2012-23", con, if_exists="replace")
con.close()
