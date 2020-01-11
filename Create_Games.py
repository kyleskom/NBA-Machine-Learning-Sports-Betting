import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.Dictionaries import team_index_07, team_index_08, team_index_12, team_index_13, team_index_14

season = ["2007-08", "2008-09", "2009-10", "2010-11", "2011-12", "2012-13", "2013-14", "2014-15", "2015-16", "2016-17",
          "2017-18", "2018-19", "2019-20"]
odds_directory = os.fsdecode('../Odds-Data-Clean')
df = pd.DataFrame
scores = []
win_margin = []
games = []
for x in tqdm(season):
    file = pd.read_excel(odds_directory + '/' + '{}.xlsx'.format(x))

    team_data_directory = os.fsdecode('../Team-Data/{}'.format(x))

    for row in file.itertuples():
        home_team = row[3]
        away_team = row[4]

        date = row[2]
        date_array = date.split('-')
        year = date_array[0] + '-' + date_array[1]
        month = date_array[2][:2]
        day = date_array[2][2:]

        if month[0] == '0':
            month = month[1:]
        if day[0] == '0':
            day = day[1:]

        team_data_file = month + '-' + day + '-' + year + '.xlsx'

        data_frame = pd.read_excel(team_data_directory + '/' + team_data_file)

        if len(data_frame.index) == 30:
            scores.append(row[9])
            if row[10] > 0:
                win_margin.append(1)
            else:
                win_margin.append(0)

            if x == '2007-08':
                home_team_series = data_frame.iloc[team_index_07.get(home_team)]
                away_team_series = data_frame.iloc[team_index_07.get(away_team)]
            elif x == '2008-09' or x == "2009-10" or x == "2010-11" or x == "2011-12":
                home_team_series = data_frame.iloc[team_index_08.get(home_team)]
                away_team_series = data_frame.iloc[team_index_08.get(away_team)]
            elif x == "2012-13":
                home_team_series = data_frame.iloc[team_index_12.get(home_team)]
                away_team_series = data_frame.iloc[team_index_12.get(away_team)]
            elif x == '2013-14':
                home_team_series = data_frame.iloc[team_index_13.get(home_team)]
                away_team_series = data_frame.iloc[team_index_13.get(away_team)]
            else:
                home_team_series = data_frame.iloc[team_index_14.get(home_team)]
                away_team_series = data_frame.iloc[team_index_14.get(away_team)]

            game = home_team_series.append(away_team_series)
            games.append(game)
x = pd.concat(games, ignore_index=True, axis=1)
x = x.T

frame = x.drop(columns=['TEAM_ID', 'CFID', 'CFPARAMS', 'Unnamed: 0'])
frame['Score'] = np.asarray(scores)
frame['Home-Team-Win'] = np.asarray(win_margin)
frame.to_excel('../Full-Data-Set.xlsx')
