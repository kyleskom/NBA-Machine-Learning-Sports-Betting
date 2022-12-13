import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.Utils.Dictionaries import team_index_07, team_index_08, team_index_12, team_index_13, team_index_14

season_array = ["2007-08", "2008-09", "2009-10", "2010-11", "2011-12", "2012-13", "2013-14", "2014-15", "2015-16",
                "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22"]
odds_directory = os.fsdecode('../../Odds-Data/Odds-Data-Clean')
df = pd.DataFrame
scores = []
win_margin = []
OU = []
OU_Cover = []
games = []
for season in tqdm(season_array):
    file = pd.read_excel(odds_directory + '/' + '{}.xlsx'.format(season))

    team_data_directory = os.fsdecode('../../Team-Data/{}'.format(season))

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
            OU.append(row[5])
            if row[10] > 0:
                win_margin.append(1)
            else:
                win_margin.append(0)

            if row[9] < row[5]:
                OU_Cover.append(0)
            elif row[9] > row[5]:
                OU_Cover.append(1)
            elif row[9] == row[5]:
                OU_Cover.append(2)

            if season == '2007-08':
                home_team_series = data_frame.iloc[team_index_07.get(home_team)]
                away_team_series = data_frame.iloc[team_index_07.get(away_team)]
            elif season == '2008-09' or season == "2009-10" or season == "2010-11" or season == "2011-12":
                home_team_series = data_frame.iloc[team_index_08.get(home_team)]
                away_team_series = data_frame.iloc[team_index_08.get(away_team)]
            elif season == "2012-13":
                home_team_series = data_frame.iloc[team_index_12.get(home_team)]
                away_team_series = data_frame.iloc[team_index_12.get(away_team)]
            elif season == '2013-14':
                home_team_series = data_frame.iloc[team_index_13.get(home_team)]
                away_team_series = data_frame.iloc[team_index_13.get(away_team)]
            else:
                home_team_series = data_frame.iloc[team_index_14.get(home_team)]
                away_team_series = data_frame.iloc[team_index_14.get(away_team)]

            game = home_team_series.append(away_team_series)
            games.append(game)
season = pd.concat(games, ignore_index=True, axis=1)
season = season.T

frame = season.drop(columns=['TEAM_ID', 'CFID', 'CFPARAMS', 'Unnamed: 0'])
frame['Score'] = np.asarray(scores)
frame['Home-Team-Win'] = np.asarray(win_margin)
frame['OU'] = np.asarray(OU)
frame['OU-Cover'] = np.asarray(OU_Cover)
frame.to_excel('../../Datasets/DataSet-2021-22.xlsx')
