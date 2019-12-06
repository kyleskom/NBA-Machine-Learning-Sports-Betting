import os
from tqdm import tqdm
import pandas as pd

directory = os.fsdecode('../Odds-Data-Clean')
team_data_directory = os.fsdecode('../Team-Data')

df = pd.DataFrame
scores = []

file = pd.read_excel(directory + '/' + '2007-08.xlsx')

for row in file.itertuples():
    home_team = row[3]
    away_team = row[4]
    scores.append(row[9])
    date = row[2]
    date_array = date.split('-')
    year = date_array[0] + '-' + date_array[1]
    month = date_array[2][:2]
    day = date_array[2][2:]

    if month[0] is '0':
        month = month[1:]
    if day[0] is '0':
        day = day[1:]

    team_data_file = month + '-' + day + '-' + year + '.xlsx'

    data_frame = pd.read_excel(team_data_directory + '/' + team_data_file)

    if len(data_frame.index) == 30:
        home_team_series = ''
        away_team_series = ''
        for team_row in data_frame.itertuples():
            print(team_row[3])
            if team_row[3] is home_team:
                home_team_series = team_row
        for team_row in data_frame.itertuples():
            if team_row[3] is away_team:
                away_team_series = team_row
        print('here')
