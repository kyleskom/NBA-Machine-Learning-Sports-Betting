import os
from tqdm import tqdm
import pandas as pd
import numpy as np

team_codes = {
    'Atlanta Hawks': 0,
    'Boston Celtics': 1,
    'Charlotte Bobcats': 2,
    'Chicago Bulls': 3,
    'Cleveland Cavaliers': 4,
    'Dallas Mavericks': 5,
    'Denver Nuggets': 6,
    'Detroit Pistons': 7,
    'Golden State Warriors': 8,
    'Houston Rockets': 9,
    'Indiana Pacers': 10,
    'Los Angeles Clippers': 11,
    'Los Angeles Lakers': 12,
    'Memphis Grizzlies': 13,
    'Miami Heat': 14,
    'Milwaukee Bucks': 15,
    'Minnesota Timberwolves': 16,
    'New Jersey Nets': 17,
    'New Orleans Pelicans': 18,
    'New York Knicks': 19,
    'Orlando Magic': 20,
    'Philadelphia 76ers': 21,
    'Phoenix Suns': 22,
    'Portland Trail Blazers': 23,
    'Sacramento Kings': 24,
    'San Antonio Spurs': 25,
    'Seattle SuperSonics': 26,
    'Toronto Raptors': 27,
    'Utah Jazz': 28,
    'Washington Wizards': 29
}


directory = os.fsdecode('../Odds-Data-Clean')
team_data_directory = os.fsdecode('../Team-Data')

df = pd.DataFrame
scores = []
win_margin = []
games = []
file = pd.read_excel(directory + '/' + '2007-08.xlsx')

for row in tqdm(file.itertuples()):
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

        home_team_series = data_frame.iloc[team_codes.get(home_team)]
        away_team_series = data_frame.iloc[team_codes.get(away_team)]

        game = home_team_series.append(away_team_series)
        games.append(game)
x = pd.concat(games, ignore_index=True, axis=1)
x = x.T

frame = x.drop(columns=['TEAM_ID', 'CFID', 'CFPARAMS', 'Unnamed: 0'])
frame['Score'] = np.asarray(scores)
frame['Home-Team-Win'] = np.asarray(win_margin)
frame.to_excel('../2007-2008-Games.xlsx')
