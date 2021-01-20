import copy
import numpy as np
import pandas as pd
import tensorflow as tf
from colorama import Fore, Style
from tensorflow.keras.models import load_model
from src.Dictionaries import team_index_current
from src.tools import get_json_data, to_data_frame, get_todays_games_json, create_todays_games

model = load_model('Models/Trained-Model-ML')
ou_model = load_model("Models/Trained-Model-OU")

todays_games_url = 'https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/2020/scores/00_todays_scores.json'
data_url = 'https://stats.nba.com/stats/leaguedashteamstats?' \
           'Conference=&DateFrom=&DateTo=&Division=&GameScope=&' \
           'GameSegment=&LastNGames=0&LeagueID=00&Location=&' \
           'MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&' \
           'PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&' \
           'PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&' \
           'Season=2020-21&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&' \
           'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='

data = get_todays_games_json(todays_games_url)
games = create_todays_games(data)

data = get_json_data(data_url)
df = to_data_frame(data)

match_data = []
todays_games_uo = []

for game in games:
    home_team = game[0]
    away_team = game[1]
    todays_games_uo.append(input(home_team + ' vs ' + away_team + ': '))
    home_team_series = df.iloc[team_index_current.get(home_team)]
    away_team_series = df.iloc[team_index_current.get(away_team)]
    stats = home_team_series.append(away_team_series)
    match_data.append(stats)

games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
games_data_frame = games_data_frame.T

frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'CFID', 'CFPARAMS', 'TEAM_NAME'])
data = frame_ml.values
data = data.astype(float)
data = tf.keras.utils.normalize(data, axis=1)

ml_predictions_array = []

for row in data:
    ml_predictions_array.append(model.predict(np.array([row])))

frame_uo = copy.deepcopy(frame_ml)
frame_uo['OU'] = np.asarray(todays_games_uo)
data = frame_uo.values
data = data.astype(float)
data = tf.keras.utils.normalize(data, axis=1)

ou_predictions_array = []

for row in data:
    ou_predictions_array.append(ou_model.predict(np.array([row])))

count = 0
for game in games:
    home_team = game[0]
    away_team = game[1]
    winner = int(np.argmax(ml_predictions_array[count]))
    under_over = int(np.argmax(ou_predictions_array[count]))
    winner_confidence = ml_predictions_array[count]
    un_confidence = ou_predictions_array[count]
    if winner == 1:
        winner_confidence = round(winner_confidence[0][1] * 100, 1)
        if under_over == 0:
            un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
            print(Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                  Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)")
        else:
            un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
            print(Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                  Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)")
    else:
        winner_confidence = round(winner_confidence[0][0] * 100, 1)
        if under_over == 0:
            un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
            print(Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ': ' +
                  Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)")
        else:
            un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
            print(Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ': ' +
                  Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)")
    count += 1
