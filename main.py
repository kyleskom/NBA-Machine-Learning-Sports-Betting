# from tensorflow.keras.models import load_model
# import tensorflow as tf
from src.tools import get_teams_playing_today, get_json_data, to_data_frame
import pandas as pd
import numpy as np
import copy

todays_games_url = 'https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/2019/scores/00_todays_scores.json'

data = get_json_data(todays_games_url)
df = to_data_frame(data)



url = 'https://stats.nba.com/stats/leaguedashteamstats?' \
      'Conference=&DateFrom=&DateTo=&Division=&GameScope=&' \
      'GameSegment=&LastNGames=0&LeagueID=00&Location=&' \
      'MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&' \
      'PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&' \
      'PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&' \
      'Season=2019-20&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&' \
      'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='

data = get_json_data(url)
df = to_data_frame(data)

#games = get_teams_playing_today()
print('DONE')

# model = load_model('Trained-Model')
# data = pd.read_excel('Full-Data-Set.xlsx')
# data = data.iloc[15068:]
# copy = copy.deepcopy(data)
# scores = data['Score']
# margin = data['Home-Team-Win']
# data.drop(['Score'], axis=1, inplace=True)
# data.drop(['Home-Team-Win'], axis=1, inplace=True)
#
# data = data.drop(columns=['Unnamed: 0', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1'])
# data = data.values
# data = data.astype(float)
#
# x_train = tf.keras.utils.normalize(data, axis=1)
# arr = []
# for row in x_train:
#       arr.append(model.predict(np.array([row])))
#
# for x in arr:
#       print(np.argmax(x))
