import xgboost as xgb
from src.tools import get_json_data, to_data_frame, get_todays_games_json, create_todays_games
from tensorflow.keras.models import load_model
import pandas as pd
from src.Dictionaries import team_index_current

todays_games_url = 'https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/2020/scores/00_todays_scores.json'
model = load_model('Models/Trained-Model-ML')

data_url = 'https://stats.nba.com/stats/leaguedashteamstats?' \
           'Conference=&DateFrom=&DateTo=&Division=&GameScope=&' \
           'GameSegment=&LastNGames=0&LeagueID=00&Location=&' \
           'MeasureType={}&Month=0&OpponentTeamID=0&Outcome=&' \
           'PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&' \
           'PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&' \
           'Season=2020-21&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&' \
           'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='

booster = xgb.Booster()
xgb_uo_model = booster.load_model('Models/XGBoost_59.0%_UO.model')

data = get_todays_games_json(todays_games_url)
games = create_todays_games(data)

stats = ['Base', 'Advanced', 'Four Factors', 'Misc', 'Scoring', 'Defense', 'Opponent']

df = pd.DataFrame()
df_list = []
for x in stats:
    y = get_json_data(data_url.format(x))
    df_list.append(to_data_frame(y))
df = pd.concat(df_list, axis=1)



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


print('test')
