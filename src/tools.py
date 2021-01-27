import pandas as pd
from apis.stats_nba_com import get_connection


conn = get_connection()


def get_json_data(url):
    json = conn.get(url)
    return json.get('resultSets')


def get_todays_games_json(url):
    json = conn.get(url)
    return json.get('gs').get('g')


def to_data_frame(data):
    data_list = data[0]
    return pd.DataFrame(data=data_list.get('rowSet'), columns=data_list.get('headers'))


def create_todays_games(input_list):
    games = []
    for game in input_list:
        home = game.get('h')
        away = game.get('v')
        home_team = home.get('tc') + ' ' + home.get('tn')
        away_team = away.get('tc') + ' ' + away.get('tn')
        games.append([home_team, away_team])
    return games
