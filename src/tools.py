import requests
import pandas as pd

HEADERS = {
    'user-agent': 'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/57.0.2987.133 Safari/537.36',
    'Dnt': '1',
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Accept-Language': 'en',
    'origin': 'http://stats.nba.com',
    'Referer': 'https://github.com'
}


def get_json_data(url):
    raw_data = requests.get(url, headers=HEADERS)
    json = raw_data.json()
    return json.get('resultSets')


def get_todays_games_json(url):
    raw_data = requests.get(url, headers=HEADERS)
    json = raw_data.json()
    return json.get('gs').get('g')


def to_data_frame(data):
    data_list = data[0]
    return pd.DataFrame(data=data_list.get('rowSet'), columns=data_list.get('headers'))


def create_todays_games(list):
    games = []
    for game in list:
        home = game.get('h')
        away = game.get('v')
        home_team = home.get('tc') + ' ' + home.get('tn')
        away_team = away.get('tc') + ' ' + away.get('tn')
        games.append([home_team, away_team])
    return games
