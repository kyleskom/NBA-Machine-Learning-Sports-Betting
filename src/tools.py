import requests
import pandas as pd
import nba_py

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


def to_data_frame(data):
    data_list = data[0]
    return pd.DataFrame(data=data_list.get('rowSet'), columns=data_list.get('headers'))


def get_teams_playing_today():
    nba_py.HEADERS = HEADERS
    sb = nba_py.Scoreboard()
    results = sb.json.get('resultSets')
    raw_data = results[0]
    game_data = raw_data.get('rowSet')
    games = []
    for game in game_data:
        games.append([game[6], game[7]])
    return games
