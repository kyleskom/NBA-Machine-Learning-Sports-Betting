import requests
import pandas as pd
from bs4 import BeautifulSoup

games_header = {
    'user-agent': 'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/57.0.2987.133 Safari/537.36',
    'Dnt': '1',
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Accept-Language': 'en',
    'origin': 'http://stats.nba.com',
    'Referer': 'https://github.com'
}

data_headers = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.4 Safari/605.1.15',
    'Accept-Language': 'en-us',
    'Referer': 'https://stats.nba.com/teams/traditional/?sort=W_PCT&dir=-1&Season=2019-20&SeasonType=Regular%20Season',
    'Connection': 'keep-alive',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true'
}


def get_json_data(url):
    raw_data = requests.get(url, headers=data_headers)
    json = raw_data.json()
    return json.get('resultSets')


def get_todays_games_json(url):
    raw_data = requests.get(url, headers=games_header)
    json = raw_data.json()
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
        # added abrevations to get the data for each team from dictionary
        abrev_home = home.get('ta')
        abrev_away = away.get('ta')
        games.append([home_team, away_team, abrev_home, abrev_away])
    return games

#  the method that scrapes for data in yahoo website and returns the dictionary
def get_odds():
    dict_res = {}
    url = 'https://sports.yahoo.com/nba/odds/'
    html_page = requests.get(url)
    soup = BeautifulSoup(html_page.text, 'lxml')
    matches_box = soup.find_all("div", class_=r'D(f) Jc(sb) Mb(4px)')
    for matche in matches_box:
        matche_side = matche.find_all(class_=r'D(f) Jc(sb) Ai(c) W(96px)')
        i = 0
        for side in matche_side:
            match_info = side.find_all("span", class_=r'Fw(b)')
            if i % 2 == 0:
                team_name = match_info[0].get_text()
                team_money_line_odds = match_info[1].get_text()
                dict_res[team_name] = team_money_line_odds
                i += 1
            elif i % 2 != 0:
                team_name = match_info[1].get_text()
                team_money_line_odds = match_info[0].get_text()
                dict_res[team_name] = team_money_line_odds
                i += 1
            print(i)

    return dict_res
