import re
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .Dictionaries import team_index_current

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
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Origin": "https://www.nba.com",
    "Priority": "u=3, i",
    "Referer": "https://www.nba.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0.1 Safari/605.1.15"
}


def get_json_data(url):
    raw_data = requests.get(url, headers=data_headers)
    try:
        json = raw_data.json()
    except Exception as e:
        print(e)
        return {}
    return json.get('resultSets')


def _scoreboard_games(payload, game_date):
    """Normalize current NBA/ESPN scoreboards to the existing h/v format."""
    if not isinstance(payload, dict):
        raise ValueError('Scoreboard must be a JSON object')
    if 'scoreboard' in payload:
        board = payload['scoreboard']
        if board.get('gameDate') != game_date:
            raise ValueError('NBA scoreboard is for a different game date')
        games = board['games']
        source = 'nba'
    elif 'events' in payload:
        games = payload['events']
        source = 'espn'
    elif isinstance(payload.get('gs'), dict):
        games = payload['gs']['g']
        if not isinstance(games, list):
            raise ValueError('Scoreboard games must be a list')
        return games
    else:
        raise ValueError('Unrecognized scoreboard response')
    if not isinstance(games, list):
        raise ValueError('Scoreboard games must be a list')
    result = []
    for game in games:
        if source == 'nba':
            home = game['homeTeam']
            away = game['awayTeam']
            home_name = f"{home['teamCity']} {home['teamName']}"
            away_name = f"{away['teamCity']} {away['teamName']}"
        else:
            sides = {item['homeAway']: item['team']['displayName']
                     for item in game['competitions'][0]['competitors']}
            home_name, away_name = sides['home'], sides['away']
        # Keep the team-name contract expected by feature and odds lookups.
        home_name = home_name.replace('Los Angeles Clippers', 'LA Clippers')
        away_name = away_name.replace('Los Angeles Clippers', 'LA Clippers')
        result.append({'h': {'tc': '', 'tn': home_name}, 'v': {'tc': '', 'tn': away_name}})
    return result


def get_todays_games_json(url, game_date=None):
    day = game_date or datetime.now(ZoneInfo('America/New_York')).date().isoformat()
    fallback = ('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'
                f'?dates={day.replace("-", "")}&limit=100')
    for endpoint in (url, fallback):
        try:
            response = requests.get(endpoint, headers=games_header, timeout=20)
            response.raise_for_status()
            return _scoreboard_games(response.json(), day)
        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            logging.getLogger(__name__).warning('Schedule feed unavailable (%s): %s', endpoint, exc)
    print('Unable to retrieve today\'s NBA schedule. Try again later; no predictions generated.')
    return []


def to_data_frame(data):
    try:
        data_list = data[0]
    except Exception as e:
        print(e)
        return pd.DataFrame(data={})
    return pd.DataFrame(data=data_list.get('rowSet'), columns=data_list.get('headers'))


def create_todays_games(input_list):
    games = []
    for game in input_list:
        home = game.get('h')
        away = game.get('v')
        home_team = (home.get('tc') + ' ' + home.get('tn')).strip()
        away_team = (away.get('tc') + ' ' + away.get('tn')).strip()
        games.append([home_team, away_team])
    return games


def create_todays_games_from_odds(input_dict):
    games = []
    for game in input_dict.keys():
        home_team, away_team = game.split(":")
        if home_team not in team_index_current or away_team not in team_index_current:
            continue
        games.append([home_team, away_team])
    return games


def get_date(date_string):
    year1, month, day = re.search(r'(\d+)-\d+-(\d\d)(\d\d)', date_string).groups()
    year = year1 if int(month) > 8 else int(year1) + 1
    return datetime.strptime(f"{year}-{month}-{day}", '%Y-%m-%d')
