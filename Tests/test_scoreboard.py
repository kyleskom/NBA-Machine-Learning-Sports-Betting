import unittest
from unittest.mock import Mock, patch

import requests

from src.Utils.tools import create_todays_games, get_todays_games_json


class TestScoreboard(unittest.TestCase):
    day = '2026-10-26'

    def response(self, payload):
        response = Mock()
        response.json.return_value = payload
        return response

    def espn(self):
        return {'events': [{'competitions': [{'competitors': [
            {'homeAway': 'away', 'team': {'displayName': 'Boston Celtics'}},
            {'homeAway': 'home', 'team': {'displayName': 'Los Angeles Clippers'}},
        ]}]}]}

    def test_http_error_falls_back_to_date_specific_espn(self):
        blocked = Mock()
        blocked.raise_for_status.side_effect = requests.HTTPError('403 Forbidden')
        with patch('src.Utils.tools.requests.get', side_effect=[blocked, self.response(self.espn())]) as get:
            games = create_todays_games(get_todays_games_json('https://nba.test', self.day))
        self.assertEqual(games, [['LA Clippers', 'Boston Celtics']])
        blocked.json.assert_not_called()
        self.assertIn('dates=20261026', get.call_args_list[1].args[0])
        self.assertEqual(get.call_args.kwargs['timeout'], 20)

    def test_non_json_response_falls_back(self):
        html = Mock()
        html.json.side_effect = ValueError('Not JSON')
        with patch('src.Utils.tools.requests.get', side_effect=[html, self.response(self.espn())]):
            self.assertEqual(len(get_todays_games_json('https://nba.test', self.day)), 1)

    def test_stale_nba_board_falls_back(self):
        stale = {'scoreboard': {'gameDate': '2026-10-25', 'games': []}}
        with patch('src.Utils.tools.requests.get', side_effect=[self.response(stale), self.response(self.espn())]):
            self.assertEqual(len(get_todays_games_json('https://nba.test', self.day)), 1)

    def test_current_nba_board_preserves_home_and_away(self):
        payload = {'scoreboard': {'gameDate': self.day, 'games': [{
            'homeTeam': {'teamCity': 'Boston', 'teamName': 'Celtics'},
            'awayTeam': {'teamCity': 'New York', 'teamName': 'Knicks'},
        }]}}
        with patch('src.Utils.tools.requests.get', return_value=self.response(payload)) as get:
            games = create_todays_games(get_todays_games_json('https://nba.test', self.day))
        self.assertEqual(games, [['Boston Celtics', 'New York Knicks']])
        get.assert_called_once()

    def test_valid_empty_slate_does_not_fall_back(self):
        payload = {'scoreboard': {'gameDate': self.day, 'games': []}}
        with patch('src.Utils.tools.requests.get', return_value=self.response(payload)) as get:
            self.assertEqual(get_todays_games_json('https://nba.test', self.day), [])
        get.assert_called_once()

    def test_failed_feeds_return_empty_without_json_exception(self):
        with patch('src.Utils.tools.requests.get', side_effect=requests.Timeout), patch('builtins.print') as output:
            self.assertEqual(get_todays_games_json('https://nba.test', self.day), [])
        self.assertIn('Unable to retrieve', output.call_args.args[0])
