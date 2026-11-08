from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import main


class TestMainPredictions(unittest.TestCase):
    def test_input_collection_uses_team_data_without_player_reports(self):
        today = datetime(2026, 11, 7, 18, 0, tzinfo=timezone.utc)
        with patch.object(main, 'live_team_stats_url', return_value='https://example.test/stats'), \
             patch.object(main, 'get_json_data'), patch.object(main, 'to_data_frame'), \
             patch.object(main, 'load_schedule'), \
             patch.object(main, 'create_todays_games_data', return_value='inputs') as build:
            inputs = main.collect_prediction_inputs({}, '2026-27', today, [], {})
        self.assertEqual(build.call_args.args[4], today)
        self.assertEqual(inputs, 'inputs')

    def run_legacy_board(self, rows):
        games = [['Atlanta Hawks', 'Boston Celtics'], ['Denver Nuggets', 'Orlando Magic']]
        data = np.array([[10.0], [20.0]])
        inputs = (data, [210.5, 220.5], pd.DataFrame(rows), [rows[0].get('home_odds'), rows[1].get('home_odds')], [100, 110])
        args = SimpleNamespace(A=True, xgb=False, nn=False, odds='fanduel', kc=False)
        with patch.object(main.toml, 'load', return_value={}), \
             patch.object(main, 'runtime_season', return_value='2026-27'), \
             patch.object(main, 'SbrOddsProvider'), \
             patch.object(main, 'resolve_games', return_value=(games, {})), \
             patch.object(main, 'collect_prediction_inputs', return_value=inputs), \
             patch.object(main, 'run_models') as run, patch('builtins.print') as output:
            main.main(args)
        return run, output

    def test_incomplete_game_does_not_block_a_complete_game(self):
        run, output = self.run_legacy_board([{'home_odds': None}, {'home_odds': -120}])
        np.testing.assert_array_equal(run.call_args.args[0], [[20.0]])
        self.assertEqual(run.call_args.args[2], [220.5])
        self.assertEqual(run.call_args.args[4], [['Denver Nuggets', 'Orlando Magic']])
        self.assertEqual(run.call_args.args[5:7], ([-120], [110]))
        self.assertTrue(any('PASS Boston Celtics @ Atlanta Hawks: home moneyline is missing or invalid' in str(call)
                            for call in output.call_args_list))

    def test_all_incomplete_games_print_pass_without_model_calls(self):
        run, output = self.run_legacy_board([{'home_odds': None}, {'home_odds': None}])
        run.assert_not_called()
        self.assertEqual(sum('PASS ' in str(call) for call in output.call_args_list), 2)

    def test_shared_board_restores_colored_predictions_and_expected_value(self):
        from colorama import Fore
        record = {'status': 'READY', 'home_team': 'Atlanta Hawks', 'away_team': 'Boston Celtics',
                  'moneyline': {'pick': 'Atlanta Hawks', 'home_probability': .6, 'away_probability': .4},
                  'total': {'pick': 'UNDER', 'line': 220.5, 'probability': .55},
                  'expected_value': {'home': 4.5, 'away': -7.0},
                  'kelly': {'home': 2.0, 'away': 0.0}}
        with patch.object(main.XGBoost_Runner, 'init'), patch.object(main.XGBoost_Runner, 'deinit'), \
             patch('builtins.print') as output:
            main.print_board({'games': [record]})
        rendered = '\n'.join(str(call.args[0]) for call in output.call_args_list)
        for color in (Fore.GREEN, Fore.RED, Fore.CYAN, Fore.MAGENTA):
            self.assertIn(color, rendered)
        self.assertIn('Fraction of Bankroll: 2.0%', rendered)
