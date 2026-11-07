from contextlib import nullcontext
from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import main


def ready_row():
    return {**{column: 1 for column in ['H_PLR_GAME_COVERAGE', 'A_PLR_GAME_COVERAGE',
            'H_INJ_COVERAGE', 'A_INJ_COVERAGE', 'H_PLR_ROSTER_CONFIRMED', 'A_PLR_ROSTER_CONFIRMED',
            'H_LINEUP_COVERAGE', 'A_LINEUP_COVERAGE']},
            'H_LINEUP_CONFIRMED_STARTERS': 5, 'A_LINEUP_CONFIRMED_STARTERS': 5}


class TestMainPredictions(unittest.TestCase):
    def test_feature_cutoff_is_captured_after_report_refresh(self):
        before = datetime(2026, 11, 7, 18, 0, tzinfo=timezone.utc)
        after = datetime(2026, 11, 7, 18, 1, tzinfo=timezone.utc)
        clock = {'now': before}

        def refresh(*args):
            clock['now'] = after
            return {'lineups': 'updated'}

        with patch.object(main, 'datetime') as date_time, \
             patch.object(main, 'connect_players', return_value=nullcontext(None)), \
             patch.object(main, 'refresh_reports', side_effect=refresh), \
             patch.object(main, 'live_team_stats_url', return_value='https://example.test/stats'), \
             patch.object(main, 'get_json_data'), patch.object(main, 'to_data_frame'), \
             patch.object(main, 'load_schedule'), \
             patch.object(main, 'create_todays_games_data', return_value='inputs') as build:
            date_time.now.side_effect = lambda *args: clock['now']
            reports, inputs = main.collect_prediction_inputs({}, '2026-27', before, [], {})
        self.assertEqual(build.call_args.args[4], after)
        self.assertEqual(reports, {'lineups': 'updated'})
        self.assertEqual(inputs, 'inputs')

    def run_legacy_board(self, rows):
        games = [['Atlanta Hawks', 'Boston Celtics'], ['Denver Nuggets', 'Orlando Magic']]
        data = np.array([[10.0], [20.0]])
        inputs = (data, [210.5, 220.5], pd.DataFrame(rows), [-110, -120], [100, 110])
        args = SimpleNamespace(A=True, xgb=False, nn=False, odds='fanduel', kc=False)
        with patch.object(main.toml, 'load', return_value={}), \
             patch.object(main, 'runtime_season', return_value='2026-27'), \
             patch.object(main, 'SbrOddsProvider'), \
             patch.object(main, 'resolve_games', return_value=(games, {})), \
             patch.object(main, 'collect_prediction_inputs', return_value=({}, inputs)), \
             patch.object(main, 'run_models') as run, patch('builtins.print') as output:
            main.main(args)
        return run, output

    def test_incomplete_game_does_not_block_a_complete_game(self):
        blocked = ready_row()
        blocked['H_INJ_COVERAGE'] = 0
        run, output = self.run_legacy_board([blocked, ready_row()])
        np.testing.assert_array_equal(run.call_args.args[0], [[20.0]])
        self.assertEqual(run.call_args.args[2], [220.5])
        self.assertEqual(run.call_args.args[4], [['Denver Nuggets', 'Orlando Magic']])
        self.assertEqual(run.call_args.args[5:7], ([-120], [110]))
        self.assertTrue(any('PASS Boston Celtics @ Atlanta Hawks: home injury report incomplete' in str(call)
                            for call in output.call_args_list))

    def test_all_incomplete_games_print_pass_without_model_calls(self):
        row = ready_row()
        row['H_PLR_ROSTER_CONFIRMED'] = 0
        run, output = self.run_legacy_board([row, row])
        run.assert_not_called()
        self.assertEqual(sum('PASS ' in str(call) for call in output.call_args_list), 2)
