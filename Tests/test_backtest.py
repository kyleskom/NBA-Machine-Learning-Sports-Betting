import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.Evaluation.backtest import (
    align_candidate, canonical_games, compare_reports, evaluate, load_baseline, main,
)
from src.Evaluation.experiment import moneyline_ledger, summarize_moneyline_ledger, attach_moneyline_market


class TestFrozenBacktest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / 'data.sqlite'
        self.frame = pd.DataFrame([
            {'Date': f'2024-01-{day:02}', 'TEAM_NAME': f'H{game}', 'TEAM_NAME.1': f'A{game}',
             'Home-Team-Win': (day + game) % 2, 'rating': float(day + game),
             'H_PLR_FORM': 0.1 * day, 'A_PLR_FORM': 0.2 * game,
             'Score': 200 + day}
            for day in range(1, 11) for game in range(3)
        ])
        with sqlite3.connect(self.db) as con:
            self.frame.to_sql('games', con, index=False)
        self.output = self.root / 'runs'
        self.common = ['--dataset-db', str(self.db), '--dataset', 'games', '--output-root', str(self.output)]

    def attach(self, frame, sportsbook):
        merged = frame.rename(columns={'TEAM_NAME': 'Home', 'TEAM_NAME.1': 'Away'}).copy()
        merged['ML_Home'], merged['ML_Away'] = -110., -110.
        return merged, {'available': True, 'quote_timing': 'unknown'}

    def baseline(self):
        with patch('src.Evaluation.backtest.attach_moneyline_market', self.attach), contextlib.redirect_stdout(io.StringIO()):
            main(['baseline', '--name', 'base', '--start', '2024-01-08', '--calibration-fraction', '0.25', *self.common])
        return self.output / 'base'

    def test_repeat_uses_frozen_inputs_and_produces_identical_metrics(self):
        baseline = self.baseline()
        with patch('src.Evaluation.backtest.attach_moneyline_market', side_effect=AssertionError('Must not fetch odds')), contextlib.redirect_stdout(io.StringIO()):
            main(['run', '--name', 'same', '--baseline', str(baseline), *self.common])
        first = json.loads((baseline / 'report.json').read_text())
        second = json.loads((self.output / 'same/report.json').read_text())
        self.assertEqual(first['diagnostics'], second['diagnostics'])
        self.assertEqual(first['betting'], second['betting'])
        self.assertEqual(first['model']['feature_data_sha256'], second['model']['feature_data_sha256'])
        self.assertEqual(second['comparison']['log_loss']['delta'], 0)
        self.assertEqual(first['diagnostics']['rows'], 9)
        self.assertNotIn('Score', first['model']['features'])
        self.assertTrue((baseline / 'model.joblib').exists())

    def test_ablation_and_external_shuffled_predictions_share_protocol(self):
        baseline = self.baseline()
        with contextlib.redirect_stdout(io.StringIO()):
            main(['run', '--name', 'drop', '--baseline', str(baseline), '--drop-feature', '*PLR*', *self.common])
        report = json.loads((self.output / 'drop/report.json').read_text())
        self.assertEqual(report['model']['features'], ['rating'])
        ledger = pd.read_csv(baseline / 'predictions.csv', float_precision='round_trip')
        prediction_file = self.root / 'external.csv'
        ledger[['game_id', 'home_probability']].sample(frac=1, random_state=7).to_csv(prediction_file, index=False, float_format='%.17g')
        with contextlib.redirect_stdout(io.StringIO()):
            main(['run', '--name', 'external', '--baseline', str(baseline), '--predictions', str(prediction_file), *self.common])
        external = json.loads((self.output / 'external/report.json').read_text())
        self.assertEqual(external['comparison']['log_loss']['delta'], 0)

    def test_baseline_integrity_and_overwrite_are_rejected(self):
        baseline = self.baseline()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['baseline', '--name', 'base', '--start', '2024-01-08', *self.common])
        with (baseline / 'quotes.csv').open('a') as file:
            file.write('\n')
        with self.assertRaisesRegex(ValueError, 'Baseline file changed'):
            load_baseline(baseline)

    def test_feature_updates_require_same_game_population_and_outcomes(self):
        reference = canonical_games(self.frame)
        changed = self.frame.sample(frac=1, random_state=7).copy()
        changed['new_feature'] = 10.
        aligned = align_candidate(changed, reference)
        self.assertEqual(aligned.game_id.tolist(), reference.game_id.tolist())
        with self.assertRaisesRegex(ValueError, 'coverage'):
            align_candidate(changed.iloc[1:], reference)
        changed.loc[changed.index[0], 'Home-Team-Win'] = 1 - changed.iloc[0]['Home-Team-Win']
        with self.assertRaisesRegex(ValueError, 'outcomes'):
            align_candidate(changed, reference)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            canonical_games(pd.concat([self.frame, self.frame.iloc[:1]]))

    def test_incompatible_protocols_cannot_be_compared(self):
        with self.assertRaisesRegex(ValueError, 'different games'):
            compare_reports({'protocol_id': 'a'}, {'protocol_id': 'b'})

    def test_evaluation_keyed_quotes_cannot_reorder_predictions(self):
        frame = canonical_games(self.frame)
        test = frame.loc[frame.Date >= '2024-01-08']
        quotes = test[['game_id']].copy()
        quotes['ML_Home'] = [-200., 150., -110.] * 3
        quotes['ML_Away'] = [170., -180., -110.] * 3
        p = np.linspace(.1, .9, len(test))
        protocol = {'start': '2024-01-08', 'policy': {'min_ev': .02, 'min_probability': 0}}
        a, ledger = evaluate(frame, quotes, p, protocol)
        b, shuffled = evaluate(frame, quotes.sample(frac=1, random_state=1), p, protocol)
        self.assertEqual(a, b)
        pd.testing.assert_frame_equal(ledger, shuffled)
        with self.assertRaisesRegex(ValueError, 'probabilities'):
            evaluate(frame, quotes, np.full(len(test), np.nan), protocol)

    def test_underdog_value_missing_quotes_and_settlement(self):
        quotes = pd.DataFrame({'ML_Home': [200., -110., np.nan], 'ML_Away': [-240., -110., np.nan]})
        ledger = moneyline_ledger(quotes, [1, 0, 1], [.4, .4, .9], min_ev=.02)
        self.assertEqual(ledger.side.tolist()[:2], ['home', 'away'])
        self.assertEqual(ledger.status.tolist(), ['BET', 'BET', 'PASS'])
        self.assertAlmostEqual(ledger.iloc[0].expected_value, .2)
        self.assertAlmostEqual(ledger.iloc[0].profit_units, 2.)
        self.assertEqual(summarize_moneyline_ledger(ledger)['bets'], 2)
        mismatched = quotes.assign(**{'Home-Team-Win': [0, 0, 1]})
        with self.assertRaisesRegex(ValueError, 'misaligned'):
            moneyline_ledger(mismatched, [1, 0, 1], [.4, .4, .9])
        with self.assertRaisesRegex(ValueError, 'aligned'):
            moneyline_ledger(quotes, [1], [.4])

    def test_quote_join_preserves_nondefault_index_and_same_date_order(self):
        frame = self.frame.iloc[[2, 0, 1]].copy()
        quotes = frame.rename(columns={'TEAM_NAME': 'Home', 'TEAM_NAME.1': 'Away'})[['Date', 'Home', 'Away']]
        quotes['Date'] = pd.to_datetime(quotes.Date)
        quotes['ML_Home'] = [300., 100., 200.]
        quotes['ML_Away'] = [-300., -100., -200.]
        with patch('src.Evaluation.experiment._moneyline_quotes', return_value=quotes.iloc[::-1]):
            merged, _ = attach_moneyline_market(frame)
        self.assertEqual(merged.Home.tolist(), ['H2', 'H0', 'H1'])
        self.assertEqual(merged.ML_Home.tolist(), [300., 100., 200.])

    def test_legacy_xgboost_preparation_preserves_same_date_game_order(self):
        for kind in ['ML', 'UO']:
            path = Path(__file__).resolve().parents[1] / f'src/Train-Models/XGBoost_Model_{kind}.py'
            spec = importlib.util.spec_from_file_location(f'test_trainer_{kind}', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            frame = pd.DataFrame({'Date': ['2024-01-01'] * 100, 'marker': range(100),
                                  'Home-Team-Win': np.arange(100) % 2, 'OU-Cover': np.arange(100) % 3})
            ordered = frame.sort_values('Date', kind='stable').reset_index(drop=True)
            X, y, columns = module.prepare_data(ordered)
            np.testing.assert_array_equal(X[:, columns.index('marker')], ordered.marker)
            np.testing.assert_array_equal(y, ordered[module.TARGET_COLUMN])


if __name__ == '__main__':
    unittest.main()
