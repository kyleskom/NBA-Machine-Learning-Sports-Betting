import importlib.util
from pathlib import Path
import unittest

import pandas as pd


PATH = Path(__file__).resolve().parents[1] / 'src/Process-Data/Import_Consensus_Odds.py'
SPEC = importlib.util.spec_from_file_location('import_consensus_odds', PATH)
consensus = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(consensus)


class TestConsensusOdds(unittest.TestCase):
    def test_game_dates_rest_and_consensus_prices(self):
        frame = pd.DataFrame([
            dict(date_et='2026-10-23', start_time_utc='2026-10-23T23:00:00Z',
                 home_team='Boston Celtics', away_team='Detroit Pistons',
                 close_total=221.5, close_spread_home=-3.5, home_score=112, away_score=100),
            dict(date_et='2026-10-20', start_time_utc='2026-10-20T23:00:00Z',
                 home_team='Detroit Pistons', away_team='Boston Celtics',
                 close_total=220.5, close_spread_home=1.5, home_score=121, away_score=113),
        ])
        rows = consensus.build_rows(frame)
        self.assertEqual([row['Date'] for row in rows], ['2026-10-20', '2026-10-23'])
        self.assertEqual(rows[0]['Days_Rest_Home'], 7)
        self.assertEqual(rows[1]['Days_Rest_Home'], 3)
        self.assertEqual(rows[1]['Days_Rest_Away'], 3)
        self.assertEqual(rows[0]['Spread'], -1.5)
        self.assertEqual(rows[0]['Points'], 234)
        self.assertEqual(rows[0]['Win_Margin'], 8)
        self.assertTrue(all(row['ML_Home'] is None and row['ML_Away'] is None for row in rows))

    def test_duplicate_games_are_rejected(self):
        row = dict(date_et='2026-10-20', home_team='Detroit Pistons', away_team='Boston Celtics')
        with self.assertRaisesRegex(ValueError, 'duplicate games'):
            consensus.build_rows(pd.DataFrame([row, row]))
