import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.Players.identity import team_id
from src.Storage.normalized import (connect, load_games, load_latest_team_snapshot,
                                    migrate_legacy_odds_data, migrate_legacy_team_data,
                                    save_odds_rows, save_team_snapshot)


class TestNormalizedStorage(unittest.TestCase):
    def test_snapshot_and_quote_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nba.sqlite'
            with connect(path) as con:
                saved = save_team_snapshot(con, pd.DataFrame([
                    {'TEAM_ID': team_id('ATL'), 'TEAM_NAME': 'Atlanta Hawks', 'W': 5, 'L': 2},
                    {'TEAM_ID': team_id('BOS'), 'TEAM_NAME': 'Boston Celtics', 'W': 4, 'L': 3},
                ]), '2025-01-02', '2024-25')
                saved_games = save_odds_rows(con, [{
                    'Date': '2025-01-03', 'Home': 'Atlanta Hawks', 'Away': 'Boston Celtics',
                    'OU': 220.5, 'Spread': -2.5, 'ML_Home': -130, 'ML_Away': 110,
                    'Points': 222, 'Win_Margin': 4, 'Days_Rest_Home': 2, 'Days_Rest_Away': 1,
                }], '2024-25')
                snapshot = load_latest_team_snapshot(con, '2025-01-03')
                games = load_games(con, '2024-25')
            self.assertEqual(saved, 2)
            self.assertEqual(saved_games, 1)
            self.assertEqual(set(snapshot.TEAM_ID), {team_id('ATL'), team_id('BOS')})
            self.assertEqual(games.iloc[0].Home, 'Atlanta Hawks')
            self.assertEqual(games.iloc[0].ML_Away, 110)
            self.assertEqual(games.iloc[0].OU, 220.5)

    def test_legacy_migration_uses_preferred_odds_table(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy_path = Path(directory) / 'legacy.sqlite'
            canonical_path = Path(directory) / 'canonical.sqlite'
            with sqlite3.connect(legacy_path) as legacy, connect(canonical_path) as canonical:
                pd.DataFrame([{'TEAM_ID': team_id('ATL'), 'TEAM_NAME': 'Atlanta Hawks', 'W': 1}]).to_sql(
                    '2025-01-02', legacy, index=False)
                row = {'Date': '2025-01-03', 'Home': 'Atlanta Hawks', 'Away': 'Boston Celtics',
                       'OU': 220, 'Spread': -1, 'ML_Home': -110, 'ML_Away': -110,
                       'Points': 210, 'Win_Margin': 2, 'Days_Rest_Home': 1, 'Days_Rest_Away': 1}
                pd.DataFrame([row]).to_sql('2024-25', legacy, index=False)
                row['OU'] = 225
                pd.DataFrame([row]).to_sql('odds_2024-25_new', legacy, index=False)
                self.assertEqual(migrate_legacy_team_data(legacy, canonical), 1)
                self.assertEqual(migrate_legacy_odds_data(legacy, canonical), 1)
                self.assertEqual(load_games(canonical, '2024-25').iloc[0].OU, 225)
