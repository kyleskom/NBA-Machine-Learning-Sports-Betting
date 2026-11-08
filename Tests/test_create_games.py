import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import toml
from src.Players.identity import team_id

spec=importlib.util.spec_from_file_location('create_games',Path(__file__).resolve().parents[1]/'src/Process-Data/Create_Games.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class TestCreateGames(unittest.TestCase):
    def test_pregame_team_table_and_id_mapping(self):
        with sqlite3.connect(':memory:') as con:
            for day,pts in [('2025-01-02',100),('2025-01-03',999)]:
                pd.DataFrame([{'TEAM_ID':team_id('BOS'),'TEAM_NAME':'Boston Celtics','PTS':pts},
                              {'TEAM_ID':team_id('ATL'),'TEAM_NAME':'Atlanta Hawks','PTS':pts+1}]).to_sql(day,con,index=False)
            frame=m.fetch_team_table(con,'2025-01-03')
            features=m.build_game_features(frame,'ATL','BOS')
            self.assertEqual(features['PTS'],101)
            self.assertEqual(features['PTS.1'],100)
            self.assertIsNone(m.fetch_team_table(con,'2025-01-01'))

    def test_build_dataset_uses_only_team_stats_without_player_database(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            paths={name:root/(name+'.sqlite') for name in ['odds','teams','output']}
            config={'create-games':{'2024-25':{}}}
            (root/'config.toml').write_text(toml.dumps(config))
            with sqlite3.connect(paths['teams']) as con:
                pd.DataFrame([{'TEAM_ID':team_id(t),'TEAM_NAME':name,'PTS':100,'Date':'2025-01-02'} for t,name in [('BOS','Boston Celtics'),('ATL','Atlanta Hawks')]]).to_sql('2025-01-02',con,index=False)
            with sqlite3.connect(paths['odds']) as con:
                pd.DataFrame([{'Date':'2025-01-03','Home':'Atlanta Hawks','Away':'Boston Celtics','Points':220,'OU':219.5,'Win_Margin':5,'Days_Rest_Home':1,'Days_Rest_Away':2}]).to_sql('2024-25',con,index=False)
            with patch.multiple(m,CONFIG_PATH=root/'config.toml',ODDS_DB_PATH=paths['odds'],TEAMS_DB_PATH=paths['teams'],OUTPUT_DB_PATH=paths['output']):
                m.main('integration')
            with sqlite3.connect(paths['output']) as con:
                frame=pd.read_sql_query('select * from integration',con)
            self.assertFalse(any(column.startswith(('H_PLR_', 'A_PLR_', 'H_INJ_', 'A_INJ_', 'H_LINEUP_', 'A_LINEUP_')) for column in frame))
            self.assertEqual(frame.iloc[0]['PTS'],100)
            self.assertEqual(frame.iloc[0]['Date'],'2025-01-03')
            self.assertEqual(frame.iloc[0]['OU-Cover'],1)
