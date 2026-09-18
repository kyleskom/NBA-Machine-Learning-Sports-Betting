import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import toml
from src.Players.store import initialize,save_logs
from src.Players.identity import team_id
from test_get_player_data import logs

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

    def test_build_dataset_integrates_player_histories(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            paths={name:root/(name+'.sqlite') for name in ['odds','teams','players','output']}
            config={'create-games':{'2024-25':{}},'player-features':{'rolling_windows':[7]}}
            (root/'config.toml').write_text(toml.dumps(config))
            with sqlite3.connect(paths['players']) as con:
                initialize(con);save_logs(con,logs())
            with sqlite3.connect(paths['teams']) as con:
                pd.DataFrame([{'TEAM_ID':team_id(t),'TEAM_NAME':name,'PTS':100,'Date':'2025-01-02'} for t,name in [('BOS','Boston Celtics'),('ATL','Atlanta Hawks')]]).to_sql('2025-01-02',con,index=False)
            with sqlite3.connect(paths['odds']) as con:
                pd.DataFrame([{'Date':'2025-01-03','Home':'Atlanta Hawks','Away':'Boston Celtics','Points':220,'OU':219.5,'Win_Margin':5,'Days_Rest_Home':1,'Days_Rest_Away':2}]).to_sql('2024-25',con,index=False)
            with patch.multiple(m,CONFIG_PATH=root/'config.toml',ODDS_DB_PATH=paths['odds'],TEAMS_DB_PATH=paths['teams'],PLAYER_DB_PATH=paths['players'],OUTPUT_DB_PATH=paths['output']):
                m.main('integration')
            with sqlite3.connect(paths['output']) as con:
                frame=pd.read_sql_query('select * from integration',con)
            self.assertEqual(frame.iloc[0]['H_PLR_HISTORY_GAMES'],2)
            self.assertEqual(frame.iloc[0]['A_PLR_HISTORY_GAMES'],1)
            self.assertEqual(frame.iloc[0]['Date'],'2025-01-03')
            self.assertEqual(frame.iloc[0]['OU-Cover'],1)

    def test_missing_player_database_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'config.toml').write_text('[create-games]\n')
            with patch.multiple(m,CONFIG_PATH=root/'config.toml',ODDS_DB_PATH=root/'odds',TEAMS_DB_PATH=root/'teams',PLAYER_DB_PATH=root/'missing'):
                with self.assertRaisesRegex(ValueError,'No player game logs'):
                    m.main()

    def test_existing_dataset_can_be_upgraded_from_v2_reference(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'dataset.sqlite'
            legacy=pd.DataFrame([{'Date':'2025-01-03','TEAM_NAME':'Atlanta Hawks',
                                  'TEAM_NAME.1':'Boston Celtics','Score':220}])
            reference=legacy.assign(H_PLR_GAME_COVERAGE=1.0,A_PLR_GAME_COVERAGE=1.0)
            with sqlite3.connect(path) as con:
                legacy.to_sql('legacy',con,index=False)
                reference.to_sql(m.OUTPUT_TABLE,con,index=False)
            with patch.object(m,'OUTPUT_DB_PATH',path):
                output,rows,features=m.augment_existing_dataset('legacy')
            with sqlite3.connect(path) as con:
                upgraded=pd.read_sql_query(f'SELECT * FROM "{output}"',con)
            self.assertEqual((output,rows,features),('legacy_player_v2',1,2))
            self.assertEqual(upgraded.iloc[0].H_PLR_GAME_COVERAGE,1.0)

    def test_default_v2_name_is_not_reused_for_a_legacy_copy(self):
        self.assertEqual(m.player_v2_table_name(m.OUTPUT_TABLE),
                         'dataset_2012-26_player_v2_legacy_copy')
