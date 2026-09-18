import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from src.Players.features import PlayerFeatureEngine, allocate_minutes
from src.Players.http import get_json, FeedError
from src.Players.identity import team_id
from src.Players.ingest import normalize_logs, normalize_archive
from src.Players.reports import (normalize_injury_status, normalize_lineup_status,
                                 parse_snapshot, refresh_reports,
                                 _parse_official_injury_text,
                                 nba_daily_lineups_snapshot)
from src.Players.store import initialize, save_logs, replace_logs, save_snapshot, NUMERIC
from src.Utils import PlayerContext

NOW='2025-01-03T16:00:00+00:00'


def game_rows(day='2025-01-01',gid='1',tid='ATL',points=10):
    rows=[]
    for n in range(8):
        row={c:0. for c in NUMERIC}
        row.update(GAME_ID=gid,PLAYER_ID=str(team_id(tid))+str(n),PLAYER_NAME=f'Player {tid} {n}',
                   TEAM_ID=team_id(tid),GAME_DATE=day,MIN=30.,PTS=float(points),FGA=10.,FGM=4.,FTA=2.,FTM=2.,AST=4.,REB=5.)
        rows.append(row)
    return rows


def logs():
    return normalize_logs(pd.DataFrame(game_rows()+game_rows('2025-01-02','2')+game_rows(tid='BOS',gid='3')),'2024-25','Regular Season',NOW)


def envelope(records=None,kind='injuries',published='2025-01-03T14:00:00Z',complete=None):
    return parse_snapshot({'game_date':'2025-01-03','published_at':published,
                          'records':records or [],'complete_teams':complete or []},kind,'test',NOW)


class TestPlayerData(unittest.TestCase):
    def test_identity_and_status(self):
        self.assertEqual(team_id('ATL'),team_id('Atlanta Hawks'))
        self.assertEqual(team_id('Los Angeles Clippers'),team_id('LA Clippers'))
        self.assertEqual(normalize_injury_status('Inactive'),'OUT')
        self.assertEqual(normalize_injury_status('Not yet submitted'),'UNKNOWN')
        self.assertEqual(normalize_injury_status('Game Time Decision'),'QUESTIONABLE')
        self.assertEqual(normalize_lineup_status('Inactive'),'OUT')
        self.assertEqual(normalize_lineup_status(None),'UNKNOWN')

    def test_real_nba_schema_maps_id_without_team_name(self):
        rows=logs()
        self.assertEqual(rows[0]['TEAM_NAME'],'Atlanta Hawks')
        self.assertNotEqual(rows[0]['TEAM_ID'],rows[-1]['TEAM_ID'])
        self.assertTrue(rows[0]['PLAYER_ID'].startswith('nba:'))

    def test_invalid_stats_rejected(self):
        frame=pd.DataFrame(game_rows())
        frame.loc[0,'FGA']=np.inf
        with self.assertRaises(ValueError):
            normalize_logs(frame,'2024-25','Regular Season',NOW)

    def test_archive_repairs_only_zero_stat_nonparticipants(self):
        mapping = {
            'GAME_ID':'game_id','PLAYER_ID':'athlete_id','PLAYER_NAME':'athlete_display_name',
            'GAME_DATE':'game_date','MIN':'minutes','PTS':'points','REB':'rebounds',
            'AST':'assists','FGM':'field_goals_made','FGA':'field_goals_attempted',
            'FG3M':'three_point_field_goals_made','FG3A':'three_point_field_goals_attempted',
            'FTM':'free_throws_made','FTA':'free_throws_attempted','OREB':'offensive_rebounds',
            'DREB':'defensive_rebounds','TOV':'turnovers','STL':'steals','BLK':'blocks',
            'PF':'fouls','PLUS_MINUS':'plus_minus',
        }
        source=game_rows()[0]
        source.update({field:0. for field in NUMERIC})
        row={mapping.get(key,key.lower()):value for key,value in source.items()
             if key in mapping}
        row.update(team_abbreviation='ATL',season_type=2,did_not_play=False,
                   minutes=np.nan,plus_minus=3,reason="COACH'S DECISION",active=True)
        repaired=normalize_archive(pd.DataFrame([row]),'2024-25',NOW)
        self.assertEqual(repaired[0]['MIN'],0)

        row['points']=1
        with self.assertRaises(ValueError):
            normalize_archive(pd.DataFrame([row]),'2024-25',NOW)

    def test_archive_drops_unidentifiable_placeholders(self):
        mapping = {
            'GAME_ID':'game_id','PLAYER_ID':'athlete_id','PLAYER_NAME':'athlete_display_name',
            'GAME_DATE':'game_date','MIN':'minutes','PTS':'points','REB':'rebounds',
            'AST':'assists','FGM':'field_goals_made','FGA':'field_goals_attempted',
            'FG3M':'three_point_field_goals_made','FG3A':'three_point_field_goals_attempted',
            'FTM':'free_throws_made','FTA':'free_throws_attempted','OREB':'offensive_rebounds',
            'DREB':'defensive_rebounds','TOV':'turnovers','STL':'steals','BLK':'blocks',
            'PF':'fouls','PLUS_MINUS':'plus_minus',
        }
        row={mapping.get(key,key.lower()):value for key,value in game_rows()[0].items()
             if key in mapping}
        row.update(athlete_id=np.nan,athlete_display_name=np.nan,team_abbreviation='ATL',
                   season_type=2,did_not_play=False,minutes=np.nan)
        self.assertEqual(normalize_archive(pd.DataFrame([row]),'2024-25',NOW),[])

    def test_archive_recovers_short_name_and_missing_minutes(self):
        mapping = {
            'GAME_ID':'game_id','PLAYER_ID':'athlete_id','PLAYER_NAME':'athlete_display_name',
            'GAME_DATE':'game_date','MIN':'minutes','PTS':'points','REB':'rebounds',
            'AST':'assists','FGM':'field_goals_made','FGA':'field_goals_attempted',
            'FG3M':'three_point_field_goals_made','FG3A':'three_point_field_goals_attempted',
            'FTM':'free_throws_made','FTA':'free_throws_attempted','OREB':'offensive_rebounds',
            'DREB':'defensive_rebounds','TOV':'turnovers','STL':'steals','BLK':'blocks',
            'PF':'fouls','PLUS_MINUS':'plus_minus',
        }
        source=game_rows()[0]
        source.update({field:0. for field in NUMERIC})
        rows=[]
        for number_ in range(9):
            current={mapping.get(key,key.lower()):value for key,value in source.items()
                     if key in mapping}
            current.update(athlete_id=number_,athlete_display_name=f'Player {number_}',
                           team_abbreviation='ATL',season_type=2,did_not_play=False,
                           minutes=25,athlete_short_name=f'P. {number_}')
            rows.append(current)
        missing=dict(rows[0])
        missing.update(athlete_id=np.nan,athlete_display_name=np.nan,
                       athlete_short_name='Rookie',minutes=np.nan)
        rows.append(missing)
        normalized=normalize_archive(pd.DataFrame(rows),'2024-25',NOW)
        recovered=[row for row in normalized if row['PLAYER_NAME']=='Rookie'][0]
        self.assertEqual(recovered['MIN'],15)
        self.assertTrue(recovered['PLAYER_ID'].startswith('espn-synthetic:'))
        self.assertIn('minutes_imputed',recovered['RAW_JSON'])
        self.assertEqual(recovered['STARTER'],0)
        self.assertEqual(recovered['HOME_AWAY'],'')
        frame=pd.DataFrame(game_rows()).drop(columns=['GAME_ID'])
        with self.assertRaises(ValueError):
            normalize_logs(frame,'2024-25','Regular Season',NOW)

    def test_log_upsert_and_provider_conflict(self):
        with sqlite3.connect(':memory:') as con:
            initialize(con);save_logs(con,logs());save_logs(con,logs())
            self.assertEqual(con.execute('select count(*) from player_game_logs').fetchone()[0],24)
            changed=logs();changed[0]['SOURCE']='espn'
            with self.assertRaises(ValueError): save_logs(con,changed)

    def test_complete_scope_replacement_removes_stale_rows(self):
        with sqlite3.connect(':memory:') as con:
            initialize(con)
            rows=logs()
            replace_logs(con,rows,'2024-25','nba')
            replace_logs(con,rows[:-1],'2024-25','nba')
            self.assertEqual(con.execute('select count(*) from player_game_logs').fetchone()[0],23)

    def test_pregame_target_mutation_invariant(self):
        original=logs()
        future=normalize_logs(pd.DataFrame(game_rows('2025-01-03','4',points=100)),'2024-25','Regular Season',NOW)
        a=PlayerFeatureEngine(pd.DataFrame(original)).team_features('ATL','2025-01-03')
        b=PlayerFeatureEngine(pd.DataFrame(original+future)).team_features('ATL','2025-01-03')
        self.assertEqual(a,b)
        self.assertEqual(a['PLR_PTS_ROLL7'],80)
        self.assertEqual(a['PLR_HISTORY_GAMES'],2)

    def test_team_separation(self):
        result=PlayerFeatureEngine(pd.DataFrame(logs())).matchup('ATL','BOS','2025-01-03')
        self.assertEqual(result['H_PLR_HISTORY_GAMES'],2)
        self.assertEqual(result['A_PLR_HISTORY_GAMES'],1)

    def test_intraday_and_stale_reports(self):
        snap=envelope([{'team':'ATL','player_name':'Player ATL 0','status':'OUT'}],complete=['ATL'])
        engine=PlayerFeatureEngine(pd.DataFrame(logs()),[snap])
        early=engine.team_features('ATL','2025-01-03','2025-01-03T13:00:00Z')
        late=engine.team_features('ATL','2025-01-03',NOW)
        self.assertEqual(early['INJ_COVERAGE'],0)
        self.assertEqual(late['INJ_COVERAGE'],1)
        self.assertEqual(late['INJ_OUT_MIN_WT'],30)
        self.assertAlmostEqual(late['PLR_PROJECTED_MIN_SUM'],240)
        self.assertEqual(late['PLR_ROTATION_SIZE'],7)

    def test_cross_provider_report_id_resolves_by_team_scoped_name(self):
        snap=envelope([{'team':'ATL','player_id':'other:123',
                        'player_name':'Player ATL 0','status':'OUT'}],complete=['ATL'])
        result=PlayerFeatureEngine(pd.DataFrame(logs()),[snap]).team_features('ATL','2025-01-03',NOW)
        self.assertEqual(result['INJ_COVERAGE'],1)
        self.assertEqual(result['INJ_OUT_MIN_WT'],30)

    def test_no_history_is_unknown_not_healthy(self):
        result=PlayerFeatureEngine(pd.DataFrame()).team_features('ATL','2025-01-03')
        self.assertEqual(result['PLR_GAME_COVERAGE'],0)
        self.assertEqual(result['INJ_AVAIL_MIN_WT'],0)
        self.assertEqual(result['INJ_COVERAGE'],0)

    def test_unknown_report_player_disables_complete_coverage(self):
        snap=envelope([{'team':'ATL','player_name':'Unresolved','status':'OUT'}],complete=['ATL'])
        result=PlayerFeatureEngine(pd.DataFrame(logs()),[snap]).team_features('ATL','2025-01-03',NOW)
        self.assertEqual(result['INJ_COVERAGE'],0)

    def test_configured_injury_weight_applies(self):
        snap=envelope([{'team':'ATL','player_name':'Player ATL 0','status':'QUESTIONABLE'}],complete=['ATL'])
        engine=PlayerFeatureEngine(pd.DataFrame(logs()),[snap],{'player-features':{'defaults':{'injury_weights':{'questionable':0}}}})
        self.assertEqual(engine.team_features('ATL','2025-01-03',NOW)['INJ_Q_MIN_WT'],0)

    def test_bench_is_not_confirmed_starter(self):
        snap=envelope([{'team':'ATL','player_name':'Player ATL 0','status':'BENCH'}],kind='lineups')
        result=PlayerFeatureEngine(pd.DataFrame(logs()),[snap]).team_features('ATL','2025-01-03',NOW)
        self.assertEqual(result['LINEUP_COVERAGE'],0)
        self.assertEqual(result['LINEUP_CONFIRMED_STARTERS'],0)

    def test_report_contract_rejects_missing_time_conflicts_and_future(self):
        for payload in [[],{'records':[]}, {'game_date':'2025-01-03','published_at':'2025-01-03','records':[]}]:
            with self.assertRaises(ValueError):parse_snapshot(payload,'injuries','test',NOW)
        with self.assertRaises(ValueError):envelope(published='2025-01-04T12:00:00Z')
        with self.assertRaises(ValueError):envelope([{'team':'ATL','player_name':'A','status':'OUT'},{'team':'ATL','player_name':'A','status':'ACTIVE'}])

    def test_empty_response_does_not_imply_complete(self):
        snap=envelope()
        result=PlayerFeatureEngine(pd.DataFrame(logs()),[snap]).team_features('ATL','2025-01-03',NOW)
        self.assertEqual(result['INJ_COVERAGE'],0)
        complete=envelope(complete=['ATL'])
        result=PlayerFeatureEngine(pd.DataFrame(logs()),[complete]).team_features('ATL','2025-01-03',NOW)
        self.assertEqual(result['INJ_COVERAGE'],1)
        self.assertEqual(result['INJ_AVAIL_MIN_WT'],1)

    def test_failed_refresh_preserves_history_and_retries_today(self):
        with sqlite3.connect(':memory:') as con:
            initialize(con);save_snapshot(con,envelope())
            config={'player-injuries':{'primary_url':'https://example.invalid/{date}'}}
            with patch('src.Players.reports.get_json',side_effect=FeedError('test')) as fetch:
                refresh_reports(con,config,'2025-01-03');refresh_reports(con,config,'2025-01-03')
            self.assertEqual(fetch.call_count,2)
            self.assertEqual(con.execute('select count(*) from player_report_snapshots').fetchone()[0],1)

    def test_official_injury_parser_keeps_teams_and_slate_separate(self):
        text = '''01/17/2026 05:00 (ET) UTA@DAL Utah Jazz Harkless, Elijah Out Right hip soreness
Kessler, Walker Questionable Illness
Dallas Mavericks Christie, Max Probable Back soreness
Lively II, Dereck Out Foot injury
01/18/2026 01:00 (ET) LAL@BOS Los Angeles Lakers James, LeBron Out Rest'''
        records, complete = _parse_official_injury_text(text, '2026-01-17')
        self.assertEqual(complete, sorted([team_id('UTA'), team_id('DAL')]))
        self.assertEqual([(row['team_id'], row['player_name'], row['status']) for row in records], [
            (team_id('UTA'), 'Elijah Harkless', 'Out'),
            (team_id('UTA'), 'Walker Kessler', 'Questionable'),
            (team_id('DAL'), 'Max Christie', 'Probable'),
            (team_id('DAL'), 'Dereck Lively II', 'Out'),
        ])

    def test_nba_daily_lineups_adapter_maps_lineup_and_roster(self):
        players = [{'personId': n, 'playerName': f'Player {n}',
                    'lineupStatus': 'Confirmed'} for n in range(1, 8)]
        payload = {'games': [{'homeTeam': {'teamId': team_id('ATL'), 'players': players},
                              'awayTeam': {'teamId': team_id('BOS'), 'players': players}}]}
        settings = {'daily_lineups_url_template': 'https://example.test/{date_compact}'}
        artifact = b'{"games":[]}'
        with patch('src.Players.reports.get_json_with_metadata', return_value=(payload, artifact, {'Content-Type': 'application/json'})):
            lineup = nba_daily_lineups_snapshot('2025-01-03', 'lineups', settings, NOW)
            roster = nba_daily_lineups_snapshot('2025-01-03', 'rosters', settings, NOW)
        self.assertEqual(len(lineup['records']), 10)
        self.assertTrue(all(row['status'] == 'Confirmed' for row in lineup['records']))
        self.assertTrue(all(row['player_id'].startswith('nba:') for row in lineup['records']))
        self.assertEqual(lineup['_artifact']['content'], artifact)
        self.assertEqual(len(roster['records']), 14)
        self.assertEqual(roster['complete_teams'], sorted([team_id('ATL'), team_id('BOS')]))

    def test_availability_readiness_requires_all_live_inputs(self):
        engine = PlayerFeatureEngine(pd.DataFrame(logs()))
        missing = engine.availability_readiness('ATL', '2025-01-03', NOW)
        self.assertFalse(missing['eligible'])
        self.assertIn('injury report is missing, stale, or unresolved', missing['reasons'])

        records = [{'team': 'ATL', 'player_name': f'Player ATL {n}', 'status': 'ACTIVE'} for n in range(8)]
        roster = [{'team': 'ATL', 'player_name': f'Player ATL {n}', 'status': 'ROSTER'} for n in range(8)]
        lineup = [{'team': 'ATL', 'player_name': f'Player ATL {n}', 'status': 'CONFIRMED'} for n in range(5)]
        engine = PlayerFeatureEngine(pd.DataFrame(logs()), [
            envelope(records, complete=['ATL']),
            envelope(roster, kind='rosters', complete=['ATL']),
            envelope(lineup, kind='lineups'),
        ])
        ready = engine.availability_readiness('ATL', '2025-01-03', NOW)
        self.assertTrue(ready['eligible'])
        self.assertEqual(ready['features']['PLR_ROSTER_CONFIRMED'], 1)
        self.assertEqual(ready['features']['LINEUP_COVERAGE'], 1)

    def test_unresolved_roster_player_keeps_coverage_unconfirmed(self):
        roster = [{'team': 'ATL', 'player_name': 'Unknown New Player', 'status': 'ROSTER'}]
        result = PlayerFeatureEngine(pd.DataFrame(logs()), [
            envelope(roster, kind='rosters', complete=['ATL']),
        ]).team_features('ATL', '2025-01-03', NOW)
        self.assertEqual(result['PLR_ROSTER_CONFIRMED'], 0)

    def test_snapshot_archives_immutable_source_artifact(self):
        with sqlite3.connect(':memory:') as con:
            initialize(con)
            snapshot = envelope(complete=['ATL'])
            artifact = {'source_url': 'https://source.example/report.json',
                        'content': b'{"records":[]}', 'content_type': 'application/json',
                        'parser_version': 'test-v1'}
            save_snapshot(con, snapshot, artifact)
            row = con.execute('SELECT source_url,content_sha256,content_type,parser_version,content FROM player_report_artifacts').fetchone()
        self.assertEqual(row[0], artifact['source_url'])
        self.assertEqual(len(row[1]), 64)
        self.assertEqual(row[2:], ('application/json', 'test-v1', artifact['content']))

    def test_timeout_retried_with_timeout(self):
        with patch('src.Players.http.requests.get',side_effect=requests.Timeout),patch('src.Players.http.time.sleep'):
            with self.assertRaises(FeedError):get_json('https://example.invalid',attempts=3,timeout=2)
        with patch('src.Players.http.requests.get',side_effect=requests.Timeout) as fetch,patch('src.Players.http.time.sleep'):
            with self.assertRaises(FeedError):get_json('https://example.invalid',attempts=3,timeout=2)
            self.assertEqual(fetch.call_count,3)
            self.assertEqual(fetch.call_args.kwargs['timeout'],2)

    def test_minute_caps_and_total(self):
        result=allocate_minutes([40,35,30,25,20,20,10,10],[20,48,48,48,48,48,48,48])
        self.assertAlmostEqual(result.sum(),240)
        self.assertLessEqual(result[0],20)
        self.assertTrue((result<=48).all())

    def test_legacy_schema_rejected_and_sidecar_roundtrip(self):
        with self.assertRaises(ValueError):PlayerContext.align_frame_to_columns(pd.DataFrame({'a':[1]}),['a','missing'])
        with tempfile.TemporaryDirectory() as d:
            model=Path(d)/'test.json'
            PlayerContext.save_feature_columns(model,['a','H_PLR_PTS_SUM'])
            self.assertEqual(PlayerContext.load_feature_columns(model),['a','H_PLR_PTS_SUM'])
            PlayerContext.feature_sidecar_path(model).write_text(json.dumps({'version':'old','columns':['a']}))
            with self.assertRaises(ValueError):PlayerContext.load_feature_columns(model)

    def test_database_engine_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'players.sqlite'
            with sqlite3.connect(path) as con:
                initialize(con);save_logs(con,logs());save_snapshot(con,envelope(complete=['ATL']))
            engine=PlayerFeatureEngine.from_database(path)
            self.assertEqual(engine.team_features('ATL','2025-01-03',NOW)['INJ_COVERAGE'],1)
