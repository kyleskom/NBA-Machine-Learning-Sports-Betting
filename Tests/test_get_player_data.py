import importlib.util
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

GET_PLAYER_DATA_PATH = Path(__file__).resolve().parents[1] / "src" / "Process-Data" / "Get_Player_Data.py"
SPEC = importlib.util.spec_from_file_location("get_player_data", GET_PLAYER_DATA_PATH)
get_player_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(get_player_data)


class TestGetPlayerData(unittest.TestCase):

    def test_normalize_injury_status(self):
        self.assertEqual(get_player_data.normalize_injury_status("out"), "OUT")
        self.assertEqual(get_player_data.normalize_injury_status("Questionable"), "QUESTIONABLE")
        self.assertEqual(get_player_data.normalize_injury_status("healthy"), "ACTIVE")
        self.assertEqual(get_player_data.normalize_injury_status(None), "UNKNOWN")

    def test_normalize_lineup_status(self):
        self.assertEqual(get_player_data.normalize_lineup_status("confirmed"), "CONFIRMED")
        self.assertEqual(get_player_data.normalize_lineup_status("projected starter"), "PROJECTED")
        self.assertEqual(get_player_data.normalize_lineup_status(None), "STARTER")

    def test_parse_injury_payload(self):
        payload = {
            "injuries": [
                {"player_name": "Player A", "team_name": "Team A", "status": "out"},
                {"name": "Player B", "team": "Team B", "injury_status": "questionable"},
                {"name": ""},
            ]
        }
        df = get_player_data.parse_injury_payload(payload, date(2025, 1, 2), "primary")
        self.assertEqual(len(df.index), 2)
        self.assertEqual(df.iloc[0]["STATUS"], "OUT")
        self.assertEqual(df.iloc[1]["STATUS"], "QUESTIONABLE")
        self.assertEqual(df.iloc[0]["Date"], "2025-01-02")

    def test_parse_lineup_payload(self):
        payload = {
            "lineups": [
                {"player_name": "Player A", "team_name": "Team A", "status": "confirmed"},
                {"name": "Player B", "team": "Team B", "lineup_status": "projected"},
                {"name": ""},
            ]
        }
        df = get_player_data.parse_lineup_payload(payload, date(2025, 1, 2), "primary")
        self.assertEqual(len(df.index), 2)
        self.assertEqual(df.iloc[0]["STATUS"], "CONFIRMED")
        self.assertEqual(df.iloc[1]["STATUS"], "PROJECTED")
        self.assertEqual(df.iloc[0]["Date"], "2025-01-02")

    def test_select_current_season(self):
        config = {
            "get-player-data": {
                "2025-26": {
                    "start_date": "2025-10-01",
                    "end_date": "2026-06-01",
                    "start_year": "2025",
                    "end_year": "2026",
                }
            }
        }
        season_key, value, start_date, end_date = get_player_data.select_current_season(
            config, date(2025, 12, 1)
        )
        self.assertEqual(season_key, "2025-26")
        self.assertEqual(value["start_year"], "2025")
        self.assertEqual(start_date.isoformat(), "2025-10-01")
        self.assertEqual(end_date.isoformat(), "2026-06-01")

    def test_get_existing_dates(self):
        with sqlite3.connect(":memory:") as con:
            pd.DataFrame(
                [
                    {"Date": "2025-01-02"},
                    {"Date": "2025-01-03 00:00:00"},
                ]
            ).to_sql(get_player_data.PLAYER_STATS_TABLE, con, if_exists="replace", index=False)
            dates = get_player_data.get_existing_dates(con)
        self.assertEqual(dates, {date(2025, 1, 2), date(2025, 1, 3)})

    def test_rebuild_team_features(self):
        stats_rows = [
            {
                "PLAYER_NAME": "Player A1",
                "TEAM_NAME": "Team A",
                "MIN": 30,
                "PTS": 20,
                "REB": 6,
                "AST": 5,
                "USG_PCT": 0.25,
                "Date": "2025-01-01",
            },
            {
                "PLAYER_NAME": "Player A1",
                "TEAM_NAME": "Team A",
                "MIN": 32,
                "PTS": 22,
                "REB": 7,
                "AST": 6,
                "USG_PCT": 0.26,
                "Date": "2025-01-02",
            },
            {
                "PLAYER_NAME": "Player B1",
                "TEAM_NAME": "Team B",
                "MIN": 31,
                "PTS": 18,
                "REB": 8,
                "AST": 4,
                "USG_PCT": 0.22,
                "Date": "2025-01-02",
            },
        ]
        injury_rows = [
            {
                "Date": "2025-01-02",
                "PLAYER_NAME": "Player A1",
                "TEAM_NAME": "Team A",
                "STATUS": "out",
                "SOURCE": "primary",
            }
        ]
        lineup_rows = [
            {
                "Date": "2025-01-02",
                "PLAYER_NAME": "Player A1",
                "TEAM_NAME": "Team A",
                "STATUS": "confirmed",
                "SOURCE": "primary",
            }
        ]
        log_rows = [
            {
                "Date": "2025-01-02",
                "Primary_Success": 1,
                "Secondary_Success": 0,
                "Records": 1,
            }
        ]

        config = {
            "player-features": {
                "rolling_windows": [7, 14, 30],
                "defaults": {
                    "neutral_player_minutes": 16.0,
                    "neutral_player_usage": 0.18,
                    "neutral_team_minutes": 240.0,
                    "injury_weights": {
                        "out": 1.0,
                        "doubtful": 0.75,
                        "questionable": 0.5,
                        "probable": 0.25,
                        "active": 0.0,
                        "unknown": 0.5,
                    },
                },
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "player.sqlite"
            with sqlite3.connect(db_path) as con:
                pd.DataFrame(stats_rows).to_sql(
                    get_player_data.PLAYER_STATS_TABLE,
                    con,
                    if_exists="replace",
                    index=False,
                )
                pd.DataFrame(injury_rows).to_sql(
                    get_player_data.INJURY_TABLE,
                    con,
                    if_exists="replace",
                    index=False,
                )
                pd.DataFrame(lineup_rows).to_sql(
                    get_player_data.STARTING_LINEUPS_TABLE,
                    con,
                    if_exists="replace",
                    index=False,
                )
                pd.DataFrame(log_rows).to_sql(
                    get_player_data.INJURY_FETCH_LOG_TABLE,
                    con,
                    if_exists="replace",
                    index=False,
                )
                get_player_data.rebuild_team_features(con, config)
                features = pd.read_sql_query(
                    f'SELECT * FROM "{get_player_data.TEAM_FEATURES_TABLE}"',
                    con,
                )

        self.assertIn("INJ_AVAIL_MIN_WT", features.columns)
        self.assertIn("PLR_MIN_ROLL7", features.columns)
        self.assertIn("LINEUP_COVERAGE", features.columns)
        row = features[(features["Date"] == "2025-01-02") & (features["TEAM_NAME"] == "Team A")].iloc[0]
        self.assertGreater(float(row["INJ_OUT_MIN_WT"]), 0.0)
        self.assertEqual(float(row["INJ_COVERAGE"]), 1.0)
        self.assertEqual(float(row["LINEUP_CONFIRMED_STARTERS"]), 1.0)
        self.assertEqual(float(row["LINEUP_COVERAGE"]), 0.2)
