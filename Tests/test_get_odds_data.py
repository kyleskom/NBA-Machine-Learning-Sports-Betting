import importlib.util
import sqlite3
import tempfile
import types
import sys
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest import mock

import pandas as pd

try:
    import sbrscrape  # noqa: F401
except ImportError:
    stub = types.ModuleType("sbrscrape")

    class Scoreboard:
        def __init__(self, *args, **kwargs):
            self.games = []

    stub.Scoreboard = Scoreboard
    sys.modules["sbrscrape"] = stub

GET_ODDS_PATH = Path(__file__).resolve().parents[1] / "src" / "Process-Data" / "Get_Odds_Data.py"
SPEC = importlib.util.spec_from_file_location("get_odds_data", GET_ODDS_PATH)
get_odds_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(get_odds_data)


class DummyScoreboard:
    def __init__(self, games):
        self.games = games


def build_game():
    return {
        "home_team": "Team A",
        "away_team": "Team B",
        "total": {"fanduel": 210.5},
        "away_spread": {"fanduel": -3.5},
        "home_ml": {"fanduel": -150},
        "away_ml": {"fanduel": 130},
        "away_score": 100,
        "home_score": 110,
    }


def build_row(date_value):
    return {
        "Date": date_value,
        "Home": "Team A",
        "Away": "Team B",
        "OU": 210.5,
        "Spread": -3.5,
        "ML_Home": -150,
        "ML_Away": 130,
        "Points": 210,
        "Win_Margin": 10,
        "Days_Rest_Home": 2,
        "Days_Rest_Away": 7,
    }


class TestGetOddsData(unittest.TestCase):

    def test_iter_dates_inclusive(self):
        dates = list(get_odds_data.iter_dates(date(2025, 1, 1), date(2025, 1, 3)))
        self.assertEqual(dates, [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)])

    def test_parse_date_value(self):
        self.assertEqual(get_odds_data.parse_date_value("2025-01-02"), date(2025, 1, 2))
        self.assertEqual(get_odds_data.parse_date_value("2025-01-03 00:00:00"), date(2025, 1, 3))
        self.assertEqual(get_odds_data.parse_date_value(datetime(2025, 1, 4, 5, 6, 7)), date(2025, 1, 4))
        self.assertIsNone(get_odds_data.parse_date_value("not-a-date"))

    def test_select_current_season(self):
        config = {
            "get-odds-data": {
                "2025-26": {
                    "start_date": "2025-10-01",
                    "end_date": "2026-06-01",
                    "start_year": "2025",
                    "end_year": "2026",
                }
            }
        }
        season_key, value, start_date, end_date = get_odds_data.select_current_season(
            config, date(2025, 12, 1)
        )
        self.assertEqual(season_key, "2025-26")
        self.assertEqual(value["start_year"], "2025")
        self.assertEqual(start_date, date(2025, 10, 1))
        self.assertEqual(end_date, date(2026, 6, 1))

    def test_get_existing_dates(self):
        with sqlite3.connect(":memory:") as con:
            con.execute('CREATE TABLE "2025-26" (Date TEXT)')
            con.execute('INSERT INTO "2025-26" (Date) VALUES (?)', ("2025-10-01",))
            con.execute('INSERT INTO "2025-26" (Date) VALUES (?)', ("2025-10-02 00:00:00",))
            dates = get_odds_data.get_existing_dates(con, "2025-26")
        self.assertEqual(dates, {date(2025, 10, 1), date(2025, 10, 2)})

    def test_get_teams_last_played(self):
        with sqlite3.connect(":memory:") as con:
            con.execute('CREATE TABLE "2025-26" (Date TEXT, Home TEXT, Away TEXT)')
            con.execute(
                'INSERT INTO "2025-26" (Date, Home, Away) VALUES (?, ?, ?)',
                ("2025-10-01", "Team A", "Team B"),
            )
            con.execute(
                'INSERT INTO "2025-26" (Date, Home, Away) VALUES (?, ?, ?)',
                ("2025-10-03", "Team A", "Team C"),
            )
            con.execute(
                'INSERT INTO "2025-26" (Date, Home, Away) VALUES (?, ?, ?)',
                ("2025-10-02", "Team B", "Team C"),
            )
            teams_last_played = get_odds_data.get_teams_last_played(con, "2025-26", date(2025, 10, 4))
        self.assertEqual(teams_last_played["Team A"], date(2025, 10, 3))
        self.assertEqual(teams_last_played["Team B"], date(2025, 10, 2))
        self.assertEqual(teams_last_played["Team C"], date(2025, 10, 3))

    def test_append_game_rows_updates_days_rest(self):
        game_rows = []
        teams_last_played = {"Team A": date(2025, 1, 1)}
        get_odds_data.append_game_rows(
            game_rows,
            date(2025, 1, 3),
            build_game(),
            "fanduel",
            teams_last_played,
        )
        self.assertEqual(len(game_rows), 1)
        self.assertEqual(game_rows[0]["Days_Rest_Home"], 2)
        self.assertEqual(game_rows[0]["Days_Rest_Away"], 7)
        self.assertEqual(teams_last_played["Team A"], date(2025, 1, 3))
        self.assertEqual(teams_last_played["Team B"], date(2025, 1, 3))

    def test_collect_odds_for_dates(self):
        with mock.patch.object(get_odds_data, "fetch_scoreboard", return_value=DummyScoreboard([build_game()])), \
             mock.patch.object(get_odds_data.time, "sleep"):
            rows = get_odds_data.collect_odds_for_dates(
                [date(2025, 1, 1)],
                "fanduel",
                {},
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Home"], "Team A")

    def test_backfill_season_appends_missing_dates(self):
        config_value = {
            "start_date": "2025-10-01",
            "end_date": "2025-10-02",
            "start_year": "2025",
            "end_year": "2026",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "odds.sqlite"
            with sqlite3.connect(db_path) as con:
                pd.DataFrame([build_row("2025-10-01")]).to_sql(
                    "2025-26", con, if_exists="replace", index=False
                )
                with mock.patch.object(
                    get_odds_data,
                    "fetch_scoreboard",
                    side_effect=lambda d: DummyScoreboard([build_game()]) if d == date(2025, 10, 2) else None,
                ), mock.patch.object(get_odds_data.time, "sleep"):
                    get_odds_data.backfill_season(
                        con,
                        "2025-26",
                        config_value,
                        "fanduel",
                        date(2025, 10, 3),
                    )
                df = pd.read_sql_query('SELECT * FROM "2025-26"', con)
        self.assertEqual(len(df.index), 2)
