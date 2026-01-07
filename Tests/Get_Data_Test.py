import importlib.util
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import pandas as pd

GET_DATA_PATH = Path(__file__).resolve().parents[1] / "src" / "Process-Data" / "Get_Data.py"
SPEC = importlib.util.spec_from_file_location("get_data", GET_DATA_PATH)
get_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(get_data)


class TestGetData(unittest.TestCase):

    def test_iter_dates_inclusive(self):
        dates = list(get_data.iter_dates(date(2025, 1, 1), date(2025, 1, 3)))
        self.assertEqual(dates, [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)])

    def test_select_current_season(self):
        config = {
            "get-data": {
                "2025-26": {
                    "start_date": "2025-10-01",
                    "end_date": "2026-06-01",
                    "start_year": "2025",
                    "end_year": "2026"
                }
            }
        }
        season_key, value, start_date, end_date = get_data.select_current_season(
            config, date(2025, 12, 1)
        )
        self.assertEqual(season_key, "2025-26")
        self.assertEqual(value["start_year"], "2025")
        self.assertEqual(start_date, date(2025, 10, 1))
        self.assertEqual(end_date, date(2026, 6, 1))

    def test_get_table_dates_ignores_non_dates(self):
        with sqlite3.connect(":memory:") as con:
            con.execute('CREATE TABLE "2025-01-01" (id INTEGER)')
            con.execute('CREATE TABLE "not_a_date" (id INTEGER)')
            table_dates = get_data.get_table_dates(con)
        self.assertEqual(table_dates, [date(2025, 1, 1)])

    def test_fetch_data_retries_until_non_empty(self):
        df_empty = pd.DataFrame(data={})
        df_ok = pd.DataFrame({"A": [1]})
        with mock.patch.object(get_data, "get_json_data", return_value={"resultSets": []}) as mock_json, \
             mock.patch.object(get_data, "to_data_frame", side_effect=[df_empty, df_ok]), \
             mock.patch.object(get_data.time, "sleep"):
            result = get_data.fetch_data("http://example.com", date(2025, 1, 1), "2025", "2025-26")
        self.assertFalse(result.empty)
        self.assertEqual(mock_json.call_count, 2)

    def test_main_fetches_only_new_dates(self):
        config = {
            "data_url": "http://example.com",
            "get-data": {
                "2025-26": {
                    "start_date": "2025-10-01",
                    "end_date": "2025-10-03",
                    "start_year": "2025",
                    "end_year": "2026"
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            with sqlite3.connect(db_path) as con:
                con.execute('CREATE TABLE "2025-10-01" (id INTEGER)')

            called_dates = []

            def fake_fetch_data(url, date_pointer, start_year, season_key):
                called_dates.append(date_pointer)
                return pd.DataFrame({"A": [1]})

            with mock.patch.object(get_data, "fetch_data", side_effect=fake_fetch_data), \
                 mock.patch.object(get_data.time, "sleep"):
                get_data.main(config=config, db_path=db_path, today=date(2025, 10, 3))

            self.assertEqual(called_dates, [date(2025, 10, 2), date(2025, 10, 3)])
