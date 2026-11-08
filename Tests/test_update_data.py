import importlib.util
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace


UPDATE_PATH = Path(__file__).resolve().parents[1] / "src" / "Process-Data" / "Update_Data.py"
SPEC = importlib.util.spec_from_file_location("update_data", UPDATE_PATH)
update_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(update_data)


class TestUpdateData(unittest.TestCase):
    def test_target_season_uses_latest_completed_season_in_offseason(self):
        config = {"get-data": {
            "2025-26": {"start_date": "2025-10-21", "end_date": "2026-06-21"},
            "2026-27": {"start_date": "2026-10-20", "end_date": "2027-06-30"},
        }}
        self.assertEqual(update_data.target_season(config, date(2026, 9, 18)), "2025-26")
        self.assertEqual(update_data.target_season(config, date(2026, 10, 20)), "2026-27")

    def test_update_runs_collectors_then_rebuilds_dataset(self):
        calls = []
        modules = {
            "team": SimpleNamespace(main=lambda **kwargs: calls.append(("team", kwargs))),
            "odds": SimpleNamespace(main=lambda **kwargs: calls.append(("odds", kwargs))),
            "player": SimpleNamespace(main=lambda args: calls.append(("player", args)) or 0),
            "dataset": SimpleNamespace(main=lambda: calls.append(("dataset",))),
        }
        status = update_data.update(["2025-26"], modules=modules, output=lambda _: None)
        self.assertEqual(status, 0)
        self.assertEqual(calls, [
            ("team", {"backfill": True, "season": "2025-26"}),
            ("odds", {"sportsbook": "fanduel", "backfill": True, "season": "2025-26"}),
            ("dataset",),
        ])
