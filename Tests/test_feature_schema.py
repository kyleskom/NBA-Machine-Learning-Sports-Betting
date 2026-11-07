"""Dependency-free schema and pre-game snapshot regressions.

Run: python3 -m unittest discover -s tests -p test_feature_schema.py -v
"""
import ast
from contextlib import closing
import sqlite3
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
import tomllib

from src.Utils.features import (
    MATCH_FEATURES, REST_FEATURES, XGB_ML_FEATURES, NN_ML_FEATURES,
    XGB_UO_FEATURES, NN_UO_FEATURES, select_features,
)
from src.Utils.snapshots import calendar_date, days_rest, snapshot_dates, select_snapshot_date

ROOT = Path(__file__).resolve().parents[1]


class Columns(list):
    @property
    def is_unique(self):
        return len(self) == len(set(self))


class NamedFrame:
    """Minimal .loc protocol to test named selection without pandas."""
    def __init__(self, columns):
        self.columns = Columns(columns)
        self.loc = self

    def __getitem__(self, key):
        return NamedFrame(key[1])

    def copy(self):
        return NamedFrame(self.columns)


class FeatureSchemaTests(unittest.TestCase):
    def test_training_orders_from_sqlite(self):
        with closing(sqlite3.connect(f"file:{ROOT / 'Data/dataset.sqlite'}?mode=ro", uri=True)) as con:
            schema = [r[1] for r in con.execute("pragma table_info('dataset_2012-26')")]
        self.assertTrue(schema)
        excluded = {"index", "Score", "Home-Team-Win", "TEAM_NAME", "Date",
                    "index.1", "TEAM_NAME.1", "Date.1", "OU-Cover", "OU"}
        legacy_ml = tuple(c for c in schema if c not in excluded)
        legacy_xgb_uo = tuple(c for c in schema if c not in excluded - {"OU"})
        self.assertEqual(legacy_ml, XGB_ML_FEATURES)
        self.assertEqual(legacy_ml, NN_ML_FEATURES)
        self.assertEqual(legacy_xgb_uo, XGB_UO_FEATURES)
        self.assertEqual(legacy_ml + ("OU",), NN_UO_FEATURES)
        self.assertEqual(len(MATCH_FEATURES), 104)
        self.assertEqual(XGB_UO_FEATURES[-3:], ("OU",) + REST_FEATURES)
        self.assertEqual(NN_UO_FEATURES[-3:], REST_FEATURES + ("OU",))
        for features in (XGB_ML_FEATURES, NN_ML_FEATURES, XGB_UO_FEATURES, NN_UO_FEATURES):
            self.assertTrue(set(features) <= set(schema))
            frame = NamedFrame(reversed(features))
            self.assertEqual(tuple(select_features(frame, features).columns), features)

    def test_missing_and_duplicate_columns_fail(self):
        with self.assertRaisesRegex(ValueError, "Missing required feature columns: OU"):
            select_features(NamedFrame(XGB_ML_FEATURES), XGB_UO_FEATURES)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            select_features(NamedFrame(["OU", "OU"]), ("OU",))

    def test_live_url_cutoff(self):
        # Inspect/execute just the URL expression, avoiding imports of ML libraries.
        tree = ast.parse((ROOT / "main.py").read_text())
        template = next(ast.literal_eval(n.value) for n in tree.body
                        if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "DATA_URL_TEMPLATE" for t in n.targets))
        format_call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                           and isinstance(n.func, ast.Attribute) and n.func.attr == "format"
                           and isinstance(n.func.value, ast.Name) and n.func.value.id == "DATA_URL_TEMPLATE")
        game_date = date(2026, 1, 7)
        url = eval(compile(ast.Expression(format_call), "main.py", "eval"),
                   {"DATA_URL_TEMPLATE": template, "game_date": game_date,
                    "timedelta": timedelta, "season_string": lambda d: "2025-26"})
        self.assertIn("DateTo=01%2F06%2F2026&", url)
        self.assertIn("Season=2025-26&", url)

    def test_snapshot_boundaries(self):
        dates = ("2025-06-01", "2025-10-21", "2025-10-23")
        self.assertIsNone(select_snapshot_date(dates, "2025-10-21", "2025-10-21"))
        self.assertEqual(select_snapshot_date(dates, "2025-10-23", "2025-10-21"), "2025-10-21")
        self.assertEqual(select_snapshot_date(dates, "2025-10-24", "2025-10-21"), "2025-10-23")

    def test_timestamp_dates_never_choose_same_day(self):
        dates = ("2025-10-21", "2026-01-06", "2026-01-07")
        for game_date in ("2026-01-07", "2026-01-07 00:00:00",
                          "2026-01-07T23:59:59-05:00", date(2026, 1, 7),
                          datetime(2026, 1, 7, 23, 59)):
            with self.subTest(game_date=game_date):
                self.assertEqual(select_snapshot_date(dates, game_date, "2025-10-21"),
                                 "2026-01-06")
                self.assertEqual(calendar_date(game_date), date(2026, 1, 7))
        self.assertIsNone(select_snapshot_date(dates, "2025-10-21 12:00:00", "2025-10-21"))
        for invalid in (None, "not a date", "2026-02-30", "2026-01-07 rubbish"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                select_snapshot_date(dates, invalid, "2025-10-21")

    def test_create_games_normalizes_timestamp_strings(self):
        tree = ast.parse((ROOT / "src/Process-Data/Create_Games.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "normalize_date")
        env = {"calendar_date": calendar_date}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "Create_Games.py", "exec"), env)
        self.assertEqual(env["normalize_date"]("2026-01-07 23:59:59"), "2026-01-07")
        self.assertEqual(env["normalize_date"](date(2026, 1, 7)), "2026-01-07")
        with self.assertRaises(ValueError):
            env["normalize_date"]("invalid")

    def test_keras_backend_and_loader_fallbacks_without_ml_imports(self):
        # Exercise compatibility branching using fake backends, not real TF weights.
        import runpy
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        path = str(ROOT / "src/Utils/keras_compat.py")
        native = SimpleNamespace(models=SimpleNamespace(load_model=Mock(return_value="native model")))
        preferred = SimpleNamespace(models=SimpleNamespace(load_model=Mock(return_value="preferred model")))
        with patch.dict("sys.modules", {"tf_keras": preferred,
                                       "tensorflow": SimpleNamespace(keras=native)}):
            env = runpy.run_path(path)
            self.assertIs(env["keras"], preferred)
            self.assertEqual(env["load_model"]("saved.keras"), "preferred model")
            preferred.models.load_model.side_effect = ValueError("new archive")
            self.assertEqual(env["load_model"]("new.keras"), "native model")
        with patch.dict("sys.modules", {"tf_keras": None,
                                       "tensorflow": SimpleNamespace(keras=native)}):
            env = runpy.run_path(path)
            self.assertIs(env["keras"], native)
            self.assertEqual(env["load_model"]("saved.keras"), "native model")

    def test_rest_uses_calendar_days_and_excludes_game_day(self):
        schedule = ["2026-01-05 23:30:00", "2026-01-07 00:00:00", "2026-01-08"]
        self.assertEqual(days_rest(schedule, "2026-01-07"), 2)
        self.assertEqual(days_rest([datetime(2026, 1, 6, 23, 59)], date(2026, 1, 7)), 1)
        self.assertEqual(days_rest(["2026-01-07 00:00:00"], date(2026, 1, 7)), 7)
        self.assertEqual(days_rest([], date(2026, 1, 7)), 7)
        self.assertEqual(days_rest(["2025-12-31 23:59:00"], date(2026, 1, 1)), 1)

    def test_historical_no_odds_exits_before_fetch(self):
        from types import SimpleNamespace
        tree = ast.parse((ROOT / "main.py").read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        messages = []
        env = {"date": date, "PREDICTION_MARKETS": {"kalshi", "polymarket"},
               "today_et": lambda: date(2026, 10, 26),
               "print_devices": lambda: None, "print": lambda *args: messages.append(str(args[0]))}
        exec(compile(ast.Module(body=[main], type_ignores=[]), "main.py", "exec"), env)
        env["main"](SimpleNamespace(date="2026-01-07", markets=False, odds=None))
        self.assertTrue(any("requires -odds" in m for m in messages))

    def test_explicit_date_provider_results_and_sportsbook_rejection(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        tree = ast.parse((ROOT / "main.py").read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        for requested_date in ("2026-01-07", "2026-10-26"):
            for venue in ("kalshi", "polymarket"):
                for empty_result in ({}, None):
                    with self.subTest(date=requested_date, venue=venue, result=empty_result):
                        provider = Mock()
                        provider.return_value.get_odds.return_value = empty_result
                        resolver = Mock(side_effect=AssertionError("must stop before resolving games"))
                        messages = []
                        env = {"date": date, "today_et": lambda: date(2026, 10, 26),
                               "PREDICTION_MARKETS": {"kalshi", "polymarket"},
                               "print_devices": lambda: None,
                               "PredictionMarketOddsProvider": provider, "resolve_games": resolver,
                               "print": lambda *args: messages.append(str(args[0]))}
                        exec(compile(ast.Module(body=[main], type_ignores=[]), "main.py", "exec"), env)
                        env["main"](SimpleNamespace(date=requested_date, markets=False, odds=venue))
                        provider.assert_called_once_with(venue=venue, date=date.fromisoformat(requested_date))
                        resolver.assert_not_called()
                        self.assertTrue(any("No date-specific odds returned" in m for m in messages))
        # A historical sportsbook source must be rejected before its undated provider is called.
        sportsbook = Mock(side_effect=AssertionError("must not request undated odds"))
        env["SbrOddsProvider"] = sportsbook
        env["main"](SimpleNamespace(date="2026-01-07", markets=False, odds="fanduel"))
        sportsbook.assert_not_called()
        self.assertTrue(any("sportsbook odds are current-only" in m for m in messages))
        # Valid dated results may proceed, but the resolver must still have fallback disabled.
        valid_odds = {"Home:Away": {"under_over_odds": 220}}
        provider.return_value.get_odds.return_value = valid_odds
        resolver.side_effect = None
        resolver.return_value = (None, None)  # Stop before stats/model execution.
        env["main"](SimpleNamespace(date="2026-01-07", markets=False, odds="polymarket"))
        resolver.assert_called_once_with(valid_odds, "polymarket", allow_today_fallback=False)

    def test_resolver_never_fetches_today_when_fallback_disabled(self):
        from unittest.mock import Mock
        tree = ast.parse((ROOT / "main.py").read_text())
        resolver = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "resolve_games")
        today_fetch = Mock(side_effect=AssertionError("today's schedule must not be fetched"))
        messages = []
        env = {"get_todays_games_json": today_fetch,
               "print": lambda *args: messages.append(str(args[0]))}
        exec(compile(ast.Module(body=[resolver], type_ignores=[]), "main.py", "exec"), env)
        self.assertEqual(env["resolve_games"]({}, "kalshi", allow_today_fallback=False), (None, None))
        today_fetch.assert_not_called()
        self.assertTrue(any("refusing to fall back" in m for m in messages))

    def test_training_defaults_and_notebook(self):
        import json
        for name in ("XGBoost_Model_ML", "XGBoost_Model_UO", "NN_Model_ML", "NN_Model_UO"):
            source = (ROOT / "src/Train-Models" / f"{name}.py").read_text()
            tree = ast.parse(source)
            defaults = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "DEFAULT_DATASET" for t in n.targets)]
            self.assertEqual(defaults, ["dataset_2012-27"])
            if name.startswith("NN"):
                self.assertNotIn("optimizers.legacy", source)
        notebook = json.loads((ROOT / "ColabNotebook.ipynb").read_text())
        sources = "\n".join(c["source"] if isinstance(c["source"], str) else "".join(c["source"])
                            for c in notebook["cells"])
        for name in ("XGBoost_Model_ML", "XGBoost_Model_UO", "NN_Model_ML", "NN_Model_UO"):
            self.assertIn(f"python -m {name} --dataset dataset_2012-27", sources)

    def test_real_game_snapshots(self):
        config = tomllib.loads((ROOT / "config.toml").read_text())
        checked = 0
        with closing(sqlite3.connect(f"file:{ROOT / 'Data/TeamData.sqlite'}?mode=ro", uri=True)) as teams, \
             closing(sqlite3.connect(f"file:{ROOT / 'Data/OddsData.sqlite'}?mode=ro", uri=True)) as odds:
            dates = snapshot_dates(teams)
            self.assertGreater(len(dates), 100)
            tables = {r[0] for r in odds.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for season, values in config["create-games"].items():
                candidates = [f"odds_{season}_new", f"odds_{season}", f"{season}_new", season]
                table = next((c for c in candidates if c in tables), None)
                if table is None:
                    continue
                start = date.fromisoformat(values["start_date"]).isoformat()
                for (game_date,) in odds.execute(f'SELECT DISTINCT Date FROM "{table}" ORDER BY Date'):
                    chosen = select_snapshot_date(dates, game_date, start)
                    if chosen is not None:
                        self.assertLess(chosen, game_date)
                        self.assertGreaterEqual(chosen, start)
                        checked += 1
            self.assertEqual(select_snapshot_date(dates, "2026-01-07", "2025-10-21"), "2026-01-06")
        self.assertGreater(checked, 100)


if __name__ == "__main__":
    unittest.main()
