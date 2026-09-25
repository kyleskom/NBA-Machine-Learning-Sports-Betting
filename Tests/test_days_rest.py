import importlib.util
import sqlite3
import sys
import types
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.Utils.days_rest import days_rest_before

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "Data"
DATASET_PATH = DATA_DIR / "dataset.sqlite"

# The season-opener default, hard-coded rather than imported from the module
# under test, so that changing DEFAULT_DAYS_REST is caught here.
TRAINING_DEFAULT_DAYS_REST = 7

try:
    import sbrscrape  # noqa: F401
except ImportError:
    stub = types.ModuleType("sbrscrape")

    class Scoreboard:
        def __init__(self, *args, **kwargs):
            self.games = []

    stub.Scoreboard = Scoreboard
    sys.modules["sbrscrape"] = stub

GET_ODDS_PATH = REPO_ROOT / "src" / "Process-Data" / "Get_Odds_Data.py"
SPEC = importlib.util.spec_from_file_location("get_odds_data", GET_ODDS_PATH)
get_odds_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(get_odds_data)


def schedule(rows):
    """Build a schedule frame the way main.load_schedule() does."""
    return pd.DataFrame(rows, columns=["Match Number", "Home Team", "Away Team", "Date"])


def reference_days_rest(rows, team, game_day):
    """Independent, deliberately naive reference used to cross-check the helper.

    Plain Python over (home, away, date) tuples, sharing no code with the pandas
    implementation under test.
    """
    game_day = pd.Timestamp(game_day).date()
    played = {
        pd.Timestamp(row[2]).date()
        for row in rows
        if team in (row[0], row[1]) and pd.Timestamp(row[2]).date() < game_day
    }
    if not played:
        return TRAINING_DEFAULT_DAYS_REST
    return (game_day - max(played)).days


def load_newest_schedule():
    """Load the most recent committed schedule CSV, by latest game date.

    Derived from the files rather than hard-coded, so pointing the project at a
    new season does not break these tests.
    """
    frames = []
    for path in sorted(DATA_DIR.glob("nba-*-UTC.csv")):
        frame = pd.read_csv(path, parse_dates=["Date"], date_format="%d/%m/%Y %H:%M")
        frames.append((frame["Date"].max(), path, frame))
    if not frames:
        raise AssertionError(f"no schedule CSV found in {DATA_DIR}")
    return max(frames, key=lambda item: item[0])[2]


def pick_game_day(frame):
    """First day in the file with enough games and more than one rest value.

    Deterministic, and robust to a new season landing in a different CSV.
    """
    days = sorted(frame["Date"].dt.normalize().dropna().unique())
    for day in days:
        games = frame[frame["Date"].dt.normalize() == day]
        if len(games) < 5:
            continue
        teams = set(games["Home Team"]) | set(games["Away Team"])
        if len({days_rest_before(frame, team, day) for team in teams}) > 1:
            return pd.Timestamp(day)
    raise AssertionError("no game day in the schedule with varied rest values")


class TestDaysRest(unittest.TestCase):

    def test_counts_calendar_days_since_previous_game(self):
        sched = schedule([
            (1, "Alpha", "Beta", pd.Timestamp("2026-01-01 23:30")),
            (2, "Alpha", "Gamma", pd.Timestamp("2026-01-03 23:30")),
            (3, "Alpha", "Delta", pd.Timestamp("2026-01-05 23:30")),
        ])
        self.assertEqual(days_rest_before(sched, "Alpha", "2026-01-05"), 2)

    def test_ignores_the_team_s_own_game_on_the_game_day(self):
        """A team's game today is not the game before today.

        The old code compared with ``<=``, so once the script ran later than the
        scheduled tip-off it found today's game and reported 1 day of rest.
        """
        sched = schedule([
            (1, "Alpha", "Beta", pd.Timestamp("2026-01-01 23:30")),
            (2, "Alpha", "Gamma", pd.Timestamp("2026-01-05 23:30")),
        ])
        self.assertEqual(days_rest_before(sched, "Alpha", "2026-01-05 23:30"), 4)

    def test_does_not_apply_an_extra_day_offset(self):
        """Regression: inference added timedelta(days=1) that training never had."""
        sched = schedule([
            (1, "Alpha", "Beta", pd.Timestamp("2026-01-04 02:00")),
            (2, "Alpha", "Gamma", pd.Timestamp("2026-01-05 23:30")),
        ])
        self.assertEqual(days_rest_before(sched, "Alpha", "2026-01-05"), 1)

    def test_is_independent_of_time_of_day(self):
        """Regression: the old code compared a clock time against UTC tip-offs."""
        sched = schedule([
            (1, "Alpha", "Beta", pd.Timestamp("2026-01-01 23:30")),
            (2, "Alpha", "Gamma", pd.Timestamp("2026-01-05 23:30")),
        ])
        results = {
            days_rest_before(sched, "Alpha", f"2026-01-05 {hour:02d}:{minute:02d}")
            for hour, minute in ((0, 0), (9, 0), (12, 0), (18, 0), (23, 45))
        }
        self.assertEqual(results, {4})

    def test_matches_away_team_rows(self):
        sched = schedule([
            (1, "Beta", "Alpha", pd.Timestamp("2026-01-01 23:30")),
            (2, "Gamma", "Alpha", pd.Timestamp("2026-01-05 23:30")),
        ])
        self.assertEqual(days_rest_before(sched, "Alpha", "2026-01-05"), 4)

    def test_multiple_games_today_are_all_excluded(self):
        """Rest is measured from before the game day, never from within it.

        No same-day repeat team-games exist in the shipped odds tables, so the
        training pipeline never emitted a 0 and inference must not invent one.
        """
        sched = schedule([
            (1, "Alpha", "Beta", pd.Timestamp("2025-12-31 23:30")),
            (2, "Alpha", "Gamma", pd.Timestamp("2026-01-05 01:00")),
            (3, "Alpha", "Delta", pd.Timestamp("2026-01-05 23:30")),
        ])
        self.assertEqual(days_rest_before(sched, "Alpha", "2026-01-05"), 5)

    def test_long_rest_is_not_clamped(self):
        """Training data stores raw values (up to 17 seen), so do not cap at 9."""
        sched = schedule([
            (1, "Alpha", "Beta", pd.Timestamp("2025-12-25 23:30")),
            (2, "Alpha", "Gamma", pd.Timestamp("2026-01-05 23:30")),
        ])
        self.assertEqual(days_rest_before(sched, "Alpha", "2026-01-05"), 11)

    def test_long_rest_crossing_a_year_boundary(self):
        sched = schedule([
            (1, "Alpha", "Beta", pd.Timestamp("2025-12-31 23:30")),
            (2, "Alpha", "Gamma", pd.Timestamp("2026-01-02 23:30")),
        ])
        self.assertEqual(days_rest_before(sched, "Alpha", "2026-01-02"), 2)

    def test_unknown_team_gets_the_training_default(self):
        sched = schedule([(1, "Alpha", "Beta", pd.Timestamp("2026-01-01 23:30"))])
        self.assertEqual(days_rest_before(sched, "Zeta", "2026-01-05"), TRAINING_DEFAULT_DAYS_REST)

    def test_team_with_only_a_game_today_gets_the_default(self):
        sched = schedule([(1, "Alpha", "Beta", pd.Timestamp("2026-01-05 23:30"))])
        self.assertEqual(days_rest_before(sched, "Alpha", "2026-01-05"), TRAINING_DEFAULT_DAYS_REST)

    def test_unparseable_dates_are_ignored_but_good_rows_survive(self):
        """A junk date is dropped without taking the rest of the schedule with it."""
        sched = schedule([
            (1, "Alpha", "Beta", pd.Timestamp("2026-01-01 23:30")),
            (2, "Alpha", "Gamma", "not-a-date"),
        ])
        self.assertEqual(days_rest_before(sched, "Alpha", "2026-01-05"), 4)

    def test_accepts_string_dates_in_the_schedule(self):
        sched = schedule([
            (1, "Alpha", "Beta", "2026-01-01 23:30"),
            (2, "Alpha", "Gamma", "2026-01-05 23:30"),
        ])
        self.assertEqual(days_rest_before(sched, "Alpha", "2026-01-05"), 4)

    def test_accepts_datetime_like_game_date(self):
        sched = schedule([
            (1, "Alpha", "Beta", pd.Timestamp("2026-01-01 23:30")),
            (2, "Alpha", "Gamma", pd.Timestamp("2026-01-05 23:30")),
        ])
        self.assertEqual(days_rest_before(sched, "Alpha", datetime(2026, 1, 5, 19, 30)), 4)

    def test_none_schedule_falls_back_to_the_default(self):
        self.assertEqual(days_rest_before(None, "Alpha", "2026-01-05"), TRAINING_DEFAULT_DAYS_REST)

    def test_schedule_missing_a_column_raises(self):
        """A malformed schedule must fail loudly, not feed a constant feature."""
        sched = pd.DataFrame([["Alpha", "Beta"]], columns=["home", "away"])
        with self.assertRaises(KeyError):
            days_rest_before(sched, "Alpha", "2026-01-05")


class TestDaysRestMatchesTrainingImplementation(unittest.TestCase):
    """Pin this module to the implementation it claims to mirror.

    The README tells maintainers to change rest in both this module and
    Get_Odds_Data.append_game_rows. These tests fail if only one is changed.

    The games are taken from the odds tables, because those are what training
    actually consumed; replaying append_game_rows over them reproduces the
    labels the models were fitted on.
    """

    ODDS_DB_PATH = DATA_DIR / "OddsData.sqlite"
    TABLES = ["odds_2025-26", "2024-25", "odds_2023-24_new", "odds_2022-23_new"]

    @staticmethod
    def _fake_game(home, away):
        return {
            "home_team": home, "away_team": away,
            "total": {"sb": 220.0}, "away_spread": {"sb": 0.0},
            "home_ml": {"sb": -110}, "away_ml": {"sb": -110},
            "home_score": 110, "away_score": 105,
        }

    def test_season_opener_default_agrees_with_append_game_rows(self):
        rows = []
        get_odds_data.append_game_rows(
            rows, pd.Timestamp("2026-01-05").date(),
            self._fake_game("Alpha", "Beta"), "sb", {},
        )
        self.assertEqual(rows[0]["Days_Rest_Home"], TRAINING_DEFAULT_DAYS_REST)
        self.assertEqual(rows[0]["Days_Rest_Away"], TRAINING_DEFAULT_DAYS_REST)

    def test_matches_append_game_rows_on_every_training_game(self):
        with sqlite3.connect(self.ODDS_DB_PATH) as con:
            for table in self.TABLES:
                with self.subTest(table=table):
                    frame = pd.read_sql_query(
                        f'SELECT Date, Home, Away FROM "{table}" ORDER BY Date', con
                    )
                    rows = list(
                        frame[["Home", "Away", "Date"]].itertuples(index=False, name=None)
                    )
                    self.assertGreater(len(rows), 100, f"{table} looks empty")

                    # The schedule inference would see, built from the same games.
                    schedule_frame = pd.DataFrame({
                        "Home Team": [r[0] for r in rows],
                        "Away Team": [r[1] for r in rows],
                        "Date": pd.to_datetime([r[2] for r in rows], errors="coerce"),
                    })

                    recorded, teams_last_played = [], {}
                    for home, away, game_date in rows:
                        get_odds_data.append_game_rows(
                            recorded, pd.Timestamp(game_date).date(),
                            self._fake_game(home, away), "sb", teams_last_played,
                        )

                    for entry, (_, _, game_date) in zip(recorded, rows):
                        home, away = entry["Home"], entry["Away"]
                        with self.subTest(game=f"{home} v {away} on {game_date}"):
                            self.assertEqual(
                                days_rest_before(schedule_frame, home, game_date),
                                entry["Days_Rest_Home"],
                            )
                            self.assertEqual(
                                days_rest_before(schedule_frame, away, game_date),
                                entry["Days_Rest_Away"],
                            )


class TestDaysRestAgainstShippedSchedule(unittest.TestCase):
    """Cross-check the helper against the repository's own schedule fixture."""

    @classmethod
    def setUpClass(cls):
        cls.frame = load_newest_schedule()
        cls.rows = list(
            cls.frame[["Home Team", "Away Team", "Date"]].itertuples(index=False, name=None)
        )
        cls.game_day = pick_game_day(cls.frame)
        games = cls.frame[cls.frame["Date"].dt.normalize() == cls.game_day]
        cls.teams = sorted(set(games["Home Team"]) | set(games["Away Team"]))

    def test_matches_independent_reference_for_every_team(self):
        for team in self.teams:
            with self.subTest(team=team):
                self.assertEqual(
                    days_rest_before(self.frame, team, self.game_day),
                    reference_days_rest(self.rows, team, self.game_day),
                )

    def test_rest_varies_across_teams(self):
        values = {days_rest_before(self.frame, team, self.game_day) for team in self.teams}
        self.assertGreater(
            len(values), 1, "rest collapsed to a constant, so the feature carries no signal"
        )

    def test_rest_is_stable_across_the_whole_game_day(self):
        late = f"{self.game_day.date()} 23:45"
        for team in self.teams:
            with self.subTest(team=team):
                self.assertEqual(
                    days_rest_before(self.frame, team, self.game_day),
                    days_rest_before(self.frame, team, late),
                )


class TestTrainingDatasetUsesUnclampedRest(unittest.TestCase):
    """Pin the premise behind this module's "no clamping, default 7" contract.

    Add_Days_Rest.py would clamp Days_Rest_* to 9 and default an unknown team to
    10. If the shipped dataset were built through it, those signatures would be
    present and matching it would be the wrong target.
    """

    TABLE = "dataset_2012-26"

    def test_dataset_has_raw_values_above_nine(self):
        with sqlite3.connect(DATASET_PATH) as con:
            values = pd.read_sql_query(
                f'SELECT "Days-Rest-Home", "Days-Rest-Away" FROM "{self.TABLE}"', con
            ).values.ravel()
        self.assertGreater(
            values.max(), 9, "no value above 9, so the dataset may have been clamped"
        )

    def test_dataset_does_not_use_the_ten_default(self):
        with sqlite3.connect(DATASET_PATH) as con:
            values = pd.read_sql_query(
                f'SELECT "Days-Rest-Home", "Days-Rest-Away" FROM "{self.TABLE}"', con
            ).values.ravel()
        self.assertLess(
            (values == 10).sum(), 10, "many 10s, so the Add_Days_Rest default may be in use"
        )

    def test_dataset_has_no_zero_rest_rows(self):
        with sqlite3.connect(DATASET_PATH) as con:
            values = pd.read_sql_query(
                f'SELECT "Days-Rest-Home", "Days-Rest-Away" FROM "{self.TABLE}"', con
            ).values.ravel()
        self.assertEqual((values == 0).sum(), 0)


class TestScheduleCalendarDiffersFromTraining(unittest.TestCase):
    """Pin the UTC-vs-ET calendar divergence documented in days_rest.py.

    The schedule CSV is on a UTC grid and the odds tables on the sportsbook's ET
    date. The visible symptom is same-day team repeats in the CSV that the odds
    tables do not have. If that ever changes, the divergence note is stale and
    this test says so.
    """

    ODDS_DB_PATH = DATA_DIR / "OddsData.sqlite"

    def _same_day_repeats(self, frame, home_col, away_col):
        dates = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
        both = pd.concat([
            pd.DataFrame({"d": dates, "t": frame[home_col]}),
            pd.DataFrame({"d": dates, "t": frame[away_col]}),
        ]).dropna()
        counts = both.groupby(["d", "t"]).size()
        return counts[counts > 1]

    def test_schedule_has_repeats_that_the_odds_tables_do_not(self):
        repeats = self._same_day_repeats(load_newest_schedule(), "Home Team", "Away Team")
        self.assertGreater(
            len(repeats), 0,
            "no same-day repeats in the schedule any more; revisit the UTC/ET note",
        )
        with sqlite3.connect(self.ODDS_DB_PATH) as con:
            frame = pd.read_sql_query('SELECT Date, Home, Away FROM "odds_2025-26"', con)
        self.assertEqual(
            len(self._same_day_repeats(frame, "Home", "Away")), 0,
            "the odds table now has same-day repeats too; revisit the UTC/ET note",
        )


if __name__ == "__main__":
    unittest.main()
