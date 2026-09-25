"""Days-rest feature computation for the prediction entry point.

``main.py`` feeds ``Days-Rest-Home`` and ``Days-Rest-Away`` to the models. Those
columns are built during training by
``src/Process-Data/Get_Odds_Data.py::append_game_rows``, which records, for each
game, the number of calendar days since that team last played:

    last_played = teams_last_played.get(team)
    if last_played is None:
        return 7
    return (date_pointer - last_played).days

This module mirrors that definition at prediction time. Four properties matter,
because the feature only carries signal if it means the same thing in training
and at inference:

* the lookup is **strictly before** the game, which in training falls out of
  ``append_game_rows`` updating ``teams_last_played[team] = date_pointer``
  straight after each game, so a team's own game on the game day can never be
  read back as the game before it;
* the difference is a plain calendar-date difference with **no day offset**;
* it is **not clamped** -- the shipped ``dataset_2012-26`` carries raw values up
  to 17;
* a team with no earlier game on record gets ``7``, which is the value training
  assigns to a team's first game of a season.

Known divergences, left alone rather than silently reconciled:

* the basis differs. Training derives rest from the odds table, so a game whose
  odds were never captured leaves the team's last-played stale and inflates the
  stored value. This module reads the schedule, so it always sees the real last
  game. Where odds coverage has holes, inference therefore reports slightly
  less rest than the label the model was fitted on.
* ``src/Process-Data/Add_Days_Rest.py`` defines the same column differently
  (no-history ``10``, clamped to ``9``). It rewrites the ``odds_<season>``
  tables, and ``Create_Games.select_odds_table`` only reads those when no
  ``odds_<season>_new`` table exists, so the effect is conditional. It is not
  part of the documented Get_Odds_Data -> Create_Games pipeline, and
  ``dataset_2012-26`` was built without it: it holds raw values above 9 and
  essentially no 10s, which ``Tests/test_days_rest.py`` asserts.
* the schedule CSV is timestamped in UTC while the training calendar is the
  sportsbook's ET date. The two do not line up: ``Data/nba-2025-UTC.csv``
  contains ten team-days that look like double headers -- a 00:00 or 00:30 UTC
  tip-off alongside a 23:00 one, on the same date -- which are really two
  consecutive ET nights, and the odds tables have no such repeat at all. For
  those games this module measures rest on the UTC grid and so disagrees with
  the ET grid the labels were built on. It predates this module; converting the
  schedule to ET would move inference *towards* the training calendar rather
  than away from it, but it changes model inputs and is left as a follow-up.
"""

import pandas as pd

# Value used when a team has no earlier game on record. Matches the season-opener
# default in Get_Odds_Data.append_game_rows.
DEFAULT_DAYS_REST = 7

HOME_TEAM_COLUMN = "Home Team"
AWAY_TEAM_COLUMN = "Away Team"
DATE_COLUMN = "Date"


def _normalised_dates(schedule_df):
    """Return the schedule's date column as normalised timestamps, NaT dropped."""
    dates = pd.to_datetime(schedule_df[DATE_COLUMN], errors="coerce")
    return dates.dt.normalize().dropna()


def days_rest_before(schedule_df, team, game_date):
    """Days since ``team`` last played, measured strictly before ``game_date``.

    Args:
        schedule_df: DataFrame with ``Home Team``/``Away Team``/``Date`` columns,
            as returned by ``main.load_schedule``, or ``None``. Dates must be
            timezone-naive; the schedule CSV is parsed without a timezone.
        team: Full team name, e.g. "Los Angeles Lakers". The string must match
            the schedule exactly, so the team-name conventions in
            ``Dictionaries.py`` and ``SbrOddsProvider.py`` have to agree with
            the CSV's spelling.
        game_date: Date or datetime of the game being predicted. Any time of day
            is ignored, so the result does not depend on when the script is run.

    Returns:
        int: Calendar days between the team's previous game and ``game_date``,
        or ``DEFAULT_DAYS_REST`` when the team has no earlier game on record.
        Rest is measured from strictly before the game day, so every game on the
        game day itself is ignored. That assumes no team plays twice on one
        date, which holds in the NBA and in the shipped odds tables; training
        would emit a 0 for the second game of a double header, a value this
        function cannot return.

    Raises:
        KeyError: If ``schedule_df`` lacks any of the three expected columns.
            Raised rather than defaulted, because silently returning a constant
            would feed the models a feature with no signal.
    """
    if schedule_df is None:
        return DEFAULT_DAYS_REST
    missing = {HOME_TEAM_COLUMN, AWAY_TEAM_COLUMN, DATE_COLUMN} - set(schedule_df.columns)
    if missing:
        raise KeyError(f"schedule is missing expected column(s): {sorted(missing)}")

    game_day = pd.Timestamp(game_date).normalize()

    team_games = schedule_df[
        (schedule_df[HOME_TEAM_COLUMN] == team) | (schedule_df[AWAY_TEAM_COLUMN] == team)
    ]
    if len(team_games) == 0:
        return DEFAULT_DAYS_REST

    played = _normalised_dates(team_games)
    previous_games = played[played < game_day]
    if len(previous_games) == 0:
        return DEFAULT_DAYS_REST

    last_played = previous_games.max()
    return int((game_day - last_played).days)
