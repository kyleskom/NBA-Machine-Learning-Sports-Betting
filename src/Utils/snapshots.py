"""Pre-game snapshot selection, independent of pandas/numpy."""
from bisect import bisect_left
from datetime import date, datetime


def calendar_date(value):
    """Parse an ISO date/timestamp or date-like object, preserving its calendar day.

    Constructing a plain date also rejects pandas NaT without importing pandas.
    """
    try:
        if isinstance(value, date):
            return date(value.year, value.month, value.day)
        if isinstance(value, str):
            text = value.strip()
            try:
                return date.fromisoformat(text)
            except ValueError:
                return datetime.fromisoformat(text).date()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Unparseable calendar date: {value!r}") from exc
    raise ValueError(f"Unparseable calendar date: {value!r}")


def days_rest(previous_dates, game_date):
    """Calendar days since the last strictly earlier game, or 7 if none."""
    game_day = calendar_date(game_date)
    earlier = [day for value in previous_dates
               if (day := calendar_date(value)) < game_day]
    return (game_day - max(earlier)).days if earlier else 7


def snapshot_dates(connection):
    """Read the table inventory once; ignore non-ISO-date tables."""
    names = []
    for (name,) in connection.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        try:
            parsed = date.fromisoformat(name)
        except ValueError:
            continue
        if parsed.isoformat() == name:
            names.append(name)
    return tuple(sorted(names))


def select_snapshot_date(sorted_dates, game_date, season_start):
    """Latest snapshot in [season_start, game_date), or None on opening day."""
    game_date = calendar_date(game_date).isoformat()
    season_start = calendar_date(season_start).isoformat()
    index = bisect_left(sorted_dates, game_date) - 1
    if index >= 0 and sorted_dates[index] >= season_start:
        return sorted_dates[index]
    return None
