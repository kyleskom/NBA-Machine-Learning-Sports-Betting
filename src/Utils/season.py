"""Season-aware URLs and paths shared by live prediction code."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]


def as_date(value):
    if isinstance(value, datetime):
        return value.date()
    if hasattr(value, "date"):
        return value.date()
    return value


def runtime_season(config, today=None):
    """Choose the active season, or the latest season already started."""
    today = as_date(today or date.today())
    candidates = []
    for season, value in config.get("get-data", {}).items():
        start = datetime.strptime(value["start_date"], "%Y-%m-%d").date()
        end = datetime.strptime(value["end_date"], "%Y-%m-%d").date()
        if start <= today <= end:
            return season
        if start <= today:
            candidates.append((start, season))
    if not candidates:
        raise ValueError("No configured NBA season has started yet")
    return max(candidates)[1]


def scores_url(season):
    """NBA's current scoreboard is independent of the season start year."""
    return "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"


def live_team_stats_url(config, season, as_of):
    """Use the collector's configured endpoint for the given season and cutoff."""
    cutoff = as_date(as_of)
    start_year = int(season.split("-", 1)[0])
    return config["data_url"].format(10, 1, start_year, cutoff.year, season)


def schedule_path(season):
    start_year = season.split("-", 1)[0]
    return BASE_DIR / "Data" / f"nba-{start_year}-UTC.csv"
