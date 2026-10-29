"""Refresh the NBA data pipeline with one command.

The default run repairs the active season, or the most recently completed
season during the offseason, refreshes live player reports, and rebuilds the
player-aware training dataset.  Use --all-history to repair every configured
season instead.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import importlib.util
from pathlib import Path
import sys

import toml


BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config.toml"
PROCESS_DATA_DIR = Path(__file__).resolve().parent


def load_script(filename, module_name):
    """Load an existing collector from this hyphenated directory."""
    spec = importlib.util.spec_from_file_location(module_name, PROCESS_DATA_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configured_seasons(config, section):
    return config.get(section, {})


def target_season(config, today=None):
    """Return the season underway today, or the latest already-started season."""
    today = today or date.today()
    seasons = configured_seasons(config, "get-data")
    candidates = []
    for key, value in seasons.items():
        start = datetime.strptime(value["start_date"], "%Y-%m-%d").date()
        end = datetime.strptime(value["end_date"], "%Y-%m-%d").date()
        if start <= today <= end:
            return key
        if start <= today:
            candidates.append((start, key))
    if not candidates:
        raise ValueError("No configured NBA season has started yet")
    return max(candidates)[1]


def update(seasons, sportsbook="fanduel", rebuild_dataset=True, reports_only=False,
           modules=None, output=print):
    """Run collectors in dependency order.  Injection keeps this orchestration testable."""
    modules = modules or {
        "team": load_script("Get_Data.py", "update_team_data"),
        "odds": load_script("Get_Odds_Data.py", "update_odds_data"),
        "player": load_script("Get_Player_Data.py", "update_player_data"),
        "consensus": load_script("Import_Consensus_Odds.py", "update_consensus_odds"),
        "dataset": load_script("Create_Games.py", "update_create_games"),
    }
    failed = False
    if reports_only:
        output("Refreshing live injury, lineup, and roster reports...")
        return 1 if modules["player"].main(["--reports-only"]) else 0
    for season in seasons:
        output(f"Updating team data for {season}...")
        modules["team"].main(backfill=True, season=season)
        output(f"Updating {sportsbook} odds for {season}...")
        modules["odds"].main(sportsbook=sportsbook, backfill=True, season=season)
        output(f"Updating player logs for {season} and live reports...")
        status = modules["player"].main(["--season", season])
        failed |= bool(status)
        if "consensus" in modules:
            output(f"Importing configured consensus market data for {season}...")
            status = modules["consensus"].main(season)
            failed |= bool(status)
    if rebuild_dataset and not reports_only:
        output("Rebuilding the player-aware training dataset...")
        modules["dataset"].main()
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", action="append", help="Season to update, e.g. 2025-26. Repeat as needed.")
    parser.add_argument("--all-history", action="store_true",
                        help="Repair missing data for every configured season.")
    parser.add_argument("--sportsbook", default="fanduel", help="Sportsbook used for historical odds.")
    parser.add_argument("--reports-only", action="store_true", help="Refresh live availability reports only.")
    parser.add_argument("--skip-dataset", action="store_true", help="Do not rebuild the training dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Show the chosen update scope without collecting data.")
    args = parser.parse_args(argv)
    config = toml.load(CONFIG_PATH)
    seasons = args.season or (
        list(configured_seasons(config, "get-data")) if args.all_history else [target_season(config)]
    )
    unknown = [season for season in seasons if season not in configured_seasons(config, "get-data")]
    if unknown:
        parser.error("Unknown configured season(s): " + ", ".join(unknown))
    if args.dry_run:
        print("Would update:", ", ".join(seasons))
        print("Sportsbook:", args.sportsbook)
        print("Dataset rebuild:", not args.skip_dataset and not args.reports_only)
        return 0
    return update(seasons, args.sportsbook, not args.skip_dataset, args.reports_only)


if __name__ == "__main__":
    raise SystemExit(main())
