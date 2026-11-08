"""Migrate legacy daily/season SQLite tables into Data/NBAData.sqlite."""
import argparse
import sqlite3
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from src.Storage.normalized import connect, migrate_legacy_odds_data, migrate_legacy_team_data


def main(team_db=BASE_DIR / 'Data' / 'TeamData.sqlite', odds_db=BASE_DIR / 'Data' / 'OddsData.sqlite',
         output=BASE_DIR / 'Data' / 'NBAData.sqlite', sportsbook='fanduel'):
    with sqlite3.connect(team_db) as legacy_teams, sqlite3.connect(odds_db) as legacy_odds, connect(output) as canonical:
        snapshots = migrate_legacy_team_data(legacy_teams, canonical)
        games = migrate_legacy_odds_data(legacy_odds, canonical, sportsbook)
    print(f'Migrated {snapshots} team snapshots and {games} games into {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--team-db', type=Path, default=BASE_DIR / 'Data' / 'TeamData.sqlite')
    parser.add_argument('--odds-db', type=Path, default=BASE_DIR / 'Data' / 'OddsData.sqlite')
    parser.add_argument('--output', type=Path, default=BASE_DIR / 'Data' / 'NBAData.sqlite')
    parser.add_argument('--sportsbook', default='fanduel')
    args = parser.parse_args()
    main(args.team_db, args.odds_db, args.output, args.sportsbook)
