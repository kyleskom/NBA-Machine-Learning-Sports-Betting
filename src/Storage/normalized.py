"""Normalized NBA game data with idempotent legacy migration helpers."""
from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re
import sqlite3

import pandas as pd

from src.Players.identity import team_id, team_name

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / 'Data' / 'NBAData.sqlite'


def canonical_path_for(source_path):
    return Path(source_path).resolve().parent / 'NBAData.sqlite'


def connect(path=DB_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute('PRAGMA foreign_keys=ON')
    initialize(con)
    return con


def initialize(con):
    con.execute('''CREATE TABLE IF NOT EXISTS team_snapshots (
        snapshot_date TEXT NOT NULL, team_id INTEGER NOT NULL, team_name TEXT NOT NULL,
        season TEXT NOT NULL, stats_json TEXT NOT NULL, source TEXT NOT NULL,
        PRIMARY KEY(snapshot_date, team_id))''')
    con.execute('CREATE INDEX IF NOT EXISTS team_snapshots_lookup ON team_snapshots(snapshot_date)')
    con.execute('''CREATE TABLE IF NOT EXISTS games (
        game_id TEXT PRIMARY KEY, game_date TEXT NOT NULL, season TEXT NOT NULL,
        home_team_id INTEGER NOT NULL, away_team_id INTEGER NOT NULL,
        home_team_name TEXT NOT NULL, away_team_name TEXT NOT NULL,
        points INTEGER, win_margin INTEGER, days_rest_home INTEGER, days_rest_away INTEGER,
        source TEXT NOT NULL, UNIQUE(game_date, home_team_id, away_team_id))''')
    con.execute('CREATE INDEX IF NOT EXISTS games_season_date ON games(season, game_date)')
    con.execute('''CREATE TABLE IF NOT EXISTS odds_quotes (
        game_id TEXT NOT NULL REFERENCES games(game_id), sportsbook TEXT NOT NULL,
        market TEXT NOT NULL, side TEXT NOT NULL, line REAL, price REAL,
        source TEXT NOT NULL, PRIMARY KEY(game_id, sportsbook, market, side))''')
    con.execute('CREATE INDEX IF NOT EXISTS odds_quotes_game ON odds_quotes(game_id, sportsbook)')


def season_for_date(value):
    day = pd.Timestamp(value).date()
    start_year = day.year if day.month >= 7 else day.year - 1
    return f'{start_year}-{str(start_year + 1)[-2:]}'


def _json_stats(row):
    values = {key: value for key, value in row.items() if key not in {'TEAM_ID', 'TEAM_NAME', 'Date'}}
    return json.dumps(values, default=lambda value: None if pd.isna(value) else value)


def save_team_snapshot(con, frame, snapshot_date, season=None, source='nba'):
    if frame.empty:
        return 0
    if not {'TEAM_ID', 'TEAM_NAME'}.issubset(frame.columns):
        return 0
    day = pd.Timestamp(snapshot_date).date().isoformat()
    season = season or season_for_date(day)
    rows = []
    for row in frame.to_dict(orient='records'):
        tid = team_id(row['TEAM_ID'])
        rows.append((day, tid, team_name(tid), season, _json_stats(row), source))
    with con:
        con.executemany('''INSERT INTO team_snapshots
            (snapshot_date,team_id,team_name,season,stats_json,source) VALUES (?,?,?,?,?,?)
            ON CONFLICT(snapshot_date,team_id) DO UPDATE SET
            team_name=excluded.team_name,season=excluded.season,stats_json=excluded.stats_json,source=excluded.source''', rows)
    return len(rows)


def load_latest_team_snapshot(con, game_date, max_age_days=7):
    target = pd.Timestamp(game_date).date()
    row = con.execute('SELECT MAX(snapshot_date) FROM team_snapshots WHERE snapshot_date < ?',
                      (target.isoformat(),)).fetchone()
    if not row or not row[0]:
        return None
    snapshot_date = pd.Timestamp(row[0]).date()
    if (target - snapshot_date).days > max_age_days:
        return None
    raw = con.execute('SELECT team_id,team_name,stats_json FROM team_snapshots WHERE snapshot_date=?',
                      (snapshot_date.isoformat(),)).fetchall()
    records = []
    for tid, name, stats in raw:
        record = json.loads(stats)
        record.update(TEAM_ID=tid, TEAM_NAME=name, Date=snapshot_date.isoformat())
        records.append(record)
    return pd.DataFrame(records)


def _game_id(day, home_id, away_id):
    key = f'{day}:{home_id}:{away_id}'.encode()
    return f'legacy:{hashlib.sha256(key).hexdigest()[:24]}'


def _number(value, kind=float):
    if value is None or pd.isna(value):
        return None
    try:
        return kind(value)
    except (TypeError, ValueError):
        return None


def save_odds_rows(con, rows, season, sportsbook='fanduel', source='sbr'):
    """Store one game record plus normalized moneyline, total, and spread quotes."""
    count = 0
    for row in rows:
        try:
            day = pd.Timestamp(row['Date']).date().isoformat()
            home_id, away_id = team_id(row['Home']), team_id(row['Away'])
        except (KeyError, ValueError, TypeError):
            continue
        gid = _game_id(day, home_id, away_id)
        game = (gid, day, season, home_id, away_id, team_name(home_id), team_name(away_id),
                _number(row.get('Points'), int), _number(row.get('Win_Margin'), int),
                _number(row.get('Days_Rest_Home'), int), _number(row.get('Days_Rest_Away'), int), source)
        quotes = [
            (gid, sportsbook, 'moneyline', 'home', None, _number(row.get('ML_Home')), source),
            (gid, sportsbook, 'moneyline', 'away', None, _number(row.get('ML_Away')), source),
            (gid, sportsbook, 'total', 'total', _number(row.get('OU')), None, source),
            (gid, sportsbook, 'spread', 'away', _number(row.get('Spread')), None, source),
        ]
        with con:
            con.execute('''INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(game_date,home_team_id,away_team_id) DO UPDATE SET
                season=excluded.season,home_team_name=excluded.home_team_name,
                away_team_name=excluded.away_team_name,points=excluded.points,
                win_margin=excluded.win_margin,days_rest_home=excluded.days_rest_home,
                days_rest_away=excluded.days_rest_away,source=excluded.source''', game)
            con.executemany('''INSERT INTO odds_quotes VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(game_id,sportsbook,market,side) DO UPDATE SET
                line=excluded.line,price=excluded.price,source=excluded.source''', quotes)
        count += 1
    return count


def load_games(con, season, sportsbook='fanduel'):
    query = '''SELECT g.game_date AS Date, g.home_team_name AS Home, g.away_team_name AS Away,
        MAX(CASE WHEN q.market='total' AND q.side='total' THEN q.line END) AS OU,
        MAX(CASE WHEN q.market='spread' AND q.side='away' THEN q.line END) AS Spread,
        MAX(CASE WHEN q.market='moneyline' AND q.side='home' THEN q.price END) AS ML_Home,
        MAX(CASE WHEN q.market='moneyline' AND q.side='away' THEN q.price END) AS ML_Away,
        g.points AS Points, g.win_margin AS Win_Margin,
        g.days_rest_home AS Days_Rest_Home, g.days_rest_away AS Days_Rest_Away
        FROM games g LEFT JOIN odds_quotes q ON q.game_id=g.game_id AND q.sportsbook=?
        WHERE g.season=? GROUP BY g.game_id ORDER BY g.game_date'''
    return pd.read_sql_query(query, con, params=[sportsbook, season])


_DAY_TABLE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
_ODDS_TABLE = re.compile(r'^(?:odds_)?(?P<season>\d{4}-\d{2})(?P<new>_new)?$')


def migrate_legacy_team_data(legacy_con, canonical_con):
    tables = [name for (name,) in legacy_con.execute("SELECT name FROM sqlite_master WHERE type='table'")
              if _DAY_TABLE.match(name)]
    count = 0
    for table in sorted(tables):
        frame = pd.read_sql_query(f'SELECT * FROM "{table}"', legacy_con)
        count += save_team_snapshot(canonical_con, frame, table, source='legacy-team-table')
    return count


def _preferred_odds_tables(legacy_con):
    selected = {}
    for (table,) in legacy_con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        match = _ODDS_TABLE.match(table)
        if not match:
            continue
        priority = (2 if table.startswith('odds_') else 0) + (1 if match.group('new') else 0)
        season = match.group('season')
        if season not in selected or priority > selected[season][0]:
            selected[season] = (priority, table)
    return {season: table for season, (_, table) in selected.items()}


def migrate_legacy_odds_data(legacy_con, canonical_con, sportsbook='fanduel'):
    count = 0
    for season, table in _preferred_odds_tables(legacy_con).items():
        frame = pd.read_sql_query(f'SELECT * FROM "{table}"', legacy_con)
        count += save_odds_rows(canonical_con, frame.to_dict(orient='records'), season, sportsbook,
                                source=f'legacy:{table}')
    return count
