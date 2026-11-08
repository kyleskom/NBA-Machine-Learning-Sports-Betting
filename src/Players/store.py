"""Transactional player histories and immutable report snapshots."""
import json
import sqlite3
import hashlib
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / 'Data' / 'PlayerData.sqlite'
NUMERIC = ['MIN','PTS','REB','AST','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','TOV','STL','BLK','PF','PLUS_MINUS']
DETAILS = ['GAME_DATETIME','OPPONENT_TEAM_ID','HOME_AWAY','STARTER','DID_NOT_PLAY',
           'DNP_REASON','POSITION','MIN_IMPUTED','IDENTITY_RECOVERED','MIN_SCALE']
LOG_COLUMNS = ['GAME_ID','PLAYER_ID','TEAM_ID','TEAM_NAME','PLAYER_NAME','GAME_DATE','SEASON','SEASON_TYPE'] + DETAILS + NUMERIC + ['SOURCE','FETCHED_AT','RAW_JSON']


def initialize(con):
    con.execute('PRAGMA foreign_keys=ON')
    con.execute('''CREATE TABLE IF NOT EXISTS player_game_logs (
        GAME_ID TEXT NOT NULL, PLAYER_ID TEXT NOT NULL, TEAM_ID INTEGER NOT NULL,
        TEAM_NAME TEXT NOT NULL, PLAYER_NAME TEXT NOT NULL, GAME_DATE TEXT NOT NULL,
        SEASON TEXT NOT NULL, SEASON_TYPE TEXT NOT NULL,
        GAME_DATETIME TEXT NOT NULL DEFAULT '', OPPONENT_TEAM_ID INTEGER NOT NULL DEFAULT 0,
        HOME_AWAY TEXT NOT NULL DEFAULT '', STARTER INTEGER NOT NULL DEFAULT 0,
        DID_NOT_PLAY INTEGER NOT NULL DEFAULT 0, DNP_REASON TEXT NOT NULL DEFAULT '',
        POSITION TEXT NOT NULL DEFAULT '', MIN_IMPUTED INTEGER NOT NULL DEFAULT 0,
        IDENTITY_RECOVERED INTEGER NOT NULL DEFAULT 0, MIN_SCALE REAL NOT NULL DEFAULT 1,
        ''' + ','.join(f'"{c}" REAL NOT NULL' for c in NUMERIC) + ''',
        SOURCE TEXT NOT NULL, FETCHED_AT TEXT NOT NULL, RAW_JSON TEXT NOT NULL,
        PRIMARY KEY (GAME_ID, PLAYER_ID))''')
    con.execute('CREATE INDEX IF NOT EXISTS player_logs_date ON player_game_logs(GAME_DATE,TEAM_ID)')
    existing={row[1] for row in con.execute('PRAGMA table_info(player_game_logs)')}
    migrations={
        'GAME_DATETIME':"TEXT NOT NULL DEFAULT ''",'OPPONENT_TEAM_ID':'INTEGER NOT NULL DEFAULT 0',
        'HOME_AWAY':"TEXT NOT NULL DEFAULT ''",'STARTER':'INTEGER NOT NULL DEFAULT 0',
        'DID_NOT_PLAY':'INTEGER NOT NULL DEFAULT 0','DNP_REASON':"TEXT NOT NULL DEFAULT ''",
        'POSITION':"TEXT NOT NULL DEFAULT ''",'MIN_IMPUTED':'INTEGER NOT NULL DEFAULT 0',
        'IDENTITY_RECOVERED':'INTEGER NOT NULL DEFAULT 0','MIN_SCALE':'REAL NOT NULL DEFAULT 1',
    }
    for column,declaration in migrations.items():
        if column not in existing:
            con.execute(f'ALTER TABLE player_game_logs ADD COLUMN {column} {declaration}')
    con.execute('''CREATE TABLE IF NOT EXISTS player_report_snapshots (
        snapshot_id TEXT PRIMARY KEY, kind TEXT NOT NULL, game_date TEXT NOT NULL,
        published_at TEXT NOT NULL, fetched_at TEXT NOT NULL, source TEXT NOT NULL,
        complete_teams TEXT NOT NULL, records TEXT NOT NULL)''')
    con.execute('CREATE INDEX IF NOT EXISTS player_reports_time ON player_report_snapshots(game_date,published_at)')
    con.execute('''CREATE TABLE IF NOT EXISTS player_report_artifacts (
        snapshot_id TEXT PRIMARY KEY REFERENCES player_report_snapshots(snapshot_id),
        source_url TEXT NOT NULL, content_sha256 TEXT NOT NULL, content_type TEXT NOT NULL,
        parser_version TEXT NOT NULL, content BLOB NOT NULL)''')
    con.execute('''CREATE TABLE IF NOT EXISTS player_ingest_runs (
        id INTEGER PRIMARY KEY, fetched_at TEXT NOT NULL, source TEXT NOT NULL,
        scope TEXT NOT NULL, success INTEGER NOT NULL, records INTEGER NOT NULL, error TEXT)''')


def connect(path=DB_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    initialize(con)
    return con


def save_logs(con, rows):
    if not rows:
        return 0
    for season, source in {(r['SEASON'],r['SOURCE']) for r in rows}:
        other = con.execute('SELECT DISTINCT SOURCE FROM player_game_logs WHERE SEASON=? AND SOURCE<>?', (season,source)).fetchall()
        if other:
            raise ValueError(f'Season {season} already uses a different provider; do not duplicate player/game history')
    columns = ','.join(f'"{c}"' for c in LOG_COLUMNS)
    updates = ','.join(f'"{c}"=excluded."{c}"' for c in LOG_COLUMNS if c not in ('GAME_ID','PLAYER_ID'))
    with con:
        con.executemany(f'INSERT INTO player_game_logs ({columns}) VALUES ({",".join("?" for _ in LOG_COLUMNS)}) ON CONFLICT(GAME_ID,PLAYER_ID) DO UPDATE SET {updates}',
                        [[row[c] for c in LOG_COLUMNS] for row in rows])
    return len(rows)


def replace_logs(con, rows, season, source, season_type=None):
    """Atomically replace one complete provider scope instead of leaving stale rows."""
    if not rows:
        raise ValueError(f'Refusing to replace {season} with an empty player-log response')
    if any(r['SEASON'] != season or r['SOURCE'] != source for r in rows):
        raise ValueError('Replacement rows do not match the requested provider scope')
    other = con.execute(
        'SELECT DISTINCT SOURCE FROM player_game_logs WHERE SEASON=? AND SOURCE<>?',
        (season,source)).fetchall()
    if other:
        raise ValueError(f'Season {season} already uses a different provider; do not duplicate player/game history')
    columns = ','.join(f'"{c}"' for c in LOG_COLUMNS)
    placeholders = ','.join('?' for _ in LOG_COLUMNS)
    where='SEASON=? AND SOURCE=?'
    params=[season,source]
    if season_type is not None:
        where += ' AND SEASON_TYPE=?'
        params.append(season_type)
    with con:
        con.execute(f'DELETE FROM player_game_logs WHERE {where}',params)
        con.executemany(
            f'INSERT INTO player_game_logs ({columns}) VALUES ({placeholders})',
            [[row[c] for c in LOG_COLUMNS] for row in rows])
    return len(rows)


def save_snapshot(con, snapshot, artifact=None):
    with con:
        con.execute('INSERT OR IGNORE INTO player_report_snapshots VALUES (?,?,?,?,?,?,?,?)',
            (snapshot['snapshot_id'], snapshot['kind'], snapshot['game_date'], snapshot['published_at'],
             snapshot['fetched_at'], snapshot['source'], json.dumps(snapshot['complete_teams']), json.dumps(snapshot['records'])))
        if artifact is not None:
            content = artifact.get('content')
            if not isinstance(content, bytes):
                raise ValueError('Report artifact content must be bytes')
            digest = hashlib.sha256(content).hexdigest()
            con.execute('''INSERT OR IGNORE INTO player_report_artifacts
                (snapshot_id,source_url,content_sha256,content_type,parser_version,content)
                VALUES (?,?,?,?,?,?)''',
                (snapshot['snapshot_id'], artifact.get('source_url', ''), digest,
                 artifact.get('content_type', 'application/octet-stream'),
                 artifact.get('parser_version', 'unknown'), content))


def log_run(con, now, source, scope, success, count=0, error=None):
    with con:
        con.execute('INSERT INTO player_ingest_runs(fetched_at,source,scope,success,records,error) VALUES (?,?,?,?,?,?)',
                    (now, source, scope, int(success), count, error))
