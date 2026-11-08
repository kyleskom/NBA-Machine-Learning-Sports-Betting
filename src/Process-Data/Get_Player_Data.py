"""Player-data CLI: real game logs, independent availability refresh, diagnostics.

Run from the repository root: python -m src.Process-Data.Get_Player_Data --help
Legacy cumulative snapshots are retained in the DB but never used as game logs.
"""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

BASE_DIR=Path(__file__).resolve().parents[2]
sys.path.insert(0,os.fspath(BASE_DIR))
import toml
from src.Players.store import connect, DB_PATH
from src.Players.ingest import ingest_season, ingest_archive
from src.Players.reports import refresh_reports, import_report
from src.Players.http import FeedError


def configured_seasons(config):
    return list(config.get('create-games', config.get('get-player-data',{})))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season',help='Season key, e.g. 2025-26')
    parser.add_argument('--backfill',action='store_true',help='Fetch all configured training seasons')
    parser.add_argument('--provider',choices=['nba','archive'],help='NBA game logs or SportsDataverse ESPN archive')
    parser.add_argument('--csv',help='Import a downloaded archive CSV; requires --season')
    parser.add_argument('--reports-only',action='store_true',help='Refresh reports independently, including repeated runs today')
    parser.add_argument('--report-json',help='Import a timestamped report envelope')
    parser.add_argument('--kind',choices=['injuries','lineups','rosters'],default='injuries')
    parser.add_argument('--date',help='Report game date; defaults to today in America/New_York')
    parser.add_argument('--status',action='store_true',help='Show stored data coverage without network calls')
    parser.add_argument('--db',type=Path,default=DB_PATH)
    parser.add_argument('--rebuild-derived',action='store_true',help='Compatibility flag: features now build on demand at a cutoff')
    args=parser.parse_args(argv)
    config=toml.load(BASE_DIR/'config.toml')
    day=args.date or datetime.now(ZoneInfo('America/New_York')).date().isoformat()
    if args.csv and not args.season:
        parser.error('--csv requires --season')
    failed=False
    with connect(args.db) as con:
        if args.status:
            rows=con.execute('SELECT SEASON,SOURCE,COUNT(*),COUNT(DISTINCT GAME_ID),COUNT(DISTINCT PLAYER_ID),MIN(GAME_DATE),MAX(GAME_DATE) FROM player_game_logs GROUP BY SEASON,SOURCE').fetchall()
            print('season | source | rows | games | players | first | last')
            for row in rows:
                print(' | '.join(map(str,row)))
            print('Total player/game rows:',con.execute('SELECT COUNT(*) FROM player_game_logs').fetchone()[0])
            print('Minute-imputed rows:',con.execute('SELECT COUNT(*) FROM player_game_logs WHERE MIN_IMPUTED=1').fetchone()[0])
            print('Recovered identities:',con.execute('SELECT COUNT(*) FROM player_game_logs WHERE IDENTITY_RECOVERED=1').fetchone()[0])
            print('Missing opponent IDs:',con.execute('SELECT COUNT(*) FROM player_game_logs WHERE OPPONENT_TEAM_ID=0').fetchone()[0])
            reports=con.execute('SELECT kind,COUNT(*) FROM player_report_snapshots GROUP BY kind').fetchall()
            print('Report snapshots:',dict(reports))
            artifacts=con.execute('SELECT COUNT(*),COALESCE(SUM(length(content)),0) FROM player_report_artifacts').fetchone()
            print('Archived report artifacts:', {'count': artifacts[0], 'bytes': artifacts[1]})
            return 0
        if args.report_json:
            snapshot=import_report(con,json.loads(Path(args.report_json).read_text()),args.kind,'file')
            print('Imported snapshot',snapshot['snapshot_id'])
            return 0
        if not args.reports_only:
            seasons=[args.season] if args.season else configured_seasons(config)
            if not args.backfill and not args.season:
                seasons=seasons[-1:]
            for season in seasons:
                try:
                    provider=args.provider or config.get('player-ingestion',{}).get('provider','archive')
                    if provider=='archive':
                        count=ingest_archive(con,season,args.csv,config.get('player-ingestion',{}))
                    else:
                        count=sum(ingest_season(con,season,kind,config.get('player-ingestion',{})) for kind in ['Regular Season','Playoffs'])
                    print(f'{season}: upserted {count} player/game records')
                except (FeedError,ValueError,KeyError) as exc:
                    print(f'{season}: {exc}',file=sys.stderr)
                    failed=True
        # Reports are always attempted even if statistics failed or already existed.
        results=refresh_reports(con,config,day)
        for kind,status in results.items():
            print(f'{kind}: {status}')
            failed |= status.startswith('failed')
    return 1 if failed else 0


if __name__=='__main__':
    raise SystemExit(main())
