"""Provider-neutral timestamped availability snapshots, never date-stamped guesses.

Feeds must supply a JSON envelope: game_date, published_at (timezone required),
records, and optionally complete_teams. IDs are matched exactly first; a supplied
name can bridge provider namespaces only when it uniquely resolves within a team.
"""
from datetime import datetime, timezone, timedelta
import hashlib
import io
import json
import re
from zoneinfo import ZoneInfo
import pandas as pd
from .identity import team_id, TEAM_NAMES
from .http import get_bytes, get_json, get_json_with_metadata, FeedError
from .ingest import utc_now
from .store import save_snapshot, log_run

INJURY_STATUS_MAP = {'out':'OUT','o':'OUT','inactive':'OUT','doubtful':'DOUBTFUL',
    'questionable':'QUESTIONABLE','game time decision':'QUESTIONABLE','gtd':'QUESTIONABLE',
    'day-to-day':'QUESTIONABLE','probable':'PROBABLE','healthy':'ACTIVE','active':'ACTIVE','available':'ACTIVE'}
LINEUP_STATUS_MAP = {'confirmed':'CONFIRMED','confirmed starter':'CONFIRMED','starting':'CONFIRMED',
    'projected':'PROJECTED','projected starter':'PROJECTED','starter':'PROJECTED','expected':'PROJECTED',
    'bench':'BENCH','out':'OUT','inactive':'OUT'}


def normalize_injury_status(status):
    return INJURY_STATUS_MAP.get(str(status or '').strip().casefold(), 'UNKNOWN')


def normalize_lineup_status(status):
    return LINEUP_STATUS_MAP.get(str(status or '').strip().casefold(), 'UNKNOWN')


EASTERN = ZoneInfo('America/New_York')
TEAM_LABELS = sorted(TEAM_NAMES.items(), key=lambda item: len(item[1]), reverse=True)
STATUS_PATTERN = re.compile(r'(?P<last>[^,]+),\s*(?P<first>.+?)\s+(?P<status>Out|Doubtful|Questionable|Probable|Available)\b', re.I)
GAME_DATE_PATTERN = re.compile(r'\b(?P<date>\d{2}/\d{2}/\d{4})\b')


def _official_report_time(day, label):
    parsed = datetime.strptime(label.replace('_', ':'), '%I:%M%p').time()
    return pd.Timestamp(datetime.combine(day, parsed, EASTERN)).tz_convert('UTC')


def _parse_official_injury_text(text, game_date):
    """Parse NBA's official PDF text while retaining only the requested slate."""
    target = pd.Timestamp(game_date).date()
    active_day = None
    active_team = None
    complete = set()
    records = []
    for source_line in text.splitlines():
        line = ' '.join(source_line.split())
        if not line or line.startswith(('Injury Report:', 'Page ')):
            continue
        date_match = GAME_DATE_PATTERN.search(line)
        if date_match:
            active_day = datetime.strptime(date_match.group('date'), '%m/%d/%Y').date()
        team_match = next(((tid, label, line.find(label)) for tid, label in TEAM_LABELS
                           if line.find(label) >= 0), None)
        entry = line
        if team_match:
            tid, label, position = team_match
            active_team = tid
            entry = line[position + len(label):].strip()
            if active_day == target and 'NOT YET SUBMITTED' in entry.upper():
                complete.discard(tid)
                continue
            if active_day == target:
                complete.add(tid)
        if active_day != target or active_team is None:
            continue
        player = STATUS_PATTERN.search(entry)
        if not player:
            continue
        records.append({
            'team_id': active_team,
            'player_name': f"{player.group('first').strip()} {player.group('last').strip()}",
            'status': player.group('status'),
            'reason': entry[player.end():].strip(),
        })
    return records, sorted(complete)


def official_injury_snapshot(game_date, settings, fetched_at=None):
    """Read the latest accessible official NBA PDF that contains this game date."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise FeedError('pypdf is required for official NBA injury reports') from exc
    now = pd.Timestamp(fetched_at or utc_now()).tz_convert('UTC')
    day = pd.Timestamp(game_date).date()
    template = settings.get('pdf_url_template', '')
    labels = settings.get('report_times', [])
    if not template or not labels:
        raise ValueError('Official injury report requires pdf_url_template and report_times')
    candidates = []
    for report_day in (day - timedelta(days=1), day):
        for label in labels:
            published = _official_report_time(report_day, label)
            if published > now:
                continue
            url = template.format(date=report_day.isoformat(), date_compact=report_day.strftime('%Y%m%d'), time=label)
            try:
                content, headers = get_bytes(url, headers={'User-Agent': 'Mozilla/5.0'}, attempts=1,
                                             timeout=float(settings.get('timeout_seconds', 20)))
                # The NBA PDF draws each page from the bottom upward, so pypdf
                # returns player rows before the team/game heading they belong
                # to. Reversing the extracted lines restores the report order.
                pages = []
                for page in PdfReader(io.BytesIO(content)).pages:
                    lines = (page.extract_text(extraction_mode='layout') or '').splitlines()
                    pages.append('\n'.join(reversed(lines)))
                text = '\n'.join(pages)
                records, complete = _parse_official_injury_text(text, game_date)
                if records or complete:
                    candidates.append((published, records, complete, url, content, headers))
            except FeedError:
                continue
    if not candidates:
        raise FeedError('No official NBA injury report is available for this slate yet')
    published, records, complete, url, content, headers = max(candidates, key=lambda item: item[0])
    return {'game_date': game_date, 'published_at': published.isoformat(),
            'records': records, 'complete_teams': complete,
            '_artifact': {'source_url': url, 'content': content,
                          'content_type': headers.get('Content-Type', 'application/pdf'),
                          'parser_version': 'nba_official_pdf_v1'}}


def nba_daily_lineups_snapshot(game_date, kind, settings, fetched_at=None):
    """Map NBA's daily lineup JSON into confirmed/projected starters and rosters."""
    template = settings.get('daily_lineups_url_template', '')
    if not template:
        raise ValueError('NBA daily lineup adapter requires daily_lineups_url_template')
    fetched = timestamp(fetched_at or utc_now())
    url = template.format(date=game_date, date_compact=game_date.replace('-', ''))
    payload, content, headers = get_json_with_metadata(
        url, headers={'User-Agent': 'Mozilla/5.0'}, attempts=int(settings.get('attempts', 2)),
        timeout=float(settings.get('timeout_seconds', 20)))
    records, complete = [], set()
    for game in payload.get('games', []):
        for side in ('homeTeam', 'awayTeam'):
            team = game.get(side) or {}
            players = team.get('players') or []
            if not players:
                continue
            tid = team_id(team.get('teamId') or team.get('teamAbbreviation'))
            if kind == 'rosters':
                # This is a lineup feed, not an authoritative full-roster feed.
                # Retain its observations without claiming roster completeness.
                selected = players
            else:
                # NBA's daily feed orders the five listed starters first and
                # labels the lineup as Confirmed or Projected at publish time.
                selected = players[:5]
            for player in selected:
                status = 'ROSTER' if kind == 'rosters' else player.get('lineupStatus', 'PROJECTED')
                records.append({'team_id': tid, 'player_id': f"nba:{player['personId']}",
                                'player_name': player.get('playerName', ''), 'status': status})
    if not records:
        raise FeedError('NBA daily lineup feed has no player records for this slate yet')
    return {'game_date': game_date, 'published_at': fetched,
            'records': records, 'complete_teams': sorted(complete),
            '_artifact': {'source_url': url, 'content': content,
                          'content_type': headers.get('Content-Type', 'application/json'),
                          'parser_version': 'nba_daily_lineups_v2'}}


def timestamp(value):
    dt = pd.Timestamp(value)
    if pd.isna(dt) or dt.tzinfo is None:
        raise ValueError('Report timestamps must include a timezone')
    return dt.tz_convert('UTC').isoformat()


def parse_snapshot(payload, kind, source, fetched_at=None):
    if kind not in ('injuries','lineups','rosters'):
        raise ValueError('Unknown player report kind')
    if not isinstance(payload, dict) or not isinstance(payload.get('records'), list):
        raise ValueError('Report must contain a records list, including for a verified empty report')
    now = timestamp(fetched_at or utc_now())
    published = timestamp(payload.get('published_at'))
    if published > now:
        raise ValueError('Report publication is in the future')
    day = str(payload.get('game_date',''))
    if len(day) != 10 or pd.Timestamp(day).date().isoformat() != day:
        raise ValueError('Report requires an ISO game_date')
    complete = sorted({team_id(t) for t in payload.get('complete_teams',[])})
    rows, seen = [], set()
    for raw in payload['records']:
        tid = team_id(raw.get('team_id') or raw.get('team'))
        pid = str(raw.get('player_id') or '').strip()
        name = str(raw.get('player_name') or '').strip()
        if not pid and not name:
            raise ValueError('Report player requires an ID or name')
        if pid and ':' not in pid:
            raise ValueError('Report player_id requires source namespace, e.g. nba:2544')
        key = tid, pid or name.casefold()
        if key in seen:
            raise ValueError('Conflicting or duplicate player report; resolve upstream')
        seen.add(key)
        status = ('ROSTER' if kind == 'rosters' else
                  normalize_injury_status(raw.get('status')) if kind == 'injuries' else
                  normalize_lineup_status(raw.get('status')))
        limit = raw.get('minutes_limit')
        if limit is not None:
            limit = float(limit)
            if not 0 <= limit <= 48:
                raise ValueError('Invalid regulation minutes limit')
        rows.append(dict(TEAM_ID=tid, PLAYER_ID=pid, PLAYER_NAME=name, STATUS=status, MINUTES_LIMIT=limit))
    content = dict(kind=kind, game_date=day, published_at=published, fetched_at=now,
                   source=source, complete_teams=complete, records=rows)
    # Re-fetching unchanged publication is idempotent; fetching time does not change identity.
    hashed = {k:v for k,v in content.items() if k != 'fetched_at'}
    content['snapshot_id'] = hashlib.sha256(json.dumps(hashed, sort_keys=True).encode()).hexdigest()
    return content


def import_report(con, payload, kind, source='file', fetched_at=None):
    snapshot = parse_snapshot(payload, kind, source, fetched_at)
    save_snapshot(con, snapshot, payload.get('_artifact'))
    return snapshot


def refresh_reports(con, config, game_date):
    """Independent of game-log ingestion. Failure never deletes earlier snapshots."""
    result = {}
    for kind in ('injuries','lineups','rosters'):
        settings = config.get(f'player-{kind}',{})
        sources = [(k,settings.get(k,'')) for k in ('primary_url','secondary_url')]
        result[kind] = 'not configured'
        adapter = settings.get('adapter', '')
        if adapter:
            now = utc_now()
            try:
                if adapter == 'nba_official_pdf':
                    payload = official_injury_snapshot(game_date, settings, now)
                    source = 'nba_official_pdf'
                elif adapter == 'nba_daily_lineups':
                    payload = nba_daily_lineups_snapshot(game_date, kind, settings, now)
                    source = 'nba_daily_lineups'
                else:
                    raise ValueError(f'Unknown player-report adapter: {adapter}')
                snap = import_report(con, payload, kind, source, now)
                log_run(con, now, source, kind, True, len(snap['records']))
                result[kind] = 'updated'
                continue
            except FeedError as exc:
                log_run(con, now, adapter, kind, False, error=str(exc))
                result[kind] = 'not available; previous snapshot retained'
            except (ValueError, KeyError) as exc:
                log_run(con, now, adapter, kind, False, error=str(exc))
                result[kind] = 'failed; previous snapshot retained'
        for label, url in sources:
            if not url:
                continue
            now = utc_now()
            try:
                payload = get_json(url.format(game_date, date=game_date))
                if not isinstance(payload,dict) or payload.get('game_date') != game_date:
                    raise ValueError('Feed returned a different game date')
                snap = import_report(con,payload,kind,label,now)
                log_run(con,now,label,kind,True,len(snap['records']))
                result[kind] = 'updated'
                break
            except (FeedError,ValueError,KeyError) as exc:
                log_run(con,now,label,kind,False,error=str(exc))
                result[kind] = 'failed; previous snapshot retained'
        path = settings.get('json_path','')
        if path:
            from pathlib import Path
            path = Path(path).expanduser()
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[2] / path
            try:
                payload = json.loads(path.read_text())
                if payload.get('game_date') != game_date:
                    raise ValueError('File report is for a different game date')
                import_report(con,payload,kind,'file')
                result[kind] = 'updated'
            except (OSError,ValueError,KeyError) as exc:
                log_run(con,utc_now(),'file',kind,False,error=str(exc))
                result[kind] = 'failed; previous snapshot retained'
    return result
