"""Fetch player game logs with normalized provenance and repair metadata."""
from datetime import datetime, timezone
import json
import math
import pandas as pd
from .http import get_json, FeedError
from .identity import team_id, name_key, TEAM_NAMES
from .store import NUMERIC, replace_logs, log_run

NBA_URL = 'https://stats.nba.com/stats/leaguegamelog'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def number(value):
    if isinstance(value, str) and ':' in value:
        minute, second = value.split(':', 1)
        value = float(minute) + float(second) / 60
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Non-finite box-score statistic')
    return value


def truthy(value):
    return str(value).strip().casefold() in {'true','1','yes'}


def clean_text(value):
    return '' if value is None or pd.isna(value) else str(value).strip()


def optional_team_id(raw):
    value=raw.get('OPPONENT_TEAM_ID')
    if value is None or pd.isna(value):
        value=raw.get('opponent_team_abbreviation')
    if value is None or pd.isna(value):
        matchup=str(raw.get('MATCHUP',''))
        value=matchup.rsplit(' ',1)[-1] if matchup else None
    try:
        return team_id(value) if value is not None else 0
    except (KeyError,TypeError,ValueError):
        return 0


def normalize_logs(frame, season, season_type, fetched_at=None, source='nba'):
    required = {'GAME_ID','PLAYER_ID','PLAYER_NAME','GAME_DATE','TEAM_ID'} | set(NUMERIC)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Game-log schema missing columns: {sorted(missing)}')
    now = fetched_at or utc_now()
    rows = []
    for raw in frame.to_dict('records'):
        tid = team_id(raw['TEAM_ID'])
        if pd.isna(raw['PLAYER_ID']) or pd.isna(raw['GAME_ID']) or not str(raw['PLAYER_NAME']).strip():
            raise ValueError('Missing player/game identity')
        pid = str(raw['PLAYER_ID']).removesuffix('.0')
        gid = str(raw['GAME_ID']).removesuffix('.0')
        # Namespaces prevent accidentally joining ESPN IDs to NBA IDs.
        pid = pid if ':' in pid else f'{source}:{pid}'
        gid = gid if ':' in gid else f'{source}:{gid.zfill(10) if source == "nba" else gid}'
        game_date = pd.Timestamp(raw['GAME_DATE']).date().isoformat()
        if game_date > pd.Timestamp(now).date().isoformat():
            raise ValueError('Future game in completed game logs')
        row = dict(GAME_ID=gid, PLAYER_ID=pid, TEAM_ID=tid, TEAM_NAME=TEAM_NAMES[tid],
                   PLAYER_NAME=str(raw['PLAYER_NAME']).strip(), GAME_DATE=game_date,
                   SEASON=season, SEASON_TYPE=season_type, SOURCE=source, FETCHED_AT=now,
                   GAME_DATETIME=clean_text(raw.get('GAME_DATETIME')) or clean_text(raw.get('game_date_time')),
                   OPPONENT_TEAM_ID=optional_team_id(raw),
                   HOME_AWAY=(clean_text(raw.get('HOME_AWAY')) or clean_text(raw.get('home_away')) or
                              ('HOME' if ' vs. ' in str(raw.get('MATCHUP','')) else
                               'AWAY' if ' @ ' in str(raw.get('MATCHUP','')) else '')).upper(),
                   STARTER=int(truthy(raw.get('STARTER',raw.get('starter',False)))),
                   DID_NOT_PLAY=int(truthy(raw.get('DID_NOT_PLAY',raw.get('did_not_play',False)))),
                   DNP_REASON=clean_text(raw.get('reason')),
                   POSITION=clean_text(raw.get('athlete_position_abbreviation')),
                   MIN_IMPUTED=int(truthy(raw.get('ARCHIVE_MIN_IMPUTED',False))),
                   IDENTITY_RECOVERED=int(truthy(raw.get('ARCHIVE_IDENTITY_RECOVERED',False))),
                   MIN_SCALE=float(raw.get('ARCHIVE_MIN_SCALE',1)), RAW_JSON='{}')
        # The normalized columns contain the source fields used by the model.
        # Retain only values needed to audit a minute repair instead of copying
        # redundant logos, colors, URLs, and team labels into every DB row.
        if row['MIN_IMPUTED'] or abs(row['MIN_SCALE']-1)>1e-12:
            original=raw.get('ARCHIVE_ORIGINAL_MIN')
            row['RAW_JSON']=json.dumps({'source_minutes':None if pd.isna(original) else original,
                                        'minutes_imputed':bool(row['MIN_IMPUTED']),
                                        'minutes_scale':row['MIN_SCALE']},
                                       default=str,separators=(',',':'),allow_nan=False)
        for field in NUMERIC:
            row[field] = number(raw[field])
            if field != 'PLUS_MINUS' and row[field] < 0:
                raise ValueError(f'Negative {field}')
        for made, attempted in [('FGM','FGA'),('FG3M','FG3A'),('FTM','FTA')]:
            if row[made] > row[attempted]:
                raise ValueError(f'{made} exceeds {attempted}')
        if row['MIN'] > 80:
            raise ValueError('Implausible minutes')
        rows.append(row)
    keys = [(r['GAME_ID'], r['PLAYER_ID']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate player/game rows in source response')
    return rows


def fetch_season(season, season_type='Regular Season', settings=None):
    settings = settings or {}
    payload = get_json(settings.get('game_logs_url', NBA_URL), params={
        'Counter':1000, 'DateFrom':'', 'DateTo':'', 'Direction':'ASC', 'LeagueID':'00',
        'PlayerOrTeam':'P', 'Season':season, 'SeasonType':season_type, 'Sorter':'DATE'},
        headers={'User-Agent':'Mozilla/5.0','Referer':'https://www.nba.com/'},
        timeout=settings.get('timeout_seconds',20), attempts=settings.get('attempts',3))
    sets = payload.get('resultSets', []) if isinstance(payload, dict) else []
    result = next((s for s in sets if 'PLAYER_ID' in s.get('headers',[]) and 'GAME_ID' in s.get('headers',[])), None)
    if result is None:
        raise FeedError('NBA response lacks player game-log result set')
    return pd.DataFrame(result['rowSet'], columns=result['headers'])


def ingest_season(con, season, season_type='Regular Season', settings=None):
    now = utc_now()
    scope = f'{season}/{season_type}'
    try:
        rows = normalize_logs(fetch_season(season, season_type, settings), season, season_type, now)
        count = replace_logs(con, rows, season, source='nba', season_type=season_type)
        log_run(con, now, 'nba', scope, True, count)
        return count
    except (FeedError, ValueError, KeyError) as exc:
        log_run(con, now, 'nba', scope, False, error=str(exc))
        raise


def normalize_archive(frame, season, fetched_at=None):
    """SportsDataverse/hoopR ESPN box scores; retain DNP rows and raw fields."""
    mapping = {'game_id':'GAME_ID','athlete_id':'PLAYER_ID','athlete_display_name':'PLAYER_NAME',
        'game_date':'GAME_DATE','minutes':'MIN','points':'PTS','rebounds':'REB','assists':'AST',
        'field_goals_made':'FGM','field_goals_attempted':'FGA','three_point_field_goals_made':'FG3M',
        'three_point_field_goals_attempted':'FG3A','free_throws_made':'FTM','free_throws_attempted':'FTA',
        'offensive_rebounds':'OREB','defensive_rebounds':'DREB','turnovers':'TOV','steals':'STL',
        'blocks':'BLK','fouls':'PF','plus_minus':'PLUS_MINUS'}
    required = set(mapping) | {'team_abbreviation','season_type','did_not_play'}
    if required-set(frame.columns):
        raise ValueError(f'Archive schema missing {sorted(required-set(frame.columns))}')
    frame=frame.copy()
    frame=frame[frame.season_type.isin([2,3,'2','3'])]  # exclude preseason/all-star
    from .identity import ALIASES
    franchise = frame.team_abbreviation.astype(str).str.casefold().isin(ALIASES)
    if not franchise.all():
        print(f'Excluded {(~franchise).sum()} non-franchise exhibition player rows: {sorted(frame.loc[~franchise, "team_abbreviation"].astype(str).unique())}')
    frame=frame[franchise].copy()
    frame['TEAM_ID']=frame.team_abbreviation.map(team_id)
    frame=frame.rename(columns=mapping)
    frame['PLAYER_ID']=frame['PLAYER_ID'].astype(object)
    frame['PLAYER_NAME']=frame['PLAYER_NAME'].astype(object)

    # Recover the one known archive failure mode where the full name and ESPN
    # ID are blank but a stable short name is present across the player's team.
    frame['ARCHIVE_IDENTITY_RECOVERED']=False
    missing_name=(frame['PLAYER_NAME'].isna()
                  | frame['PLAYER_NAME'].astype(str).str.strip().eq(''))
    short=frame.get('athlete_short_name',pd.Series(index=frame.index,dtype=object))
    recoverable=missing_name & short.notna() & short.astype(str).str.strip().ne('')
    frame.loc[recoverable,'PLAYER_NAME']=short.loc[recoverable].astype(str).str.strip()
    missing_id=frame['PLAYER_ID'].isna() | frame['PLAYER_ID'].astype(str).str.strip().eq('')
    recoverable_id=missing_id & frame['PLAYER_NAME'].notna() & recoverable
    frame.loc[recoverable_id,'PLAYER_ID']=[
        f'espn-synthetic:{tid}:{name_key(player)}'
        for tid,player in zip(frame.loc[recoverable_id,'TEAM_ID'],frame.loc[recoverable_id,'PLAYER_NAME'])]
    frame.loc[recoverable | recoverable_id,'ARCHIVE_IDENTITY_RECOVERED']=True
    if (recoverable | recoverable_id).any():
        print(f'Recovered {(recoverable | recoverable_id).sum()} archive player identities from short names')

    # Any rows still lacking an identity cannot be joined safely and are not
    # player observations.
    identifiable=(frame['PLAYER_ID'].notna() & frame['PLAYER_NAME'].notna()
                  & frame['PLAYER_NAME'].astype(str).str.strip().ne(''))
    if not identifiable.all():
        print(f'Excluded {(~identifiable).sum()} player rows without an ID/name')
    frame=frame[identifiable].copy()

    # Some ESPN rows say did_not_play=false even though minutes are blank and
    # every counting stat is zero (usually COACH'S DECISION). Classify those as
    # nonparticipants, but never fill a partially populated played stat line.
    explicit_dnp=frame.did_not_play.astype(str).str.lower().isin(['true','1'])
    # PLUS_MINUS is excluded from the participation check because ESPN has a
    # small number of DNP rows carrying a team-level plus/minus value.
    other_stats=[field for field in NUMERIC if field not in ('MIN','PLUS_MINUS')]
    numeric=frame[other_stats].apply(pd.to_numeric,errors='coerce')
    blank_minutes=pd.to_numeric(frame['MIN'],errors='coerce').isna()
    zero_stat_blank_minutes=blank_minutes & numeric.fillna(0).eq(0).all(axis=1)

    # A few source rows represent a player who appeared but omit only his
    # minutes. When exactly one such row exists for a team/game, allocate the
    # shortfall to the nearest legal team total (240 plus 25 per overtime).
    # This is recorded in RAW_JSON so downstream audits can identify it.
    frame['ARCHIVE_MIN_IMPUTED']=False
    played_candidate=zero_stat_blank_minutes & ~explicit_dnp
    for _,group in frame.groupby(['GAME_ID','TEAM_ID']):
        candidates=group.index[played_candidate.loc[group.index]]
        if len(candidates) != 1:
            continue
        observed=pd.to_numeric(group['MIN'],errors='coerce').sum()
        overtime=max(0,round((observed-240)/25))
        target=240+25*overtime
        gap=target-observed
        if 0 < gap <= 48:
            frame.loc[candidates[0],'MIN']=float(gap)
            frame.loc[candidates[0],'ARCHIVE_MIN_IMPUTED']=True
    imputed=frame['ARCHIVE_MIN_IMPUTED']
    dnp=explicit_dnp | (zero_stat_blank_minutes & ~imputed)
    for field in NUMERIC:
        repair=dnp | imputed
        frame.loc[repair,field]=pd.to_numeric(frame.loc[repair,field],errors='coerce').fillna(0)

    # ESPN publishes integer minutes, so a team often sums to 238-242 rather
    # than the legal 240 (plus 25 per overtime). Proportionally reconcile each
    # team/game to the nearest legal total and retain the scale in RAW_JSON.
    frame['ARCHIVE_ORIGINAL_MIN']=frame['MIN']
    frame['ARCHIVE_MIN_SCALE']=1.0
    for _,group in frame.groupby(['GAME_ID','TEAM_ID']):
        minutes=pd.to_numeric(group['MIN'],errors='coerce')
        observed=minutes.fillna(0).sum()
        if observed <= 0:
            continue
        overtime=max(0,round((observed-240)/25))
        target=240+25*overtime
        if abs(target-observed) > 15:
            raise ValueError(f'Implausible team-minute total {observed} for archive game')
        scale=target/observed
        frame.loc[group.index,'MIN']=minutes.fillna(0)*scale
        frame.loc[group.index,'ARCHIVE_MIN_SCALE']=scale
    rows=[]
    for stype,group in frame.groupby('season_type'):
        rows.extend(normalize_logs(group,season,'Playoffs' if str(stype)=='3' else 'Regular Season',fetched_at,'espn'))
    return rows


def ingest_archive(con, season, csv_path=None, settings=None):
    import io
    import requests
    settings=settings or {}
    year=int(season[:4])+1
    now=utc_now()
    try:
        if csv_path:
            frame=pd.read_csv(csv_path,low_memory=False)
        else:
            template=settings.get('archive_url_template','https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_player_boxscores/player_box_{year}.csv')
            url=template.format(year=year,season=season)
            last=None
            attempts=int(settings.get('attempts',3))
            for attempt in range(attempts):
                try:
                    response=requests.get(url,timeout=float(settings.get('timeout_seconds',45)))
                    response.raise_for_status()
                    frame=pd.read_csv(io.BytesIO(response.content),low_memory=False)
                    break
                except (requests.RequestException,ValueError) as exc:
                    last=exc
                    if attempt+1==attempts:
                        raise FeedError(f'Archive download failed ({type(last).__name__})') from exc
                    import time
                    time.sleep(2**attempt)
        rows=normalize_archive(frame,season,now)
        count=replace_logs(con,rows,season,source='espn')
        log_run(con,now,'espn','archive/'+season,True,count)
        return count
    except (FeedError,ValueError,KeyError) as exc:
        log_run(con,now,'espn','archive/'+season,False,error=str(exc))
        raise
