"""Identical pregame player features for historical training and live inference.

Game logs are reconstructed history: only earlier local game dates are used.
Availability snapshots additionally require publication at/before the cutoff.
"""
import json
import math
import sqlite3
import numpy as np
import pandas as pd
from .identity import team_id, name_key
from .store import DB_PATH

BASE = ['PLR_MIN_SUM','PLR_PTS_SUM','PLR_REB_SUM','PLR_AST_SUM','PLR_USG_WEIGHTED_SUM']
ROLL_METRICS = ('MIN','PTS','REB','AST','USG_WEIGHTED')
EXTRA = ['PLR_GAME_COVERAGE','PLR_HISTORY_GAMES','PLR_STATS_AGE_DAYS','PLR_ROSTER_CONFIRMED',
    'PLR_PROJECTED_MIN_SUM','PLR_PROJECTED_PTS','PLR_PROJECTED_REB','PLR_PROJECTED_AST',
    'PLR_ROTATION_SIZE','PLR_USAGE_PROXY_COVERAGE','INJ_OUT_MIN_WT','INJ_Q_MIN_WT',
    'INJ_DOUBTFUL_MIN_WT','INJ_AVAIL_MIN_WT','INJ_COVERAGE','INJ_REPORT_AGE_HOURS',
    'LINEUP_EXPECTED_STARTERS','LINEUP_CONFIRMED_STARTERS','LINEUP_PROJECTED_STARTERS',
    'LINEUP_COVERAGE','LINEUP_EST_MIN','LINEUP_EST_PTS','LINEUP_EST_REB','LINEUP_EST_AST',
    'LINEUP_EST_USG_WEIGHTED']


def feature_columns(config):
    windows = sorted({int(n) for n in config.get('player-features',{}).get('rolling_windows',[7,14,30])})
    if not windows or min(windows) < 1:
        raise ValueError('Player rolling windows must be positive')
    return BASE + [f'PLR_{metric}_ROLL{w}' for w in windows for metric in ROLL_METRICS] + EXTRA


def defaults(config):
    result = {c:0.0 for c in feature_columns(config)}
    result.update(PLR_STATS_AGE_DAYS=999.0, INJ_REPORT_AGE_HOURS=999.0,
                  LINEUP_EXPECTED_STARTERS=5.0)
    # Availability=0 with coverage=0 is unknown, not an assertion of a healthy team.
    return result


def cutoff_for_date(game_date):
    return pd.Timestamp(game_date, tz='America/New_York').tz_convert('UTC')


def allocate_minutes(weights, caps, total=240.0):
    weights = np.asarray(weights,dtype=float)
    caps = np.asarray(caps,dtype=float)
    result = np.zeros(len(weights))
    for _ in range(len(weights)+1):
        remaining = total - result.sum()
        eligible = (caps-result > 1e-8) & (weights > 0)
        if remaining <= 1e-8 or not eligible.any():
            break
        additions = remaining * weights[eligible] / weights[eligible].sum()
        result[eligible] += np.minimum(additions,caps[eligible]-result[eligible])
    return result


class PlayerFeatureEngine:
    def __init__(self, logs, snapshots=None, config=None):
        self.config = config or {}
        self.settings = self.config.get('player-features',{})
        self.logs = logs.copy()
        self.snapshots = snapshots or []
        self.cache = {}
        self.teams = {}
        self.players = {}
        if not logs.empty:
            self.logs['GAME_DATE'] = self.logs.GAME_DATE.astype(str)
            self.logs = self.logs.sort_values(['GAME_DATE','GAME_ID','PLAYER_ID'])
            self.teams = {int(k):g for k,g in self.logs.groupby('TEAM_ID')}
            self.players = {str(k):g for k,g in self.logs.groupby('PLAYER_ID')}

    @classmethod
    def from_database(cls, path=DB_PATH, config=None):
        from pathlib import Path
        if not Path(path).exists():
            return cls(pd.DataFrame(),config=config)
        with sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro',uri=True) as con:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            logs = pd.read_sql_query('SELECT * FROM player_game_logs',con) if 'player_game_logs' in tables else pd.DataFrame()
            snapshots = []
            if 'player_report_snapshots' in tables:
                for row in con.execute('SELECT kind,game_date,published_at,fetched_at,source,complete_teams,records FROM player_report_snapshots'):
                    snapshots.append(dict(zip(['kind','game_date','published_at','fetched_at','source','complete_teams','records'],row)))
        for snap in snapshots:
            snap['records'] = json.loads(snap['records'])
            snap['complete_teams'] = json.loads(snap['complete_teams'])
        return cls(logs,snapshots,config)

    def report(self, kind, tid, day, cutoff):
        max_age = float(self.settings.get('report_max_age_hours',24))
        candidates = []
        for snap in self.snapshots:
            if snap['kind'] != kind or snap['game_date'] != day:
                continue
            published = pd.Timestamp(snap['published_at'])
            age = (cutoff-published).total_seconds()/3600
            if age < 0 or age > max_age:
                continue
            # A team-specific snapshot must not hide another team's earlier report.
            rows = [r for r in snap['records'] if r['TEAM_ID']==tid]
            if rows or tid in snap['complete_teams']:
                candidates.append((published,snap['fetched_at'],snap,rows,age))
        if not candidates:
            return [],False,999.0
        _,_,snap,rows,age = max(candidates,key=lambda x:(x[0],x[1]))
        return rows,tid in snap['complete_teams'],age

    def team_features(self, team, game_date, as_of=None):
        tid = team_id(team)
        day = pd.Timestamp(game_date).date().isoformat()
        cutoff = cutoff_for_date(day) if as_of is None else pd.Timestamp(as_of)
        if cutoff.tzinfo is None:
            raise ValueError('Player feature cutoff must be timezone-aware')
        cutoff = cutoff.tz_convert('UTC')
        if cutoff.tz_convert('America/New_York').date().isoformat() > day:
            raise ValueError('Postgame-date cutoff cannot be used for pregame prediction')
        # If predicting days ahead, do not include games between now and game day.
        history_day = min(day,cutoff.tz_convert('America/New_York').date().isoformat())
        cache_key = tid,day,cutoff.isoformat()
        if cache_key in self.cache:
            return dict(self.cache[cache_key])
        result = defaults(self.config)
        history = self.teams.get(tid,pd.DataFrame())
        if history.empty:
            return result
        history = history[history.GAME_DATE < history_day]
        if history.empty:
            return result
        age = (pd.Timestamp(day)-pd.Timestamp(history.GAME_DATE.max())).days
        result['PLR_STATS_AGE_DAYS'] = float(age)
        if age > int(self.settings.get('stats_max_age_days',45)):
            return result
        # Do not silently mix prior seasons into current rolling form.
        latest_season = history.iloc[-1].SEASON
        history = history[history.SEASON==latest_season].copy()
        totals = history.groupby(['GAME_DATE','GAME_ID'],sort=True)[['MIN','PTS','REB','AST','FGA','FTA','TOV']].sum()
        # This is an explicit possession-usage proxy, not the NBA advanced USG_PCT.
        totals['USG_WEIGHTED'] = totals.FGA + .44*totals.FTA + totals.TOV
        result['PLR_GAME_COVERAGE'] = 1.0
        result['PLR_HISTORY_GAMES'] = float(len(totals))
        for metric in ROLL_METRICS:
            result[f'PLR_{metric}_SUM'] = float(totals.tail(14)[metric].mean())
            for col in feature_columns(self.config):
                prefix = f'PLR_{metric}_ROLL'
                if col.startswith(prefix):
                    result[col] = float(totals.tail(int(col[len(prefix):]))[metric].mean())
        result['PLR_USAGE_PROXY_COVERAGE'] = 1.0

        # Roster inference is based only on prior observations and labeled unconfirmed.
        rotation = []
        for pid, group in history.groupby('PLAYER_ID'):
            all_player = self.players[str(pid)]
            idx = all_player.GAME_DATE.searchsorted(history_day,side='left')
            if idx == 0 or int(all_player.iloc[idx-1].TEAM_ID) != tid:
                continue
            if (pd.Timestamp(day)-pd.Timestamp(group.GAME_DATE.max())).days > 60:
                continue
            recent = group.tail(14)
            played = recent[recent.MIN>0]
            if played.empty:
                continue
            record = {'PLAYER_ID':str(pid),'PLAYER_NAME':group.iloc[-1].PLAYER_NAME,
                      'BASE_MIN':float(recent.MIN.mean()),'CAP':48.0,'PLAY':1.0}
            for stat in ['PTS','REB','AST']:
                record[stat] = float(played[stat].sum()/played.MIN.sum())
            record['USG_WEIGHTED'] = float((played.FGA.sum()+.44*played.FTA.sum()+played.TOV.sum())/played.MIN.sum())
            rotation.append(record)

        def resolve(row):
            by_id=[r for r in rotation if row.get('PLAYER_ID') and r['PLAYER_ID']==row['PLAYER_ID']]
            if len(by_id)==1:
                return by_id[0]
            if row.get('PLAYER_NAME'):
                by_name=[r for r in rotation if name_key(r['PLAYER_NAME'])==name_key(row['PLAYER_NAME'])]
                if len(by_name)==1:
                    return by_name[0]
            return None

        roster, roster_complete, _ = self.report('rosters',tid,day,cutoff)
        if roster_complete:
            resolved=[resolve(row) for row in roster]
            # A different provider ID is safe when its team-scoped name resolves.
            # A new or unresolved player can be retained as a provisional
            # rotation member, but cannot make roster coverage confirmed.
            resolved_ids = {player['PLAYER_ID'] for player in resolved if player is not None}
            safe = len(resolved_ids) > 5 and all(player is not None for player in resolved)
            if safe:
                roster_ids={player['PLAYER_ID'] for player in resolved if player is not None}
                rotation = [r for r in rotation if r['PLAYER_ID'] in roster_ids]
                result['PLR_ROSTER_CONFIRMED']=1.0

        injuries,complete,report_age = self.report('injuries',tid,day,cutoff)
        weights = {'OUT':1.,'DOUBTFUL':.75,'QUESTIONABLE':.5,'PROBABLE':.25,'ACTIVE':0.,'UNKNOWN':.5}
        for key,value in self.settings.get('defaults',{}).get('injury_weights',{}).items():
            weights[key.upper()] = float(value)
        if any(not 0 <= v <= 1 for v in weights.values()):
            raise ValueError('Injury weights must be in [0,1]')
        unresolved = False
        lost = 0.0
        for row in injuries:
            player = resolve(row)
            if player is None:
                unresolved = True
                continue
            status = row['STATUS']
            if status == 'UNKNOWN':
                unresolved=True
            missing = weights.get(status,.5)
            player['PLAY'] = 1-missing
            if row.get('MINUTES_LIMIT') is not None:
                player['CAP'] = row['MINUTES_LIMIT']
            contribution = player['BASE_MIN']*missing
            lost += contribution
            field = {'OUT':'INJ_OUT_MIN_WT','QUESTIONABLE':'INJ_Q_MIN_WT','DOUBTFUL':'INJ_DOUBTFUL_MIN_WT'}.get(status)
            if field:
                result[field] += contribution
        result['INJ_COVERAGE'] = float(complete and not unresolved)
        result['INJ_REPORT_AGE_HOURS'] = report_age
        result['INJ_AVAIL_MIN_WT'] = max(0.,1-lost/240) if result['INJ_COVERAGE'] else 0.
        if rotation:
            projected = allocate_minutes([r['BASE_MIN']*r['PLAY'] for r in rotation],
                                         [r['CAP']*r['PLAY'] for r in rotation])
            for player,minute in zip(rotation,projected):
                player['PROJECTED_MIN']=float(minute)
            result['PLR_PROJECTED_MIN_SUM']=float(projected.sum())
            result['PLR_ROTATION_SIZE']=float((projected>0).sum())
            for stat in ['PTS','REB','AST']:
                result[f'PLR_PROJECTED_{stat}']=sum(r['PROJECTED_MIN']*r[stat] for r in rotation)
        lineup,_,_ = self.report('lineups',tid,day,cutoff)
        starters=[]
        for row in lineup:
            player=resolve(row)
            if player is not None and player['PLAY']>0 and row['STATUS'] in ('CONFIRMED','PROJECTED'):
                if player not in starters:
                    starters.append(player)
                    result['LINEUP_CONFIRMED_STARTERS' if row['STATUS']=='CONFIRMED' else 'LINEUP_PROJECTED_STARTERS']+=1
        if len(starters)>5:
            raise ValueError('More than five starters in a team report')
        result['LINEUP_COVERAGE']=len(starters)/5
        # Complete a partial lineup with eligible projected players, not unavailable stars.
        for player in sorted(rotation,key=lambda r:r.get('PROJECTED_MIN',0),reverse=True):
            if len(starters)>=5:
                break
            if player not in starters and player['PLAY']>0:
                starters.append(player)
        result['LINEUP_EST_MIN']=sum(p.get('PROJECTED_MIN',0) for p in starters)
        for stat in ['PTS','REB','AST','USG_WEIGHTED']:
            result[f'LINEUP_EST_{stat}']=sum(p.get('PROJECTED_MIN',0)*p[stat] for p in starters)
        if not all(math.isfinite(v) for v in result.values()):
            raise ValueError('Non-finite player features')
        self.cache[cache_key]=result
        return dict(result)

    def availability_readiness(self, team, game_date, as_of, require_lineup=True):
        """Return explicit reasons a matchup should remain ineligible for a bet."""
        values = self.team_features(team, game_date, as_of)
        reasons = []
        if values['PLR_GAME_COVERAGE'] < 1:
            reasons.append('player history is missing or stale')
        if values['INJ_COVERAGE'] < 1:
            reasons.append('injury report is missing, stale, or unresolved')
        if values['PLR_ROSTER_CONFIRMED'] < 1:
            reasons.append('game-day roster is missing or has unresolved players')
        if require_lineup and values['LINEUP_CONFIRMED_STARTERS'] < 5:
            reasons.append('starting lineup is incomplete or unconfirmed')
        return {'eligible': not reasons, 'reasons': reasons, 'features': values}

    def matchup_readiness(self, home, away, game_date, as_of, require_lineup=True):
        return {
            'home': self.availability_readiness(home, game_date, as_of, require_lineup),
            'away': self.availability_readiness(away, game_date, as_of, require_lineup),
        }

    def matchup(self, home, away, game_date, as_of=None):
        return {f'{prefix}_{key}':value for prefix,team in [('H',home),('A',away)]
                for key,value in self.team_features(team,game_date,as_of).items()}
