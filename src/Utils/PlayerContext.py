"""Shared player feature contract used by training and inference."""
import json
from pathlib import Path
import numpy as np
from src.Players.features import PlayerFeatureEngine, feature_columns, defaults
from src.Players.store import DB_PATH

PLAYER_DB_PATH=DB_PATH
PLAYER_FEATURES_TABLE='player_team_features_v2'
FEATURE_VERSION='player_games_v2'


def get_player_feature_columns(config):
    return feature_columns(config)


def get_player_defaults(config, feature_cols=None):
    values=defaults(config)
    return {f'{prefix}_{field}':values.get(field,0.) for prefix in ('H','A') for field in (feature_cols or feature_columns(config))}


def load_player_context(config,db_path=PLAYER_DB_PATH):
    return feature_columns(config),get_player_defaults(config),PlayerFeatureEngine.from_database(db_path,config)


def build_player_feature_row(feature_lookup,date_str,home_team,away_team,feature_cols,default_values,allow_asof=False,as_of=None):
    if not isinstance(feature_lookup,PlayerFeatureEngine):
        raise ValueError('Legacy daily player snapshots are unsupported; ingest game logs first')
    return feature_lookup.matchup(home_team,away_team,date_str,as_of)


def align_frame_to_columns(frame,feature_columns,fill_value=None):
    missing=set(feature_columns)-set(frame.columns)
    if missing:
        raise ValueError(f'Missing required model features: {sorted(missing)}')
    aligned=frame.loc[:,feature_columns].astype(float)
    if not np.isfinite(aligned.to_numpy()).all():
        raise ValueError('Model features contain non-finite values')
    return aligned


def feature_sidecar_path(model_path):
    p=Path(model_path)
    return p.with_name(f'{p.stem}_features.json')


def save_feature_columns(model_path,columns):
    path=feature_sidecar_path(model_path)
    path.write_text(json.dumps({'version':FEATURE_VERSION,'columns':list(columns)},indent=2))
    return path


def load_feature_columns(model_path):
    path=feature_sidecar_path(model_path)
    if not path.exists():
        return None
    data=json.loads(path.read_text())
    if isinstance(data,list):
        if any(c.startswith(('H_PLR_','A_PLR_')) for c in data):
            raise ValueError('Legacy player feature definitions require retraining')
        return data
    if data.get('version')!=FEATURE_VERSION:
        raise ValueError('Model player feature version mismatch; retrain')
    return data['columns']
