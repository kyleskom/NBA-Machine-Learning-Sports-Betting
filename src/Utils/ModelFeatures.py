"""Persist and validate the team-stat feature contract for trained models."""
import json
from pathlib import Path

import numpy as np

FEATURE_VERSION = 'team_games_v1'
PLAYER_PREFIXES = ('H_PLR_', 'A_PLR_', 'H_INJ_', 'A_INJ_', 'H_LINEUP_', 'A_LINEUP_')


def team_only_frame(frame):
    return frame.drop(columns=[column for column in frame if column.startswith(PLAYER_PREFIXES)])


def align_frame_to_columns(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required model features: {sorted(missing)}')
    aligned = frame.loc[:, columns].astype(float)
    if not np.isfinite(aligned.to_numpy()).all():
        raise ValueError('Model features contain non-finite values')
    return aligned


def feature_sidecar_path(model_path):
    path = Path(model_path)
    return path.with_name(f'{path.stem}_features.json')


def save_feature_columns(model_path, columns):
    if any(column.startswith(PLAYER_PREFIXES) for column in columns):
        raise ValueError('Team models cannot include player or availability features')
    path = feature_sidecar_path(model_path)
    path.write_text(json.dumps({'version': FEATURE_VERSION, 'columns': list(columns)}, indent=2) + '\n')
    return path


def load_feature_columns(model_path):
    path = feature_sidecar_path(model_path)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get('version') != FEATURE_VERSION:
        raise ValueError('A team-only model feature manifest is required')
    columns = data['columns']
    if any(column.startswith(PLAYER_PREFIXES) for column in columns):
        raise ValueError('Player-aware models are no longer used for prediction')
    return columns
