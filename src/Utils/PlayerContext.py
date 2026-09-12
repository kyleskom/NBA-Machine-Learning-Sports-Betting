import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config.toml"
PLAYER_DB_PATH = BASE_DIR / "Data" / "PlayerData.sqlite"
PLAYER_FEATURES_TABLE = "player_team_features"

DEFAULT_PLAYER_WINDOWS = [7, 14, 30]
PLAYER_ROLL_METRICS = ("MIN", "PTS", "REB", "AST", "USG_WEIGHTED")
PLAYER_BASE_METRICS = [
    "PLR_MIN_SUM",
    "PLR_PTS_SUM",
    "PLR_REB_SUM",
    "PLR_AST_SUM",
    "PLR_USG_WEIGHTED_SUM",
]
PLAYER_INJURY_METRICS = [
    "INJ_OUT_MIN_WT",
    "INJ_Q_MIN_WT",
    "INJ_DOUBTFUL_MIN_WT",
    "INJ_AVAIL_MIN_WT",
    "INJ_COVERAGE",
]
PLAYER_LINEUP_METRICS = [
    "LINEUP_EXPECTED_STARTERS",
    "LINEUP_CONFIRMED_STARTERS",
    "LINEUP_PROJECTED_STARTERS",
    "LINEUP_COVERAGE",
    "LINEUP_EST_MIN",
    "LINEUP_EST_PTS",
    "LINEUP_EST_REB",
    "LINEUP_EST_AST",
    "LINEUP_EST_USG_WEIGHTED",
]


def table_exists(con, table_name):
    cursor = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def normalize_date(value):
    if isinstance(value, datetime):
        return value.date().isoformat()
    if hasattr(value, "date"):
        try:
            return value.date().isoformat()
        except Exception:
            pass
    return str(value)


def get_player_feature_columns(config):
    settings = config.get("player-features", {})
    windows = settings.get("rolling_windows", DEFAULT_PLAYER_WINDOWS)
    windows = sorted({int(window) for window in windows if int(window) > 0})
    if not windows:
        windows = DEFAULT_PLAYER_WINDOWS

    columns = list(PLAYER_BASE_METRICS)
    for window in windows:
        columns.extend([f"PLR_{metric}_ROLL{window}" for metric in PLAYER_ROLL_METRICS])
    columns.extend(PLAYER_INJURY_METRICS)
    columns.extend(PLAYER_LINEUP_METRICS)
    return columns


def get_player_defaults(config, feature_cols=None):
    if feature_cols is None:
        feature_cols = get_player_feature_columns(config)

    settings = config.get("player-features", {})
    defaults = settings.get("defaults", {})
    values = {f"H_{field}": 0.0 for field in feature_cols}
    values.update({f"A_{field}": 0.0 for field in feature_cols})

    for prefix in ("H", "A"):
        values[f"{prefix}_INJ_AVAIL_MIN_WT"] = float(defaults.get("default_inj_avail_min_wt", 1.0))
        values[f"{prefix}_INJ_COVERAGE"] = float(defaults.get("default_inj_coverage", 0.0))
        values[f"{prefix}_LINEUP_EXPECTED_STARTERS"] = float(defaults.get("default_lineup_expected_starters", 5.0))
        values[f"{prefix}_LINEUP_COVERAGE"] = float(defaults.get("default_lineup_coverage", 0.0))
    return values


def _coerce_feature_value(value):
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_player_feature_lookup(player_con, table_name=PLAYER_FEATURES_TABLE, feature_cols=None):
    if feature_cols is None:
        feature_cols = []
    if not table_exists(player_con, table_name):
        return {"exact": {}, "history": {}}

    df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', player_con)
    if df.empty:
        return {"exact": {}, "history": {}}

    for column in feature_cols:
        if column not in df.columns:
            df[column] = 0.0

    exact = {}
    history = {}
    for row in df.itertuples(index=False):
        date_str = normalize_date(getattr(row, "Date", ""))
        team_name = getattr(row, "TEAM_NAME", None)
        if team_name is None:
            continue
        values = {field: _coerce_feature_value(getattr(row, field, 0.0)) for field in feature_cols}
        exact[(date_str, team_name)] = values
        history.setdefault(team_name, []).append((date_str, values))

    for team_name in history:
        history[team_name].sort(key=lambda item: item[0])
    return {"exact": exact, "history": history}


def load_player_feature_map(player_con, table_name, feature_cols):
    return load_player_feature_lookup(player_con, table_name, feature_cols)["exact"]


def _resolve_team_features(feature_lookup, date_str, team_name, allow_asof=False):
    exact = feature_lookup.get("exact", {})
    values = exact.get((date_str, team_name))
    if values is not None or not allow_asof:
        return values or {}

    selected = {}
    for candidate_date, candidate_values in feature_lookup.get("history", {}).get(team_name, []):
        if candidate_date <= date_str:
            selected = candidate_values
        else:
            break
    return selected


def build_player_feature_row(
    feature_lookup,
    date_str,
    home_team,
    away_team,
    feature_cols,
    default_values,
    allow_asof=False,
):
    values = dict(default_values)
    home_values = _resolve_team_features(feature_lookup, date_str, home_team, allow_asof=allow_asof)
    away_values = _resolve_team_features(feature_lookup, date_str, away_team, allow_asof=allow_asof)

    for field in feature_cols:
        if field in home_values:
            values[f"H_{field}"] = home_values[field]
        if field in away_values:
            values[f"A_{field}"] = away_values[field]
    return values


def load_player_context(config, db_path=PLAYER_DB_PATH):
    feature_cols = get_player_feature_columns(config)
    defaults = get_player_defaults(config, feature_cols)
    if not db_path.exists():
        return feature_cols, defaults, {"exact": {}, "history": {}}
    with sqlite3.connect(db_path) as con:
        lookup = load_player_feature_lookup(con, PLAYER_FEATURES_TABLE, feature_cols)
    return feature_cols, defaults, lookup


def align_frame_to_columns(frame, feature_columns, fill_value=0.0):
    aligned = frame.copy()
    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = fill_value
    extra_columns = [column for column in aligned.columns if column not in feature_columns]
    if extra_columns:
        aligned = aligned.drop(columns=extra_columns)
    return aligned[feature_columns].astype(float)


def feature_sidecar_path(model_path):
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}_features.json")


def save_feature_columns(model_path, feature_columns):
    path = feature_sidecar_path(model_path)
    with path.open("w") as handle:
        json.dump(list(feature_columns), handle, indent=2)
    return path


def load_feature_columns(model_path):
    path = feature_sidecar_path(model_path)
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)
