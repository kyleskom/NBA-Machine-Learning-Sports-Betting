import argparse
import os
import random
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import toml

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.fspath(BASE_DIR))

from src.Utils.tools import get_json_data, to_data_frame  # noqa: E402

CONFIG_PATH = BASE_DIR / "config.toml"
DB_PATH = BASE_DIR / "Data" / "PlayerData.sqlite"

PLAYER_STATS_TABLE = "player_stats_raw"
INJURY_TABLE = "injury_reports_raw"
INJURY_FETCH_LOG_TABLE = "injury_fetch_log"
STARTING_LINEUPS_TABLE = "starting_lineups_raw"
LINEUP_FETCH_LOG_TABLE = "lineup_fetch_log"
TEAM_FEATURES_TABLE = "player_team_features"

MIN_DELAY_SECONDS = 1
MAX_DELAY_SECONDS = 3
MAX_RETRIES = 3
NEUTRAL_PLAYER_MINUTES = 16.0
NEUTRAL_PLAYER_USAGE = 0.18
NEUTRAL_TEAM_MINUTES = 240.0

INJURY_STATUS_MAP = {
    "out": "OUT",
    "o": "OUT",
    "doubtful": "DOUBTFUL",
    "questionable": "QUESTIONABLE",
    "probable": "PROBABLE",
    "day-to-day": "QUESTIONABLE",
    "healthy": "ACTIVE",
    "active": "ACTIVE",
    "available": "ACTIVE",
}

LINEUP_STATUS_MAP = {
    "confirmed": "CONFIRMED",
    "active": "CONFIRMED",
    "starting": "CONFIRMED",
    "projected": "PROJECTED",
    "starter": "STARTER",
    "probable": "PROJECTED",
}


def load_config():
    return toml.load(CONFIG_PATH)


def iter_dates(start_date, end_date):
    cursor = start_date
    while cursor <= end_date:
        yield cursor
        cursor += timedelta(days=1)


def table_exists(con, table_name):
    cursor = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def parse_date_value(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(value), fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


def normalize_injury_status(status):
    if status is None:
        return "UNKNOWN"
    raw = str(status).strip().lower()
    if raw in INJURY_STATUS_MAP:
        return INJURY_STATUS_MAP[raw]
    for key, mapped in INJURY_STATUS_MAP.items():
        if key in raw:
            return mapped
    return str(status).strip().upper() or "UNKNOWN"


def normalize_lineup_status(status):
    if status is None:
        return "STARTER"
    raw = str(status).strip().lower()
    if raw in LINEUP_STATUS_MAP:
        return LINEUP_STATUS_MAP[raw]
    for key, mapped in LINEUP_STATUS_MAP.items():
        if key in raw:
            return mapped
    return str(status).strip().upper() or "STARTER"


def as_float(value, default=0.0):
    if value is None:
        return default
    try:
        if isinstance(value, str) and ":" in value:
            minutes, seconds = value.split(":", 1)
            return float(minutes) + (float(seconds) / 60.0)
        return float(value)
    except (TypeError, ValueError):
        return default


def get_value(data, keys, default=None):
    for key in keys:
        if isinstance(data, dict) and key in data:
            return data.get(key)
    return default


def _coerce_name(value):
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def parse_injury_payload(payload, date_value, source_name):
    if payload is None:
        return pd.DataFrame(columns=["Date", "PLAYER_NAME", "TEAM_NAME", "STATUS", "SOURCE"])

    rows = None
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        for key in ("injuries", "data", "result", "players", "items"):
            if isinstance(payload.get(key), list):
                rows = payload.get(key)
                break
    if rows is None:
        rows = []

    parsed = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        player = get_value(
            row,
            ("player_name", "full_name", "name", "player", "athlete"),
        )
        if isinstance(player, dict):
            player = get_value(player, ("full_name", "display_name", "name"))

        team = get_value(
            row,
            ("team_name", "team", "team_display_name", "teamName", "team_abbreviation"),
        )
        if isinstance(team, dict):
            team = get_value(team, ("full_name", "display_name", "name", "abbreviation"))

        status = get_value(row, ("status", "injury_status", "designation"))
        if status is None and isinstance(row.get("injury"), dict):
            status = get_value(row["injury"], ("status", "designation"))

        player_name = _coerce_name(player)
        if not player_name:
            continue

        parsed.append(
            {
                "Date": date_value.isoformat(),
                "PLAYER_NAME": player_name,
                "TEAM_NAME": _coerce_name(team),
                "STATUS": normalize_injury_status(status),
                "SOURCE": source_name,
            }
        )

    return pd.DataFrame(parsed)


def parse_lineup_payload(payload, date_value, source_name):
    columns = ["Date", "PLAYER_NAME", "TEAM_NAME", "STATUS", "SOURCE"]
    if payload is None:
        return pd.DataFrame(columns=columns)

    rows = None
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        for key in ("lineups", "starting_lineups", "starters", "data", "result", "players", "items"):
            if isinstance(payload.get(key), list):
                rows = payload.get(key)
                break
    if rows is None:
        rows = []

    parsed = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        player = get_value(row, ("player_name", "full_name", "name", "player", "athlete"))
        if isinstance(player, dict):
            player = get_value(player, ("full_name", "display_name", "name"))

        team = get_value(row, ("team_name", "team", "team_display_name", "teamName", "team_abbreviation"))
        if isinstance(team, dict):
            team = get_value(team, ("full_name", "display_name", "name", "abbreviation"))

        status = get_value(row, ("status", "lineup_status", "designation", "type"))
        player_name = _coerce_name(player)
        if not player_name:
            continue

        parsed.append(
            {
                "Date": date_value.isoformat(),
                "PLAYER_NAME": player_name,
                "TEAM_NAME": _coerce_name(team),
                "STATUS": normalize_lineup_status(status),
                "SOURCE": source_name,
            }
        )

    return pd.DataFrame(parsed, columns=columns)


def fetch_json(url):
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return response.json(), True
    except Exception as exc:
        print(f"Failed to fetch JSON from {url}: {exc}")
        return None, False


def _read_lineup_csv(path, date_pointer):
    csv_path = Path(path).expanduser()
    if not csv_path.is_absolute():
        csv_path = BASE_DIR / csv_path
    if not csv_path.exists():
        return pd.DataFrame(columns=["Date", "PLAYER_NAME", "TEAM_NAME", "STATUS", "SOURCE"]), False

    df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame(columns=["Date", "PLAYER_NAME", "TEAM_NAME", "STATUS", "SOURCE"]), True

    rename_map = {}
    for column in df.columns:
        normalized = column.strip().lower()
        if normalized in {"date", "game_date"}:
            rename_map[column] = "Date"
        elif normalized in {"player", "player_name", "full_name", "name"}:
            rename_map[column] = "PLAYER_NAME"
        elif normalized in {"team", "team_name", "team_abbreviation"}:
            rename_map[column] = "TEAM_NAME"
        elif normalized in {"status", "lineup_status", "designation", "type"}:
            rename_map[column] = "STATUS"
    df = df.rename(columns=rename_map)
    for column in ("Date", "PLAYER_NAME", "TEAM_NAME", "STATUS"):
        if column not in df.columns:
            df[column] = None

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date.astype(str)
    df = df[df["Date"] == date_pointer.isoformat()].copy()
    if df.empty:
        return pd.DataFrame(columns=["Date", "PLAYER_NAME", "TEAM_NAME", "STATUS", "SOURCE"]), True

    df["PLAYER_NAME"] = df["PLAYER_NAME"].astype(str).str.strip()
    df["TEAM_NAME"] = df["TEAM_NAME"].astype(str).str.strip()
    df["STATUS"] = df["STATUS"].apply(normalize_lineup_status)
    df["SOURCE"] = "csv"
    df = df[df["PLAYER_NAME"] != ""]
    return df[["Date", "PLAYER_NAME", "TEAM_NAME", "STATUS", "SOURCE"]], True


def fetch_injuries_for_date(config, date_pointer):
    settings = config.get("player-injuries", {})
    primary_url = settings.get("primary_url", "").strip()
    secondary_url = settings.get("secondary_url", "").strip()

    dfs = []
    primary_ok = False
    secondary_ok = False

    if primary_url:
        payload, primary_ok = fetch_json(primary_url.format(date_pointer.isoformat()))
        if primary_ok:
            dfs.append(parse_injury_payload(payload, date_pointer, "primary"))

    if (not dfs or dfs[-1].empty) and secondary_url:
        payload, secondary_ok = fetch_json(secondary_url.format(date_pointer.isoformat()))
        if secondary_ok:
            dfs.append(parse_injury_payload(payload, date_pointer, "secondary"))

    if not dfs:
        return pd.DataFrame(columns=["Date", "PLAYER_NAME", "TEAM_NAME", "STATUS", "SOURCE"]), primary_ok, secondary_ok

    injury_df = pd.concat(dfs, ignore_index=True)
    if injury_df.empty:
        return injury_df, primary_ok, secondary_ok
    return injury_df.drop_duplicates(
        subset=["Date", "PLAYER_NAME", "TEAM_NAME", "STATUS"],
        keep="last",
    ), primary_ok, secondary_ok


def fetch_lineups_for_date(config, date_pointer):
    settings = config.get("player-lineups", {})
    primary_url = settings.get("primary_url", "").strip()
    secondary_url = settings.get("secondary_url", "").strip()
    csv_path = settings.get("csv_path", "").strip()

    dfs = []
    primary_ok = False
    secondary_ok = False
    csv_ok = False

    if primary_url:
        payload, primary_ok = fetch_json(primary_url.format(date_pointer.isoformat()))
        if primary_ok:
            dfs.append(parse_lineup_payload(payload, date_pointer, "primary"))

    if (not dfs or dfs[-1].empty) and secondary_url:
        payload, secondary_ok = fetch_json(secondary_url.format(date_pointer.isoformat()))
        if secondary_ok:
            dfs.append(parse_lineup_payload(payload, date_pointer, "secondary"))

    if (not dfs or dfs[-1].empty) and csv_path:
        csv_df, csv_ok = _read_lineup_csv(csv_path, date_pointer)
        if csv_ok:
            dfs.append(csv_df)

    if not dfs:
        return (
            pd.DataFrame(columns=["Date", "PLAYER_NAME", "TEAM_NAME", "STATUS", "SOURCE"]),
            primary_ok,
            secondary_ok,
            csv_ok,
        )

    lineup_df = pd.concat(dfs, ignore_index=True)
    if lineup_df.empty:
        return lineup_df, primary_ok, secondary_ok, csv_ok
    return lineup_df.drop_duplicates(
        subset=["Date", "PLAYER_NAME", "TEAM_NAME"],
        keep="last",
    ), primary_ok, secondary_ok, csv_ok


def fetch_player_stats(url, date_pointer, start_year, season_key):
    for attempt in range(1, MAX_RETRIES + 1):
        raw_data = get_json_data(
            url.format(date_pointer.month, date_pointer.day, start_year, date_pointer.year, season_key)
        )
        df = to_data_frame(raw_data)
        if not df.empty:
            df = df.copy()
            df["Date"] = date_pointer.isoformat()
            df["Season"] = season_key
            return df
        if attempt < MAX_RETRIES:
            time.sleep(MIN_DELAY_SECONDS + random.random() * (MAX_DELAY_SECONDS - MIN_DELAY_SECONDS))
    return pd.DataFrame()


def select_current_season(config, today):
    seasons = config.get("get-player-data", {})
    if not seasons:
        seasons = config.get("get-data", {})
    for season_key, value in seasons.items():
        start_date = datetime.strptime(value["start_date"], "%Y-%m-%d").date()
        end_date = datetime.strptime(value["end_date"], "%Y-%m-%d").date()
        if start_date <= today <= end_date:
            return season_key, value, start_date, end_date
    return None, None, None, None


def season_items(config, season=None):
    seasons = config.get("get-player-data", {})
    if not seasons:
        seasons = config.get("get-data", {})
    items = list(seasons.items())
    if season:
        items = [(key, value) for key, value in items if key == season]
    return items


def get_existing_dates(con):
    if not table_exists(con, PLAYER_STATS_TABLE):
        return set()
    cursor = con.execute(f'SELECT DISTINCT Date FROM "{PLAYER_STATS_TABLE}"')
    found = set()
    for (value,) in cursor.fetchall():
        parsed = parse_date_value(value)
        if parsed:
            found.add(parsed)
    return found


def delete_date_rows(con, table_name, date_pointer):
    if table_exists(con, table_name):
        con.execute(f'DELETE FROM "{table_name}" WHERE Date = ?', (date_pointer.isoformat(),))


def append_injury_fetch_log(con, date_pointer, primary_ok, secondary_ok, record_count):
    row = pd.DataFrame(
        [
            {
                "Date": date_pointer.isoformat(),
                "Primary_Success": int(primary_ok),
                "Secondary_Success": int(secondary_ok),
                "Records": int(record_count),
            }
        ]
    )
    if table_exists(con, INJURY_FETCH_LOG_TABLE):
        con.execute(f'DELETE FROM "{INJURY_FETCH_LOG_TABLE}" WHERE Date = ?', (date_pointer.isoformat(),))
    row.to_sql(INJURY_FETCH_LOG_TABLE, con, if_exists="append", index=False)


def append_lineup_fetch_log(con, date_pointer, primary_ok, secondary_ok, csv_ok, record_count):
    row = pd.DataFrame(
        [
            {
                "Date": date_pointer.isoformat(),
                "Primary_Success": int(primary_ok),
                "Secondary_Success": int(secondary_ok),
                "CSV_Success": int(csv_ok),
                "Records": int(record_count),
            }
        ]
    )
    if table_exists(con, LINEUP_FETCH_LOG_TABLE):
        con.execute(f'DELETE FROM "{LINEUP_FETCH_LOG_TABLE}" WHERE Date = ?', (date_pointer.isoformat(),))
    row.to_sql(LINEUP_FETCH_LOG_TABLE, con, if_exists="append", index=False)


def ingest_date(con, config, url, season_key, value, date_pointer):
    print("Getting player data:", date_pointer)
    stats_df = fetch_player_stats(url, date_pointer, value["start_year"], season_key)
    if stats_df.empty:
        print("No player stats returned for:", date_pointer)
        return False

    injuries_df, primary_ok, secondary_ok = fetch_injuries_for_date(config, date_pointer)
    lineups_df, lineup_primary_ok, lineup_secondary_ok, lineup_csv_ok = fetch_lineups_for_date(config, date_pointer)

    delete_date_rows(con, PLAYER_STATS_TABLE, date_pointer)
    stats_df.to_sql(PLAYER_STATS_TABLE, con, if_exists="append", index=False)

    delete_date_rows(con, INJURY_TABLE, date_pointer)
    if not injuries_df.empty:
        injuries_df.to_sql(INJURY_TABLE, con, if_exists="append", index=False)

    append_injury_fetch_log(con, date_pointer, primary_ok, secondary_ok, len(injuries_df.index))
    delete_date_rows(con, STARTING_LINEUPS_TABLE, date_pointer)
    if not lineups_df.empty:
        lineups_df.to_sql(STARTING_LINEUPS_TABLE, con, if_exists="append", index=False)
    append_lineup_fetch_log(
        con,
        date_pointer,
        lineup_primary_ok,
        lineup_secondary_ok,
        lineup_csv_ok,
        len(lineups_df.index),
    )
    return True


def _prepare_stats_frame(df):
    col_map = {
        "PLAYER_ID": "PLAYER_ID",
        "PLAYER_NAME": "PLAYER_NAME",
        "TEAM_NAME": "TEAM_NAME",
        "Date": "Date",
    }
    for required in ("MIN", "PTS", "REB", "AST"):
        if required not in df.columns:
            df[required] = 0.0

    usage_candidates = ("USG_PCT", "PCT_UAST_2PM", "PCT_UAST_3PM")
    usage_col = next((name for name in usage_candidates if name in df.columns), None)
    if usage_col is None:
        df["USG_PROXY"] = NEUTRAL_PLAYER_USAGE
        usage_col = "USG_PROXY"
    col_map[usage_col] = "USG"

    out = df.rename(columns=col_map).copy()
    if "PLAYER_ID" not in out.columns:
        out["PLAYER_ID"] = pd.NA
    if "PLAYER_NAME" not in out.columns:
        out["PLAYER_NAME"] = ""
    if "TEAM_NAME" not in out.columns:
        out["TEAM_NAME"] = ""
    if "Date" not in out.columns:
        out["Date"] = ""

    out["PLAYER_NAME"] = out["PLAYER_NAME"].astype(str).str.strip()
    out["TEAM_NAME"] = out["TEAM_NAME"].astype(str).str.strip()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])

    for field in ("MIN", "PTS", "REB", "AST", "USG"):
        out[field] = out[field].apply(as_float)

    if out["USG"].max() > 1.0:
        out["USG"] = out["USG"] / 100.0
    out["USG"] = out["USG"].clip(lower=0.0, upper=1.0)

    return out


def _rolling_team_features(team_daily, window_size):
    sort_df = team_daily.sort_values(["TEAM_NAME", "Date"]).copy()
    for metric in ("MIN", "PTS", "REB", "AST", "USG_WEIGHTED"):
        field_name = f"PLR_{metric}_ROLL{window_size}"
        sort_df[field_name] = (
            sort_df.groupby("TEAM_NAME")[metric]
            .transform(lambda s: s.rolling(window_size, min_periods=1).mean())
        )
    return sort_df


def _build_player_baselines(stats):
    sort_df = stats.sort_values(["PLAYER_NAME", "Date"]).copy()
    sort_df["BASE_MIN"] = (
        sort_df.groupby("PLAYER_NAME")["MIN"]
        .transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
        .fillna(NEUTRAL_PLAYER_MINUTES)
    )
    for metric in ("PTS", "REB", "AST", "USG_WEIGHTED"):
        sort_df[f"BASE_{metric}"] = (
            sort_df.groupby("PLAYER_NAME")[metric]
            .transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
            .fillna(0.0)
        )
    sort_df["BASE_USG"] = (
        sort_df.groupby("PLAYER_NAME")["USG"]
        .transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
        .fillna(NEUTRAL_PLAYER_USAGE)
    )
    return sort_df[
        [
            "Date",
            "PLAYER_NAME",
            "TEAM_NAME",
            "BASE_MIN",
            "BASE_PTS",
            "BASE_REB",
            "BASE_AST",
            "BASE_USG_WEIGHTED",
            "BASE_USG",
        ]
    ]


def _injury_weights(settings):
    return {
        "OUT": as_float(settings.get("out", 1.0), 1.0),
        "DOUBTFUL": as_float(settings.get("doubtful", 0.75), 0.75),
        "QUESTIONABLE": as_float(settings.get("questionable", 0.5), 0.5),
        "PROBABLE": as_float(settings.get("probable", 0.25), 0.25),
        "ACTIVE": as_float(settings.get("active", 0.0), 0.0),
        "UNKNOWN": as_float(settings.get("unknown", 0.5), 0.5),
    }


def _build_injury_features(con, baselines, defaults):
    if not table_exists(con, INJURY_TABLE):
        return pd.DataFrame(columns=[
            "Date",
            "TEAM_NAME",
            "INJ_OUT_MIN_WT",
            "INJ_Q_MIN_WT",
            "INJ_DOUBTFUL_MIN_WT",
        ])

    injuries = pd.read_sql_query(f'SELECT * FROM "{INJURY_TABLE}"', con)
    if injuries.empty:
        return pd.DataFrame(columns=[
            "Date",
            "TEAM_NAME",
            "INJ_OUT_MIN_WT",
            "INJ_Q_MIN_WT",
            "INJ_DOUBTFUL_MIN_WT",
        ])

    injuries["Date"] = pd.to_datetime(injuries["Date"], errors="coerce")
    injuries["PLAYER_NAME"] = injuries["PLAYER_NAME"].astype(str).str.strip()
    injuries["TEAM_NAME"] = injuries["TEAM_NAME"].astype(str).str.strip()
    injuries["STATUS"] = injuries["STATUS"].apply(normalize_injury_status)
    injuries = injuries.dropna(subset=["Date", "PLAYER_NAME"])

    merged = injuries.merge(
        baselines,
        on=["Date", "PLAYER_NAME"],
        how="left",
        suffixes=("", "_base"),
    )

    merged["TEAM_NAME"] = merged["TEAM_NAME"].where(
        merged["TEAM_NAME"].notna() & (merged["TEAM_NAME"] != ""),
        merged["TEAM_NAME_base"],
    )
    merged["BASE_MIN"] = merged["BASE_MIN"].fillna(as_float(defaults.get("neutral_player_minutes", NEUTRAL_PLAYER_MINUTES)))
    merged["BASE_USG"] = merged["BASE_USG"].fillna(as_float(defaults.get("neutral_player_usage", NEUTRAL_PLAYER_USAGE)))

    weights = _injury_weights(defaults.get("injury_weights", {}))
    merged["weight"] = merged["STATUS"].map(weights).fillna(weights["UNKNOWN"])

    merged["OUT_WT"] = merged.apply(lambda row: row["BASE_MIN"] if row["STATUS"] == "OUT" else 0.0, axis=1)
    merged["Q_WT"] = merged.apply(lambda row: row["BASE_MIN"] * weights["QUESTIONABLE"] if row["STATUS"] == "QUESTIONABLE" else 0.0, axis=1)
    merged["D_WT"] = merged.apply(lambda row: row["BASE_MIN"] * weights["DOUBTFUL"] if row["STATUS"] == "DOUBTFUL" else 0.0, axis=1)

    grouped = (
        merged.groupby(["Date", "TEAM_NAME"], as_index=False)
        .agg(
            INJ_OUT_MIN_WT=("OUT_WT", "sum"),
            INJ_Q_MIN_WT=("Q_WT", "sum"),
            INJ_DOUBTFUL_MIN_WT=("D_WT", "sum"),
        )
    )
    grouped["Date"] = grouped["Date"].dt.date.astype(str)
    return grouped


def _build_injury_coverage(con):
    if not table_exists(con, INJURY_FETCH_LOG_TABLE):
        return {}
    log_df = pd.read_sql_query(f'SELECT * FROM "{INJURY_FETCH_LOG_TABLE}"', con)
    if log_df.empty:
        return {}
    result = {}
    for row in log_df.itertuples(index=False):
        parsed = parse_date_value(row.Date)
        if parsed is None:
            continue
        ok = bool(getattr(row, "Primary_Success", 0) or getattr(row, "Secondary_Success", 0))
        result[parsed.isoformat()] = 1.0 if ok else 0.0
    return result


def _estimated_lineup_features(baselines):
    columns = [
        "Date",
        "TEAM_NAME",
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
    if baselines.empty:
        return pd.DataFrame(columns=columns)

    eligible = baselines.dropna(subset=["Date", "TEAM_NAME"]).copy()
    eligible = eligible[eligible["TEAM_NAME"].astype(str).str.strip() != ""]
    if eligible.empty:
        return pd.DataFrame(columns=columns)

    eligible = eligible.sort_values(["Date", "TEAM_NAME", "BASE_MIN"], ascending=[True, True, False])
    starters = eligible.groupby(["Date", "TEAM_NAME"], as_index=False).head(5)
    grouped = (
        starters.groupby(["Date", "TEAM_NAME"], as_index=False)
        .agg(
            LINEUP_EST_MIN=("BASE_MIN", "sum"),
            LINEUP_EST_PTS=("BASE_PTS", "sum"),
            LINEUP_EST_REB=("BASE_REB", "sum"),
            LINEUP_EST_AST=("BASE_AST", "sum"),
            LINEUP_EST_USG_WEIGHTED=("BASE_USG_WEIGHTED", "sum"),
        )
    )
    grouped["LINEUP_EXPECTED_STARTERS"] = 5.0
    grouped["LINEUP_CONFIRMED_STARTERS"] = 0.0
    grouped["LINEUP_PROJECTED_STARTERS"] = 0.0
    grouped["LINEUP_COVERAGE"] = 0.0
    return grouped[columns]


def _build_lineup_features(con, baselines):
    fallback = _estimated_lineup_features(baselines)
    if not table_exists(con, STARTING_LINEUPS_TABLE):
        return fallback

    lineups = pd.read_sql_query(f'SELECT * FROM "{STARTING_LINEUPS_TABLE}"', con)
    if lineups.empty:
        return fallback

    lineups["Date"] = pd.to_datetime(lineups["Date"], errors="coerce")
    lineups["PLAYER_NAME"] = lineups["PLAYER_NAME"].astype(str).str.strip()
    lineups["TEAM_NAME"] = lineups["TEAM_NAME"].astype(str).str.strip()
    lineups["STATUS"] = lineups["STATUS"].apply(normalize_lineup_status)
    lineups = lineups.dropna(subset=["Date", "PLAYER_NAME"])

    merged = lineups.merge(
        baselines,
        on=["Date", "PLAYER_NAME"],
        how="left",
        suffixes=("", "_base"),
    )
    merged["TEAM_NAME"] = merged["TEAM_NAME"].where(
        merged["TEAM_NAME"].notna() & (merged["TEAM_NAME"] != ""),
        merged["TEAM_NAME_base"],
    )
    merged = merged.dropna(subset=["TEAM_NAME"])
    merged = merged[merged["TEAM_NAME"].astype(str).str.strip() != ""]

    for column, default in (
        ("BASE_MIN", NEUTRAL_PLAYER_MINUTES),
        ("BASE_PTS", 0.0),
        ("BASE_REB", 0.0),
        ("BASE_AST", 0.0),
        ("BASE_USG_WEIGHTED", NEUTRAL_PLAYER_MINUTES * NEUTRAL_PLAYER_USAGE),
    ):
        merged[column] = merged[column].fillna(default)

    merged["IS_CONFIRMED"] = (merged["STATUS"] == "CONFIRMED").astype(float)
    merged["IS_PROJECTED"] = merged["STATUS"].isin(["PROJECTED", "STARTER"]).astype(float)
    explicit = (
        merged.groupby(["Date", "TEAM_NAME"], as_index=False)
        .agg(
            LINEUP_PLAYER_COUNT=("PLAYER_NAME", "nunique"),
            LINEUP_CONFIRMED_STARTERS=("IS_CONFIRMED", "sum"),
            LINEUP_PROJECTED_STARTERS=("IS_PROJECTED", "sum"),
            LINEUP_EST_MIN=("BASE_MIN", "sum"),
            LINEUP_EST_PTS=("BASE_PTS", "sum"),
            LINEUP_EST_REB=("BASE_REB", "sum"),
            LINEUP_EST_AST=("BASE_AST", "sum"),
            LINEUP_EST_USG_WEIGHTED=("BASE_USG_WEIGHTED", "sum"),
        )
    )
    explicit["LINEUP_EXPECTED_STARTERS"] = 5.0
    explicit["LINEUP_COVERAGE"] = (explicit["LINEUP_PLAYER_COUNT"] / 5.0).clip(lower=0.0, upper=1.0)
    explicit = explicit.drop(columns=["LINEUP_PLAYER_COUNT"])

    columns = list(fallback.columns) if not fallback.empty else [
        "Date",
        "TEAM_NAME",
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
    fallback["Date"] = pd.to_datetime(fallback["Date"], errors="coerce")
    combined = fallback.set_index(["Date", "TEAM_NAME"]) if not fallback.empty else pd.DataFrame(columns=columns).set_index(["Date", "TEAM_NAME"])
    explicit = explicit[columns].set_index(["Date", "TEAM_NAME"])
    combined.update(explicit)
    missing = explicit.loc[~explicit.index.isin(combined.index)]
    if not missing.empty:
        combined = pd.concat([combined, missing])
    combined = combined.reset_index()
    return combined[columns]


def rebuild_team_features(con, config):
    if not table_exists(con, PLAYER_STATS_TABLE):
        print("No player stats table found. Skipping derived feature rebuild.")
        return

    raw_stats = pd.read_sql_query(f'SELECT * FROM "{PLAYER_STATS_TABLE}"', con)
    if raw_stats.empty:
        print("Player stats table is empty. Skipping derived feature rebuild.")
        return

    settings = config.get("player-features", {})
    defaults = settings.get("defaults", {})
    windows = settings.get("rolling_windows", [7, 14, 30])
    windows = [int(window) for window in windows]
    windows = sorted({window for window in windows if window > 0})
    if not windows:
        windows = [7, 14, 30]

    stats = _prepare_stats_frame(raw_stats)
    stats["USG_WEIGHTED"] = stats["MIN"] * stats["USG"]
    team_daily = (
        stats.groupby(["Date", "TEAM_NAME"], as_index=False)
        .agg(
            MIN=("MIN", "sum"),
            PTS=("PTS", "sum"),
            REB=("REB", "sum"),
            AST=("AST", "sum"),
            USG_WEIGHTED=("USG_WEIGHTED", "sum"),
        )
    )
    team_daily["USG_WEIGHTED"] = team_daily["USG_WEIGHTED"].fillna(0.0)

    merged = team_daily.copy()
    for window_size in windows:
        with_rolls = _rolling_team_features(team_daily, window_size)
        feature_cols = ["Date", "TEAM_NAME"] + [
            f"PLR_{metric}_ROLL{window_size}" for metric in ("MIN", "PTS", "REB", "AST", "USG_WEIGHTED")
        ]
        merged = merged.merge(with_rolls[feature_cols], on=["Date", "TEAM_NAME"], how="left")

    merged = merged.rename(
        columns={
            "MIN": "PLR_MIN_SUM",
            "PTS": "PLR_PTS_SUM",
            "REB": "PLR_REB_SUM",
            "AST": "PLR_AST_SUM",
            "USG_WEIGHTED": "PLR_USG_WEIGHTED_SUM",
        }
    )

    baselines = _build_player_baselines(stats)
    injuries = _build_injury_features(con, baselines, settings)
    if not injuries.empty:
        injuries["Date"] = pd.to_datetime(injuries["Date"], errors="coerce")
    merged = merged.merge(injuries, on=["Date", "TEAM_NAME"], how="left")

    lineups = _build_lineup_features(con, baselines)
    if not lineups.empty:
        lineups["Date"] = pd.to_datetime(lineups["Date"], errors="coerce")
    merged = merged.merge(lineups, on=["Date", "TEAM_NAME"], how="left")

    neutral_team_minutes = as_float(defaults.get("neutral_team_minutes", NEUTRAL_TEAM_MINUTES))
    merged["INJ_OUT_MIN_WT"] = merged["INJ_OUT_MIN_WT"].fillna(0.0)
    merged["INJ_Q_MIN_WT"] = merged["INJ_Q_MIN_WT"].fillna(0.0)
    merged["INJ_DOUBTFUL_MIN_WT"] = merged["INJ_DOUBTFUL_MIN_WT"].fillna(0.0)

    weighted_missing = (
        merged["INJ_OUT_MIN_WT"] + merged["INJ_Q_MIN_WT"] + merged["INJ_DOUBTFUL_MIN_WT"]
    )
    merged["INJ_AVAIL_MIN_WT"] = (1.0 - (weighted_missing / max(neutral_team_minutes, 1.0))).clip(lower=0.0, upper=1.0)

    coverage_by_date = _build_injury_coverage(con)
    merged["INJ_COVERAGE"] = merged["Date"].dt.date.astype(str).map(coverage_by_date).fillna(0.0)
    merged["LINEUP_EXPECTED_STARTERS"] = merged["LINEUP_EXPECTED_STARTERS"].fillna(5.0)
    for column in (
        "LINEUP_CONFIRMED_STARTERS",
        "LINEUP_PROJECTED_STARTERS",
        "LINEUP_COVERAGE",
        "LINEUP_EST_MIN",
        "LINEUP_EST_PTS",
        "LINEUP_EST_REB",
        "LINEUP_EST_AST",
        "LINEUP_EST_USG_WEIGHTED",
    ):
        merged[column] = merged[column].fillna(0.0)
    merged["Date"] = merged["Date"].dt.date.astype(str)

    merged.to_sql(TEAM_FEATURES_TABLE, con, if_exists="replace", index=False)
    print(f"Rebuilt {TEAM_FEATURES_TABLE} with {len(merged.index)} rows.")


def _fetch_dates_for_season(value, today, backfill):
    start_date = datetime.strptime(value["start_date"], "%Y-%m-%d").date()
    end_date = datetime.strptime(value["end_date"], "%Y-%m-%d").date()
    if backfill:
        fetch_end = min(today - timedelta(days=1), end_date)
    else:
        fetch_end = min(today, end_date)
    if start_date > fetch_end:
        return []
    return list(iter_dates(start_date, fetch_end))


def backfill(con, config, season=None, today=None):
    if today is None:
        today = datetime.today().date()

    url = config.get("player_data_url", "")
    if not url:
        print("Missing player_data_url in config.toml.")
        return False

    items = season_items(config, season=season)
    if season and not items:
        print("Season not found in config:", season)
        return False

    existing_dates = get_existing_dates(con)
    fetched = False
    for season_key, value in items:
        all_dates = _fetch_dates_for_season(value, today, backfill=True)
        missing_dates = [day for day in all_dates if day not in existing_dates]
        if not missing_dates:
            print(f"No missing player dates for season {season_key}.")
            continue
        print(f"Backfilling {len(missing_dates)} player dates for season {season_key}.")
        for date_pointer in missing_dates:
            if ingest_date(con, config, url, season_key, value, date_pointer):
                existing_dates.add(date_pointer)
                fetched = True
            time.sleep(random.randint(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))
    return fetched


def fetch_incremental(con, config, today=None):
    if today is None:
        today = datetime.today().date()

    season_key, value, start_date, end_date = select_current_season(config, today)
    if not season_key:
        print("No current player-data season found for today:", today)
        return False

    url = config.get("player_data_url", "")
    if not url:
        print("Missing player_data_url in config.toml.")
        return False

    existing_dates = sorted(get_existing_dates(con))
    season_dates = [day for day in existing_dates if start_date <= day <= min(today, end_date)]
    latest_date = season_dates[-1] if season_dates else None
    fetch_start = start_date if latest_date is None else latest_date + timedelta(days=1)
    fetch_end = min(today, end_date)
    if fetch_start > fetch_end:
        print("No new player dates to fetch. Latest date:", latest_date)
        return False

    fetched = False
    for date_pointer in iter_dates(fetch_start, fetch_end):
        if ingest_date(con, config, url, season_key, value, date_pointer):
            fetched = True
        time.sleep(random.randint(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))
    return fetched


def main(backfill_mode=False, season=None, start_date=None, end_date=None, today=None, db_path=DB_PATH, rebuild_derived=False):
    config = load_config()
    if today is None:
        today = datetime.today().date()

    with sqlite3.connect(db_path) as con:
        fetched = False
        if start_date and end_date:
            parsed_start = datetime.strptime(start_date, "%Y-%m-%d").date()
            parsed_end = datetime.strptime(end_date, "%Y-%m-%d").date()
            url = config.get("player_data_url", "")
            items = season_items(config, season=season)
            if not items:
                print("No seasons available for player data.")
                return
            for date_pointer in iter_dates(parsed_start, parsed_end):
                season_match = None
                for season_key, value in items:
                    season_start = datetime.strptime(value["start_date"], "%Y-%m-%d").date()
                    season_end = datetime.strptime(value["end_date"], "%Y-%m-%d").date()
                    if season_start <= date_pointer <= season_end:
                        season_match = (season_key, value)
                        break
                if not season_match:
                    continue
                season_key, value = season_match
                if ingest_date(con, config, url, season_key, value, date_pointer):
                    fetched = True
                time.sleep(random.randint(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))
        elif backfill_mode:
            fetched = backfill(con, config, season=season, today=today)
        else:
            fetched = fetch_incremental(con, config, today=today)

        if fetched or rebuild_derived:
            rebuild_team_features(con, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NBA player stats and derive injury-aware team features.")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Fetch missing dates for all configured seasons.",
    )
    parser.add_argument(
        "--season",
        help="Limit fetching/backfill to a single season key (e.g. 2025-26).",
    )
    parser.add_argument(
        "--start-date",
        help="Explicit start date (YYYY-MM-DD). Requires --end-date.",
    )
    parser.add_argument(
        "--end-date",
        help="Explicit end date (YYYY-MM-DD). Requires --start-date.",
    )
    parser.add_argument(
        "--rebuild-derived",
        action="store_true",
        help="Rebuild derived team features even if no new rows are fetched.",
    )
    args = parser.parse_args()
    if bool(args.start_date) != bool(args.end_date):
        parser.error("--start-date and --end-date must be provided together.")
    main(
        backfill_mode=args.backfill,
        season=args.season,
        start_date=args.start_date,
        end_date=args.end_date,
        rebuild_derived=args.rebuild_derived,
    )
