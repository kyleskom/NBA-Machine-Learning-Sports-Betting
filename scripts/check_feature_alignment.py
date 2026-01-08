import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from keras.models import load_model
except ImportError:
    load_model = None

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"
MODEL_DIR = BASE_DIR / "Models"

DROP_COLUMNS_ML = [
    "Score",
    "Home-Team-Win",
    "TEAM_NAME",
    "TEAM_ID",
    "Date",
    "TEAM_NAME.1",
    "TEAM_ID.1",
    "Date.1",
    "OU",
    "OU-Cover",
]
DROP_COLUMNS_UO = [
    "Score",
    "Home-Team-Win",
    "TEAM_NAME",
    "TEAM_ID",
    "Date",
    "TEAM_NAME.1",
    "TEAM_ID.1",
    "Date.1",
    "OU-Cover",
]


def get_table_columns(dataset_name):
    with sqlite3.connect(DATASET_DB) as con:
        df = pd.read_sql_query(f'SELECT * FROM "{dataset_name}" LIMIT 1', con)
    return list(df.columns)


def get_training_columns(dataset_name, drop_columns):
    cols = get_table_columns(dataset_name)
    return [col for col in cols if col not in drop_columns]


def load_frame_columns(path):
    df = pd.read_csv(path)
    return list(df.columns)


def load_live_frames():
    sys.path.insert(0, str(BASE_DIR))
    from main import create_todays_games_data, load_schedule, DATA_URL, TODAYS_GAMES_URL
    from src.Utils.tools import get_json_data, to_data_frame, get_todays_games_json, create_todays_games

    stats_json = get_json_data(DATA_URL)
    df = to_data_frame(stats_json)
    schedule_df = load_schedule()
    today = datetime.today()
    games = create_todays_games(get_todays_games_json(TODAYS_GAMES_URL))
    data, todays_games_uo, frame_ml, _, _ = create_todays_games_data(
        games, df, None, schedule_df, today
    )
    if frame_ml is None or frame_ml.empty:
        raise RuntimeError("No games found to build inference frames.")
    frame_uo = frame_ml.copy()
    frame_uo["OU"] = np.asarray(todays_games_uo, dtype=float)
    return frame_ml, frame_uo


def find_latest_model(pattern):
    candidates = list(MODEL_DIR.rglob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def describe_model_input(model_path, label):
    if model_path is None:
        print(f"{label} model: not found")
        return None
    if load_model is None:
        print(f"{label} model: keras not installed")
        return None
    model = load_model(str(model_path), compile=False)
    input_features = model.input_shape[-1]
    print(f"{label} model input features: {input_features} ({model_path})")
    return input_features


def describe_xgb_input(model_path, label):
    if model_path is None:
        print(f"{label} model: not found")
        return None
    if xgb is None:
        print(f"{label} model: xgboost not installed")
        return None
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    feature_count = booster.num_features()
    print(f"{label} model input features: {feature_count} ({model_path})")
    return feature_count


def compare_columns(label, train_cols, infer_cols):
    train_set = set(train_cols)
    infer_set = set(infer_cols)
    missing = sorted(train_set - infer_set)
    extra = sorted(infer_set - train_set)
    print(f"{label} columns: train={len(train_cols)} infer={len(infer_cols)}")
    print(f"{label} missing: {len(missing)}; extra: {len(extra)}")
    if missing:
        print(f"{label} missing sample: {', '.join(missing[:20])}")
    if extra:
        print(f"{label} extra sample: {', '.join(extra[:20])}")


def main():
    parser = argparse.ArgumentParser(description="Compare training vs inference feature sizes.")
    parser.add_argument("--dataset", default="dataset_2012-26_new", help="Dataset table name.")
    parser.add_argument("--frame-ml", help="CSV path to inference frame_ml.")
    parser.add_argument("--frame-uo", help="CSV path to inference frame_uo.")
    parser.add_argument("--use-live", action="store_true", help="Build inference frames from live data.")
    parser.add_argument("--nn-ml-model", help="Path to NN ML model.")
    parser.add_argument("--nn-uo-model", help="Path to NN OU model.")
    parser.add_argument("--xgb-ml-model", help="Path to XGBoost ML model.")
    parser.add_argument("--xgb-uo-model", help="Path to XGBoost OU model.")
    args = parser.parse_args()

    train_cols_ml = get_training_columns(args.dataset, DROP_COLUMNS_ML)
    train_cols_uo = get_training_columns(args.dataset, DROP_COLUMNS_UO)
    print(f"Training columns ML: {len(train_cols_ml)}")
    print(f"Training columns OU: {len(train_cols_uo)}")

    frame_ml = frame_uo = None
    if args.use_live:
        frame_ml, frame_uo = load_live_frames()
    if args.frame_ml:
        frame_ml = pd.read_csv(args.frame_ml)
    if args.frame_uo:
        frame_uo = pd.read_csv(args.frame_uo)

    if frame_ml is not None:
        compare_columns("ML", train_cols_ml, list(frame_ml.columns))
    if frame_uo is not None:
        compare_columns("OU", train_cols_uo, list(frame_uo.columns))

    nn_ml_model = Path(args.nn_ml_model) if args.nn_ml_model else find_latest_model("Trained-Model-ML-*")
    nn_ou_model = Path(args.nn_uo_model) if args.nn_uo_model else find_latest_model("Trained-Model-OU-*")
    describe_model_input(nn_ml_model, "NN ML")
    describe_model_input(nn_ou_model, "NN OU")

    xgb_ml_model = Path(args.xgb_ml_model) if args.xgb_ml_model else find_latest_model("XGBoost_*_ML*.json")
    xgb_ou_model = Path(args.xgb_uo_model) if args.xgb_uo_model else find_latest_model("XGBoost_*_UO*.json")
    describe_xgb_input(xgb_ml_model, "XGB ML")
    describe_xgb_input(xgb_ou_model, "XGB OU")


if __name__ == "__main__":
    main()
