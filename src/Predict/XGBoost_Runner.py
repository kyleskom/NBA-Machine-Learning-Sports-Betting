from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc
from src.Utils import PlayerContext
from src.Predict.model_selection import selected_model_path


init()

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "Models" / "XGBoost_Models"

xgb_ml = None
xgb_uo = None
xgb_ml_calibrator = None
xgb_uo_calibrator = None
xgb_ml_features = None
xgb_uo_features = None
xgb_ml_path = None
xgb_uo_path = None


def _select_model_path(kind):
    path = selected_model_path(kind, MODEL_DIR)
    if path is None:
        raise FileNotFoundError(f"No XGBoost {kind} model found in {MODEL_DIR}")
    return path


def _load_calibrator(model_path):
    calibration_path = model_path.with_name(f"{model_path.stem}_calibration.pkl")
    if not calibration_path.exists():
        return None
    try:
        return joblib.load(calibration_path)
    except Exception as exc:
        raise ValueError(f"Calibration failed to load for {model_path.name}; retrain") from exc


def _load_models():
    global xgb_ml, xgb_uo, xgb_ml_calibrator, xgb_uo_calibrator, xgb_ml_features, xgb_uo_features, xgb_ml_path, xgb_uo_path
    if xgb_ml is None:
        ml_path = _select_model_path("ML")
        xgb_ml_path = ml_path
        xgb_ml = xgb.Booster()
        xgb_ml.load_model(str(ml_path))
        xgb_ml_calibrator = _load_calibrator(ml_path)
        xgb_ml_features = PlayerContext.load_feature_columns(ml_path)
    if xgb_uo is None:
        uo_path = _select_model_path("UO")
        xgb_uo_path = uo_path
        xgb_uo = xgb.Booster()
        xgb_uo.load_model(str(uo_path))
        xgb_uo_calibrator = _load_calibrator(uo_path)
        xgb_uo_features = PlayerContext.load_feature_columns(uo_path)


def _predict_probs(model, data, calibrator=None, num_classes=None):
    if calibrator is not None:
        return calibrator.predict_proba(data)
    probabilities = model.predict(xgb.DMatrix(data))
    if num_classes == 2 and probabilities.ndim == 1:
        return np.column_stack((1 - probabilities, probabilities))
    return probabilities


def _prepare_model_data(model, frame, feature_columns, label="model"):
    if feature_columns:
        return PlayerContext.align_frame_to_columns(frame, feature_columns).values

    raise ValueError(f"{label}: feature manifest required. Retrain with player-v2 data.")


def _american_odds(value):
    try:
        odds = int(value)
    except (TypeError, ValueError):
        return None
    return odds if odds else None


def xgb_predictions(data, totals, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion=False):
    """Return prediction records. Presentation belongs to the CLI or web layer."""
    _load_models()
    frame_uo = frame_ml.copy()
    frame_uo['OU'] = np.asarray(totals, dtype=float)
    ml_data = _prepare_model_data(xgb_ml, frame_ml, xgb_ml_features, label='XGBoost ML')
    uo_data = _prepare_model_data(xgb_uo, frame_uo, xgb_uo_features, label='XGBoost OU')
    ml_probs = _predict_probs(xgb_ml, ml_data, xgb_ml_calibrator, num_classes=2)
    uo_probs = _predict_probs(xgb_uo, uo_data, xgb_uo_calibrator, num_classes=3)
    records = []
    for index, (home_team, away_team) in enumerate(games):
        home_probability = float(ml_probs[index][1])
        away_probability = float(ml_probs[index][0])
        total_index = int(np.argmax(uo_probs[index]))
        total_pick = {0: 'UNDER', 1: 'OVER', 2: 'PUSH'}.get(total_index, 'PUSH')
        home_odds = _american_odds(home_team_odds[index])
        away_odds = _american_odds(away_team_odds[index])
        home_ev = away_ev = None
        home_kelly = away_kelly = None
        if home_odds is not None and away_odds is not None:
            home_ev = float(Expected_Value.expected_value(home_probability, home_odds))
            away_ev = float(Expected_Value.expected_value(away_probability, away_odds))
            if kelly_criterion:
                home_kelly = float(kc.calculate_kelly_criterion(home_odds, home_probability))
                away_kelly = float(kc.calculate_kelly_criterion(away_odds, away_probability))
        records.append({
            'home_team': home_team, 'away_team': away_team, 'model': 'xgboost',
            'model_paths': {'moneyline': str(xgb_ml_path), 'total': str(xgb_uo_path)},
            'moneyline': {'home_probability': home_probability, 'away_probability': away_probability,
                          'pick': home_team if home_probability >= away_probability else away_team},
            'total': {'line': float(totals[index]), 'pick': total_pick,
                      'probability': float(uo_probs[index][total_index])},
            'odds': {'home': home_odds, 'away': away_odds},
            'expected_value': {'home': home_ev, 'away': away_ev},
            'kelly': {'home': home_kelly, 'away': away_kelly},
        })
    return records


def _format_game_line(home_team, away_team, winner_is_home, winner_confidence, under_over, ou_value, ou_confidence):
    winner_team = home_team if winner_is_home else away_team
    loser_team = away_team if winner_is_home else home_team
    winner_color = Fore.GREEN if winner_is_home else Fore.RED
    loser_color = Fore.RED if winner_is_home else Fore.GREEN
    ou_label = {0: "UNDER", 1: "OVER", 2: "PUSH"}[under_over]
    ou_color = Fore.MAGENTA if under_over == 0 else Fore.BLUE
    return (
        f"{winner_color}{winner_team}{Style.RESET_ALL}"
        f"{Fore.CYAN} ({winner_confidence}%)"
        f"{Style.RESET_ALL} vs {loser_color}{loser_team}{Style.RESET_ALL}: "
        f"{ou_color}{ou_label} {Style.RESET_ALL}{ou_value}"
        f"{Style.RESET_ALL}{Fore.CYAN} ({ou_confidence}%)"
        f"{Style.RESET_ALL}"
    )


def xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion):
    try:
        records = xgb_predictions(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion)
        for record in records:
            moneyline, total = record['moneyline'], record['total']
            winner_is_home = moneyline['pick'] == record['home_team']
            confidence = moneyline['home_probability'] if winner_is_home else moneyline['away_probability']
            print(_format_game_line(record['home_team'], record['away_team'], winner_is_home,
                                    round(confidence * 100, 1), {'UNDER': 0, 'OVER': 1, 'PUSH': 2}[total['pick']],
                                    total['line'], round(total['probability'] * 100, 1)))
        print('------------Expected Value & Kelly Criterion-----------' if kelly_criterion else '---------------------Expected Value--------------------')
        for record in records:
            print(f"{record['home_team']} EV: {record['expected_value']['home']}")
            print(f"{record['away_team']} EV: {record['expected_value']['away']}")
    finally:
        deinit()
