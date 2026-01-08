import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from colorama import Fore, Style, init, deinit
from keras.models import load_model
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc

init()

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "Models" / "NN_Models"
TIMESTAMP_PATTERN = re.compile(r"Trained-Model-(?:ML|OU)-(\d+(?:\.\d+)?)")

_model = None
_ou_model = None


def _select_latest_model(prefix):
    candidates = list(MODEL_DIR.glob(f"{prefix}*"))
    if not candidates:
        raise FileNotFoundError(f"No model found for prefix {prefix} in {MODEL_DIR}")

    def score(path):
        match = TIMESTAMP_PATTERN.search(path.name)
        timestamp = float(match.group(1)) if match else 0.0
        return (timestamp, path.stat().st_mtime)

    return max(candidates, key=score)


def _load_models():
    global _model, _ou_model
    if _model is None:
        ml_path = _select_latest_model("Trained-Model-ML-")
        _model = load_model(str(ml_path))
    if _ou_model is None:
        ou_path = _select_latest_model("Trained-Model-OU-")
        _ou_model = load_model(str(ou_path))


def _format_game_line(home_team, away_team, winner_is_home, winner_confidence, under_over, ou_value, ou_confidence):
    winner_team = home_team if winner_is_home else away_team
    loser_team = away_team if winner_is_home else home_team
    winner_color = Fore.GREEN if winner_is_home else Fore.RED
    loser_color = Fore.RED if winner_is_home else Fore.GREEN
    ou_label = "UNDER" if under_over == 0 else "OVER"
    ou_color = Fore.MAGENTA if under_over == 0 else Fore.BLUE
    return (
        f"{winner_color}{winner_team}{Style.RESET_ALL}"
        f"{Fore.CYAN} ({winner_confidence}%)"
        f"{Style.RESET_ALL} vs {loser_color}{loser_team}{Style.RESET_ALL}: "
        f"{ou_color}{ou_label} {Style.RESET_ALL}{ou_value}"
        f"{Style.RESET_ALL}{Fore.CYAN} ({ou_confidence}%)"
        f"{Style.RESET_ALL}"
    )


def _print_expected_value(
    games,
    ml_predictions_array,
    home_team_odds,
    away_team_odds,
    kelly_criterion,
):
    if kelly_criterion:
        print("------------Expected Value & Kelly Criterion-----------")
    else:
        print("---------------------Expected Value--------------------")
    for idx, game in enumerate(games):
        home_team, away_team = game
        ev_home = ev_away = 0
        if home_team_odds[idx] and away_team_odds[idx]:
            ev_home = float(
                Expected_Value.expected_value(
                    ml_predictions_array[idx][1],
                    int(home_team_odds[idx]),
                )
            )
            ev_away = float(
                Expected_Value.expected_value(
                    ml_predictions_array[idx][0],
                    int(away_team_odds[idx]),
                )
            )
        expected_value_colors = {
            "home_color": Fore.GREEN if ev_home > 0 else Fore.RED,
            "away_color": Fore.GREEN if ev_away > 0 else Fore.RED,
        }
        bankroll_descriptor = " Fraction of Bankroll: "
        bankroll_fraction_home = bankroll_descriptor + str(
            kc.calculate_kelly_criterion(home_team_odds[idx], ml_predictions_array[idx][1])
        ) + "%"
        bankroll_fraction_away = bankroll_descriptor + str(
            kc.calculate_kelly_criterion(away_team_odds[idx], ml_predictions_array[idx][0])
        ) + "%"

        print(
            home_team
            + " EV: "
            + expected_value_colors["home_color"]
            + str(ev_home)
            + Style.RESET_ALL
            + (bankroll_fraction_home if kelly_criterion else "")
        )
        print(
            away_team
            + " EV: "
            + expected_value_colors["away_color"]
            + str(ev_away)
            + Style.RESET_ALL
            + (bankroll_fraction_away if kelly_criterion else "")
        )


def nn_runner(normalized_data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion):
    _load_models()

    if normalized_data is None:
        raise ValueError("normalized_data is required for neural network predictions.")

    frame_uo = frame_ml.copy()
    frame_uo["OU"] = np.asarray(todays_games_uo, dtype=float)
    data_uo = frame_uo.values.astype(float)
    normalized_uo = tf.keras.utils.normalize(data_uo, axis=1)

    try:
        ml_predictions_array = _model.predict(normalized_data, verbose=0)
        ou_predictions_array = _ou_model.predict(normalized_uo, verbose=0)

        for idx, game in enumerate(games):
            home_team, away_team = game
            winner = int(np.argmax(ml_predictions_array[idx]))
            under_over = int(np.argmax(ou_predictions_array[idx]))
            winner_confidence = round(ml_predictions_array[idx][winner] * 100, 1)
            ou_confidence = round(ou_predictions_array[idx][under_over] * 100, 1)

            print(
                _format_game_line(
                    home_team,
                    away_team,
                    winner_is_home=(winner == 1),
                    winner_confidence=winner_confidence,
                    under_over=under_over,
                    ou_value=todays_games_uo[idx],
                    ou_confidence=ou_confidence,
                )
            )

        _print_expected_value(
            games,
            ml_predictions_array,
            home_team_odds,
            away_team_odds,
            kelly_criterion,
        )
    finally:
        deinit()
