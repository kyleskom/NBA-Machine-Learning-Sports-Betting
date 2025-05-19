import numpy as np
import joblib
import glob
import os
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc

init()
_lr_model = None


def _load_model():
    global _lr_model
    if _lr_model is None:
        # automatically find the most recent saved Logistic Regression model
        model_files = glob.glob("C:/Users/antho/cursorProjects/NBA-Machine-Learning-Sports-Betting/Models/Logistic_Regression_ML_*")
        if not model_files:
            raise FileNotFoundError("No Logistic Regression model files found in Models directory")
        latest = max(model_files, key=os.path.getctime)
        print(latest)
        _lr_model = joblib.load(latest)


def lr_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion):
    _load_model()
    # prepare feature matrix, dropping columns excluded during training
    drop_cols = ['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU']
    X = frame_ml.drop(columns=drop_cols, errors='ignore').values.astype(float)
    probabilities = _lr_model.predict_proba(X)

    # Game winner predictions
    count = 0
    for home_team, away_team in games:
        home_prob = probabilities[count][1] * 100
        away_prob = probabilities[count][0] * 100
        if home_prob > away_prob:
            print(
                Fore.GREEN + home_team + Style.RESET_ALL +
                Fore.CYAN + f" ({home_prob:.1f}%)" + Style.RESET_ALL +
                " vs " +
                Fore.RED + away_team + Style.RESET_ALL
            )
        else:
            print(
                Fore.RED + home_team + Style.RESET_ALL +
                " vs " +
                Fore.GREEN + away_team + Style.RESET_ALL +
                Fore.CYAN + f" ({away_prob:.1f}%)" + Style.RESET_ALL
            )
        count += 1

    # Expected Value & Kelly Criterion
    if kelly_criterion:
        print("------------Expected Value & Kelly Criterion-----------")
    else:
        print("---------------------Expected Value--------------------")

    count = 0
    for home_team, away_team in games:
        ev_home = ev_away = 0
        if home_team_odds[count] and away_team_odds[count]:
            ev_home = float(Expected_Value.expected_value(probabilities[count][1], int(home_team_odds[count])))
            ev_away = float(Expected_Value.expected_value(probabilities[count][0], int(away_team_odds[count])))
        colors = {
            'home_color': Fore.GREEN if ev_home > 0 else Fore.RED,
            'away_color': Fore.GREEN if ev_away > 0 else Fore.RED
        }
        bankroll_descriptor = ' Fraction of Bankroll: '
        bf_home = bankroll_descriptor + str(kc.calculate_kelly_criterion(home_team_odds[count], probabilities[count][1])) + '%'
        bf_away = bankroll_descriptor + str(kc.calculate_kelly_criterion(away_team_odds[count], probabilities[count][0])) + '%'

        print(home_team + ' EV: ' + colors['home_color'] + str(ev_home) + Style.RESET_ALL + (bf_home if kelly_criterion else ''))
        print(away_team + ' EV: ' + colors['away_color'] + str(ev_away) + Style.RESET_ALL + (bf_away if kelly_criterion else ''))
        count += 1

    deinit() 