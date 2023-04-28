import copy

import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc


# from src.Utils.Dictionaries import team_index_current
# from src.Utils.tools import get_json_data, to_data_frame, get_todays_games_json, create_todays_games
init()
xgb_ml = xgb.Booster()
xgb_ml.load_model('Models/XGBoost_Models/XGBoost_68.6%_ML-2.json')
xgb_uo = xgb.Booster()
xgb_uo.load_model('Models/XGBoost_Models/XGBoost_54.8%_UO-8.json')


def xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds):
    ml_predictions_array = []

    for row in data:
        ml_predictions_array.append(xgb_ml.predict(xgb.DMatrix(np.array([row]))))

    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)
    data = frame_uo.values
    data = data.astype(float)

    ou_predictions_array = []

    for row in data:
        ou_predictions_array.append(xgb_uo.predict(xgb.DMatrix(np.array([row]))))

    count = 0
    '''
    Mapping winner, over/under combinations to colors
    - (1, 0) is the equivalent of home winner, under
    '''
    color_dict = {
        (1, 0): (Fore.GREEN, Fore.RED, Fore.CYAN, Fore.MAGENTA, 'UNDER'),
        (1, 1): (Fore.GREEN, Fore.RED, Fore.CYAN, Fore.BLUE, 'OVER'),
        (0, 0): (Fore.RED, Fore.GREEN, Fore.CYAN, Fore.MAGENTA, 'UNDER'),
        (0, 1): (Fore.RED, Fore.GREEN, Fore.CYAN, Fore.BLUE, 'OVER')
    }

    for game in games:
        home_team = game[0]
        away_team = game[1]
        winner = int(np.argmax(ml_predictions_array[count]))
        under_over = int(np.argmax(ou_predictions_array[count]))
        winner_confidence = round(ml_predictions_array[count][0][winner] * 100, 1)
        un_confidence = round(ou_predictions_array[count][0][under_over] * 100, 1)
        color_tuple = color_dict[(winner, under_over)]
        home_color, away_color, conf_color, uo_color, uo_prefix = color_tuple
        print(home_color + home_team + Style.RESET_ALL + ' vs ' + away_color + away_team + Style.RESET_ALL + ' ' + conf_color + f"({winner_confidence}%)" + Style.RESET_ALL + ': ' + uo_color + uo_prefix + Style.RESET_ALL + ' ' + str(todays_games_uo[count]) + Style.RESET_ALL + ' ' + conf_color + f"({un_confidence}%)" + Style.RESET_ALL)
        count += 1

    print("------------Expected Value & Kelly Criterion------------")
    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        ev_home = ev_away = 0
        if home_team_odds[count] and away_team_odds[count]:
            ev_home = float(Expected_Value.expected_value(ml_predictions_array[count][0][1], int(home_team_odds[count])))
            ev_away = float(Expected_Value.expected_value(ml_predictions_array[count][0][0], int(away_team_odds[count])))
        expected_value_colors = {'home_color': Fore.GREEN if ev_home > 0 else Fore.RED,
                        'away_color': Fore.GREEN if ev_away > 0 else Fore.RED}
        
        winner = int(np.argmax(ml_predictions_array[count]))
        print(winner)
        winner_confidence = ml_predictions_array[count][0][winner]
        bankroll_descriptor = ' Fraction of Bankroll: '
        if (winner == 0):
            bankroll_fraction_home = bankroll_descriptor + str(kc.calculate_kelly_criterion(home_team_odds[count], winner_confidence)) + '%'
            bankroll_fraction_away = ''
        else:
            bankroll_fraction_home = ''
            bankroll_fraction_away = bankroll_descriptor + str(kc.calculate_kelly_criterion(away_team_odds[count], winner_confidence)) + '%'

        print(home_team + ' EV: ' + expected_value_colors['home_color'] + str(ev_home) + Style.RESET_ALL + bankroll_fraction_home)
        print(away_team + ' EV: ' + expected_value_colors['away_color'] + str(ev_away) + Style.RESET_ALL + bankroll_fraction_away)
        count += 1

    deinit()
