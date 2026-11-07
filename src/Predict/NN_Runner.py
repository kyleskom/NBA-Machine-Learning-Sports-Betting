import copy
import re
from pathlib import Path

import numpy as np
from colorama import Fore, Style, init, deinit
from src.Utils.keras_compat import keras, load_model
from src.Utils.features import NN_ML_FEATURES, NN_UO_FEATURES, select_features
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc

init()

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "Models"
LEGACY_MODEL_DIR = MODEL_DIR / "NN_Models"
ML_PATTERN = re.compile(r"Trained-Model-ML-(\d+(?:\.\d+)?)")
OU_PATTERN = re.compile(r"Trained-Model-OU-(\d+(?:\.\d+)?)")

_model = None
_ou_model = None


def _list_model_candidates(prefix):
    candidates = []
    for base in (MODEL_DIR, LEGACY_MODEL_DIR):
        if not base.exists():
            continue
        for path in base.glob(f"{prefix}*"):
            if path.is_dir():
                candidates.append(path)
                continue
            if path.suffix in {".keras", ".h5"}:
                candidates.append(path)
    return candidates


def _select_best_model(prefix, pattern):
    candidates = _list_model_candidates(prefix)
    if not candidates:
        raise FileNotFoundError(f"No model found for prefix {prefix} in {MODEL_DIR}")

    def score(path):
        match = pattern.search(path.name)
        accuracy = float(match.group(1)) if match else 0.0
        if not (0.0 <= accuracy <= 100.0):
            accuracy = 0.0
        return (accuracy, path.stat().st_mtime)

    return max(candidates, key=score)


def _load_legacy_keras(path):
    """Load a .keras file saved by Keras 2.14 on newer TF/Keras.

    Keras 2.14 stored weights under `_layer_checkpoint_dependencies/<layer>/vars/<i>`,
    which newer loaders no longer find. Rebuild the model from config.json and copy
    each layer's weights in by name.
    """
    import io
    import json
    import zipfile

    import h5py
    model_from_json = keras.models.model_from_json

    with zipfile.ZipFile(path) as z:
        config = z.read("config.json").decode()
        weights = z.read("model.weights.h5")
    model = model_from_json(config)
    with h5py.File(io.BytesIO(weights), "r") as h5:
        root = h5["_layer_checkpoint_dependencies"] if "_layer_checkpoint_dependencies" in h5 else h5["layers"]
        for layer in model.layers:
            if not layer.weights:
                continue
            vars_group = root[layer.name]["vars"]
            layer.set_weights([vars_group[str(i)][()] for i in range(len(vars_group))])
    return model


def _load_any(path):
    try:
        return load_model(str(path), compile=False)
    except Exception as e:
        print(f"Standard load failed ({type(e).__name__}); using legacy Keras 2.14 loader for {Path(path).name}")
        return _load_legacy_keras(path)


def _load_models():
    global _model, _ou_model
    if _model is None:
        ml_path = _select_best_model("Trained-Model-ML-", ML_PATTERN)
        _model = _load_any(ml_path)
    if _ou_model is None:
        ou_path = _select_best_model("Trained-Model-OU-", OU_PATTERN)
        _ou_model = _load_any(ou_path)


def nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion):
    _load_models()

    ml_predictions_array = []

    data = select_features(frame_ml, NN_ML_FEATURES).to_numpy(dtype=float)
    data = keras.utils.normalize(data, axis=1)
    for row in data:
        ml_predictions_array.append(_model.predict(np.array([row])))

    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)
    data = select_features(frame_uo, NN_UO_FEATURES).to_numpy(dtype=float)
    data = keras.utils.normalize(data, axis=1)

    ou_predictions_array = []

    for row in data:
        ou_predictions_array.append(_ou_model.predict(np.array([row])))

    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        winner = int(np.argmax(ml_predictions_array[count]))
        under_over = int(np.argmax(ou_predictions_array[count]))
        winner_confidence = ml_predictions_array[count]
        un_confidence = ou_predictions_array[count]
        if under_over == 2:
            confidence = round(float(ou_predictions_array[count][0][2]) * 100, 1)
            winner_team = home_team if winner == 1 else away_team
            win_confidence = round(float(winner_confidence[0][winner]) * 100, 1)
            print(f"{winner_team} ({win_confidence}%) | {home_team} vs {away_team}: "
                  f"PUSH {todays_games_uo[count]} ({confidence}%)")
            count += 1
            continue
        if winner == 1:
            winner_confidence = round(winner_confidence[0][1] * 100, 1)
            if under_over == 0:
                un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
                print(
                    Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                    Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(todays_games_uo[
                                                                        count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
            else:
                un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
                print(
                    Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                    Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(todays_games_uo[
                                                                    count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
        else:
            winner_confidence = round(winner_confidence[0][0] * 100, 1)
            if under_over == 0:
                un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
                print(
                    Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ': ' +
                    Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(todays_games_uo[
                                                                        count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
            else:
                un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
                print(
                    Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ': ' +
                    Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(todays_games_uo[
                                                                    count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
        count += 1
    if kelly_criterion:
        print("------------Expected Value & Kelly Criterion-----------")
    else:
        print("---------------------Expected Value--------------------")
    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        ev_home = ev_away = 0
        if home_team_odds[count] and away_team_odds[count]:
            ev_home = float(
                Expected_Value.expected_value(ml_predictions_array[count][0][1], int(home_team_odds[count])))
            ev_away = float(
                Expected_Value.expected_value(ml_predictions_array[count][0][0], int(away_team_odds[count])))
        expected_value_colors = {'home_color': Fore.GREEN if ev_home > 0 else Fore.RED,
                                 'away_color': Fore.GREEN if ev_away > 0 else Fore.RED}
        bankroll_descriptor = ' Fraction of Bankroll: '
        bankroll_fraction_home = bankroll_descriptor + str(
            kc.calculate_kelly_criterion(home_team_odds[count], ml_predictions_array[count][0][1])) + '%'
        bankroll_fraction_away = bankroll_descriptor + str(
            kc.calculate_kelly_criterion(away_team_odds[count], ml_predictions_array[count][0][0])) + '%'

        print(home_team + ' EV: ' + expected_value_colors['home_color'] + str(ev_home) + Style.RESET_ALL + (
            bankroll_fraction_home if kelly_criterion else ''))
        print(away_team + ' EV: ' + expected_value_colors['away_color'] + str(ev_away) + Style.RESET_ALL + (
            bankroll_fraction_away if kelly_criterion else ''))
        count += 1

    deinit()
    return np.array([p[0] for p in ml_predictions_array])
