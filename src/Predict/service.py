"""Structured prediction board shared by the command line and Flask app."""
from datetime import datetime, timezone
import math

from .XGBoost_Runner import xgb_predictions


def market_reasons(total, home_odds, away_odds):
    """Keep incomplete prices off the prediction board without player-data gates."""
    reasons = []
    for label, value in [('total line', total), ('home moneyline', home_odds), ('away moneyline', away_odds)]:
        try:
            numeric = float(value)
            valid = math.isfinite(numeric) and (numeric > 0 if label == 'total line' else abs(numeric) >= 100)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            reasons.append(f'{label} is missing or invalid')
    return reasons


def xgboost_board(data, totals, frame, games, home_odds, away_odds, config, kelly=False):
    """Return an eligible prediction or an explicit PASS for every supplied matchup."""
    board, eligible = [], []
    for index, game in enumerate(games[:len(frame.index)]):
        reasons = market_reasons(totals[index], home_odds[index], away_odds[index])
        item = {'home_team': game[0], 'away_team': game[1], 'status': 'PASS' if reasons else 'READY',
                'reasons': reasons}
        board.append(item)
        if not reasons:
            eligible.append(index)
    if not eligible:
        return board
    predictions = xgb_predictions(data[eligible], [totals[i] for i in eligible], frame.iloc[eligible].reset_index(drop=True),
                                  [games[i] for i in eligible], [home_odds[i] for i in eligible],
                                  [away_odds[i] for i in eligible], kelly)
    for index, prediction in zip(eligible, predictions):
        board[index].update(prediction)
    return board


def response(board, sportsbook):
    return {'generated_at': datetime.now(timezone.utc).isoformat(), 'sportsbook': sportsbook,
            'games': board}


def template_games(result):
    """Temporary view adapter while the existing Jinja template is migrated."""
    games = {}
    for item in result['games']:
        key = f"{item['away_team']}:{item['home_team']}"
        if item['status'] == 'PASS':
            games[key] = {'away_team': item['away_team'], 'home_team': item['home_team'],
                          'status': 'PASS', 'reasons': item['reasons']}
            continue
        moneyline, total = item['moneyline'], item['total']
        games[key] = {
            'away_team': item['away_team'], 'home_team': item['home_team'], 'status': 'READY',
            'away_confidence': round(moneyline['away_probability'] * 100, 1),
            'home_confidence': round(moneyline['home_probability'] * 100, 1),
            'away_team_odds': item['odds']['away'], 'home_team_odds': item['odds']['home'],
            'away_team_ev': item['expected_value']['away'], 'home_team_ev': item['expected_value']['home'],
            'ou_pick': total['pick'], 'ou_value': total['line'],
            'ou_confidence': round(total['probability'] * 100, 1),
        }
    return games
