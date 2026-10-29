"""Structured prediction board shared by the command line and Flask app."""
from datetime import datetime, timezone

from .XGBoost_Runner import xgb_predictions


def availability_reasons(row, config):
    reasons = []
    if row.get('H_PLR_GAME_COVERAGE', 0) < 1 or row.get('A_PLR_GAME_COVERAGE', 0) < 1:
        reasons.append('player history is missing or stale')
    policy = config.get('bet-eligibility', {})
    if not policy.get('require_complete_availability', True):
        return reasons
    required = ['H_INJ_COVERAGE', 'A_INJ_COVERAGE', 'H_PLR_ROSTER_CONFIRMED', 'A_PLR_ROSTER_CONFIRMED']
    if policy.get('require_confirmed_lineups', True):
        required += ['H_LINEUP_COVERAGE', 'A_LINEUP_COVERAGE']
    labels = {
        'H_INJ_COVERAGE': 'home injury report incomplete', 'A_INJ_COVERAGE': 'away injury report incomplete',
        'H_PLR_ROSTER_CONFIRMED': 'home roster unresolved', 'A_PLR_ROSTER_CONFIRMED': 'away roster unresolved',
        'H_LINEUP_COVERAGE': 'home lineup incomplete', 'A_LINEUP_COVERAGE': 'away lineup incomplete',
    }
    reasons.extend(labels[column] for column in required if row.get(column, 0) < 1)
    return reasons


def xgboost_board(data, totals, frame, games, home_odds, away_odds, config, kelly=False):
    """Return an eligible prediction or an explicit PASS for every supplied matchup."""
    board, eligible = [], []
    for index, game in enumerate(games[:len(frame.index)]):
        reasons = availability_reasons(frame.iloc[index], config)
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


def response(board, sportsbook, report_refresh=None):
    return {'generated_at': datetime.now(timezone.utc).isoformat(), 'sportsbook': sportsbook,
            'report_refresh': report_refresh or {}, 'games': board}


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
