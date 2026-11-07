import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.Predict import service


class TestPredictionService(unittest.TestCase):
    def test_incomplete_availability_returns_pass_without_model_call(self):
        frame = pd.DataFrame([{'H_PLR_GAME_COVERAGE': 1, 'A_PLR_GAME_COVERAGE': 1,
                               'H_INJ_COVERAGE': 0, 'A_INJ_COVERAGE': 1,
                               'H_PLR_ROSTER_CONFIRMED': 1, 'A_PLR_ROSTER_CONFIRMED': 1,
                               'H_LINEUP_COVERAGE': 1, 'A_LINEUP_COVERAGE': 1}])
        with patch('src.Predict.service.xgb_predictions') as predict:
            board = service.xgboost_board(np.zeros((1, 2)), [220.5], frame,
                                          [['Atlanta Hawks', 'Boston Celtics']], [-110], [-110], {})
        self.assertEqual(board[0]['status'], 'PASS')
        self.assertIn('home injury report incomplete', board[0]['reasons'])
        predict.assert_not_called()

    def test_ready_game_merges_structured_prediction(self):
        frame = pd.DataFrame([{'H_PLR_GAME_COVERAGE': 1, 'A_PLR_GAME_COVERAGE': 1,
                               'H_INJ_COVERAGE': 1, 'A_INJ_COVERAGE': 1,
                               'H_PLR_ROSTER_CONFIRMED': 1, 'A_PLR_ROSTER_CONFIRMED': 1,
                               'H_LINEUP_COVERAGE': 1, 'A_LINEUP_COVERAGE': 1,
                               'H_LINEUP_CONFIRMED_STARTERS': 5, 'A_LINEUP_CONFIRMED_STARTERS': 5}])
        prediction = {'home_team': 'Atlanta Hawks', 'away_team': 'Boston Celtics',
                      'moneyline': {'home_probability': .6, 'away_probability': .4, 'pick': 'Atlanta Hawks'},
                      'total': {'line': 220.5, 'pick': 'OVER', 'probability': .55},
                      'odds': {'home': -120, 'away': 100},
                      'expected_value': {'home': 2.0, 'away': -3.0},
                      'kelly': {'home': None, 'away': None}}
        with patch('src.Predict.service.xgb_predictions', return_value=[prediction]):
            board = service.xgboost_board(np.zeros((1, 2)), [220.5], frame,
                                          [['Atlanta Hawks', 'Boston Celtics']], [-120], [100], {})
        self.assertEqual(board[0]['status'], 'READY')
        self.assertEqual(board[0]['moneyline']['pick'], 'Atlanta Hawks')
        legacy = service.template_games(service.response(board, 'fanduel'))
        self.assertEqual(legacy['Boston Celtics:Atlanta Hawks']['home_confidence'], 60.0)

    def test_projected_starters_do_not_satisfy_confirmed_lineup_policy(self):
        row = {column: 1 for column in ['H_PLR_GAME_COVERAGE', 'A_PLR_GAME_COVERAGE',
               'H_INJ_COVERAGE', 'A_INJ_COVERAGE', 'H_PLR_ROSTER_CONFIRMED',
               'A_PLR_ROSTER_CONFIRMED', 'H_LINEUP_COVERAGE', 'A_LINEUP_COVERAGE']}
        row.update(H_LINEUP_CONFIRMED_STARTERS=0, A_LINEUP_CONFIRMED_STARTERS=5,
                   H_LINEUP_PROJECTED_STARTERS=5)
        with patch('src.Predict.service.xgb_predictions') as predict:
            board = service.xgboost_board(np.zeros((1, 2)), [220.5], pd.DataFrame([row]),
                                          [['Atlanta Hawks', 'Boston Celtics']], [-110], [-110], {})
        self.assertEqual(board[0]['status'], 'PASS')
        self.assertEqual(board[0]['reasons'], ['home lineup unconfirmed (0/5 confirmed, 5/5 projected)'])
        predict.assert_not_called()
