import unittest

import numpy as np
import xgboost as xgb

from src.Utils.Calibration import BoosterWrapper
from src.Predict.XGBoost_Runner import _predict_probs


class TestCalibrationWrapper(unittest.TestCase):
    def test_binary_booster_is_exposed_as_two_probability_columns(self):
        features = np.array([[0.0], [1.0], [0.0], [1.0]])
        labels = np.array([0, 1, 0, 1])
        booster = xgb.train(
            {"objective": "binary:logistic", "eval_metric": "logloss", "seed": 1},
            xgb.DMatrix(features, label=labels), num_boost_round=3,
        )
        probabilities = BoosterWrapper(booster, 2).predict_proba(features)
        self.assertEqual(probabilities.shape, (4, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(4))

    def test_runner_normalizes_an_uncalibrated_binary_prediction(self):
        class BinaryModel:
            def predict(self, _):
                return np.array([0.2, 0.8])

        probabilities = _predict_probs(BinaryModel(), np.array([[0.0], [1.0]]), num_classes=2)
        np.testing.assert_allclose(probabilities, [[0.8, 0.2], [0.2, 0.8]])


if __name__ == "__main__":
    unittest.main()
