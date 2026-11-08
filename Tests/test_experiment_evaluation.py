import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.Evaluation.experiment import (
    chronological_cv_indices,
    chronological_holdout_indices,
    dataset_fingerprint,
    expected_calibration_error,
    no_vig_home_probability,
    moneyline_market_and_backtest,
    select_moneyline_policy,
)


class TestExperimentEvaluation(unittest.TestCase):
    def setUp(self):
        self.dates = pd.Series([
            "2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02",
            "2024-01-03", "2024-01-04", "2024-01-04",
        ])

    def test_final_split_keeps_an_entire_slate_together(self):
        train, final, metadata = chronological_holdout_indices(
            self.dates, final_start="2024-01-03"
        )
        self.assertEqual(train.tolist(), [0, 1, 2, 3])
        self.assertEqual(final.tolist(), [4, 5, 6])
        self.assertEqual(metadata["final_start"], "2024-01-03")

    def test_cv_folds_do_not_divide_games_on_a_date(self):
        for train, validation in chronological_cv_indices(self.dates, 2):
            train_dates = set(self.dates.iloc[train])
            validation_dates = set(self.dates.iloc[validation])
            self.assertFalse(train_dates & validation_dates)

    def test_fingerprint_changes_when_a_feature_value_changes(self):
        frame = pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Home-Team-Win": [1, 0], "rating": [10.0, 11.0],
        })
        initial = dataset_fingerprint(frame, "unit", "Home-Team-Win", ["rating"])
        frame.loc[1, "rating"] = 12.0
        changed = dataset_fingerprint(frame, "unit", "Home-Team-Win", ["rating"])
        self.assertNotEqual(initial["sha256"], changed["sha256"])

    def test_calibration_and_no_vig_calculations(self):
        # Predictions are perfectly calibrated in these two occupied bins.
        self.assertAlmostEqual(expected_calibration_error([0, 1], [0.0, 1.0]), 0.0)
        self.assertAlmostEqual(no_vig_home_probability(-110, -110), 0.5)
        self.assertTrue(np.isfinite(no_vig_home_probability(150, -170)))

    @patch("src.Evaluation.experiment.attach_moneyline_market")
    def test_policy_selection_applies_probability_and_ev_thresholds(self, attach_market):
        merged = pd.DataFrame({"ML_Home": [-110, -110, -110, -110], "ML_Away": [-110, -110, -110, -110]})
        attach_market.return_value = (merged, {"available": True, "sportsbook": "fanduel"})
        outcomes = np.array([1, 1, 0, 0])
        probabilities = np.array([0.65, 0.63, 0.58, 0.55])
        result = moneyline_market_and_backtest(pd.DataFrame(index=range(4)), outcomes, probabilities,
                                               min_ev=0.02, min_probability=0.60)
        self.assertEqual(result["backtest"]["stake_units"], 2.0)
        policy = select_moneyline_policy(pd.DataFrame(index=range(4)), outcomes, probabilities,
                                         ev_thresholds=(0.02,), probability_thresholds=(0.50, 0.60), min_bets=2)
        self.assertTrue(policy["available"])
        self.assertEqual(policy["selected"]["min_probability"], 0.60)


if __name__ == "__main__":
    unittest.main()
