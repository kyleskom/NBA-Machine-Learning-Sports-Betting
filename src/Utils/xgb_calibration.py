"""Probability calibration for raw XGBoost Boosters.

Lives in an importable module (not the training script's __main__) so the pickled
calibrator can be loaded later by src/Predict/XGBoost_Runner.py. Works with
scikit-learn 1.3 (cv="prefit") and >= 1.6 (FrozenEstimator; prefit removed in 1.8).
"""
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV


class BoosterWrapper(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(self, booster=None, num_class=2):
        self.booster = booster
        self.num_class = num_class

    @property
    def classes_(self):
        return np.arange(self.num_class)

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        return self.booster.predict(xgb.DMatrix(np.asarray(X, dtype=float)))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def fit_calibrator(booster, num_class, X, y, method):
    wrapper = BoosterWrapper(booster, num_class)
    try:
        from sklearn.frozen import FrozenEstimator  # scikit-learn >= 1.6
        calibrator = CalibratedClassifierCV(FrozenEstimator(wrapper), method=method)
    except ImportError:
        calibrator = CalibratedClassifierCV(wrapper, method=method, cv="prefit")
    return calibrator.fit(X, y)
