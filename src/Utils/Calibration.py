"""Importable wrapper for portable XGBoost calibration artifacts."""
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin


class BoosterWrapper(ClassifierMixin, BaseEstimator):
    def __init__(self, booster, num_class):
        self.booster=booster
        self.num_class=num_class
        self.classes_=np.arange(num_class)

    def fit(self,X,y):
        return self

    def predict_proba(self,X):
        probabilities = self.booster.predict(xgb.DMatrix(X))
        # binary:logistic returns a one-dimensional positive-class probability,
        # while sklearn calibration consistently expects one column per class.
        if self.num_class == 2 and probabilities.ndim == 1:
            return np.column_stack((1 - probabilities, probabilities))
        return probabilities
