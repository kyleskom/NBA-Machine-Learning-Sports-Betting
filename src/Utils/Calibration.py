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
        return self.booster.predict(xgb.DMatrix(X))
