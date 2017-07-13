import numpy as np
import pandas as pd
from model import Model
from sklearn import tree

SEED = 0

class DecisionTreeRegressor(Model):
    """
    Basic ElasticNet wrapper.
    """
    def __init__(self, log_data=False, name='DecisionTreeRegressor', features=None):
        self.model = None
        self.log_data = log_data
        self.name = name
        self.features = features

    def train(self, X_trn, y_trn, X_val=None, y_val=None):
        X_trn = X_trn.fillna(X_trn.median())
        if self.features is not None:
            X_trn = X_trn[self.features]
        if self.log_data:
            y_trn = np.log1p(y_trn)
        self.model = tree.DecisionTreeRegressor(max_features=100, max_depth=4, min_samples_split=4,
            random_state=SEED)
        self.model.fit(X_trn, y_trn)

    def test(self, X_test, y_test=None):
        X_test = X_test.fillna(X_test.median()).fillna(0)
        if self.features is not None:
            X_test = X_test[self.features]
        preds = self.model.predict(X_test)
        if self.log_data:
            preds = np.exp(preds) - 1
        return preds