import numpy as np
import pandas as pd
from model import Model
from sklearn import linear_model

class ElasticNet(Model):
    """
    Basic ElasticNet wrapper.
    """
    def __init__(self, log_data=False, name='ElasticNet', features=None, 
                 alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False,
                   max_iter=1000, tol=0.0001, random_state=0):
        self.model = None
        self.log_data = log_data
        self.name = name
        self.features = features
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept=fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def train(self, X_trn, y_trn, X_val=None, y_val=None):
        X_trn = X_trn.fillna(X_trn.median())
        if self.features is not None:
            X_trn = X_trn[self.features]
        if self.log_data:
            y_trn = np.log1p(y_trn)
        self.model = linear_model.ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, 
            fit_intercept=self.fit_intercept, normalize=self.normalize,
                   max_iter=self.max_iter, tol=self.tol, random_state=self.random_state)
        self.model.fit(X_trn, y_trn)

    def test(self, X_test, y_test=None):
        X_test = X_test.fillna(X_test.median()).fillna(0)
        if self.features is not None:
            X_test = X_test[self.features]
        preds = self.model.predict(X_test)
        if self.log_data:
            preds = np.exp(preds) - 1
        return preds